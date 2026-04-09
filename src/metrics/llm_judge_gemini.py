"""
llm_judge_gemini.py
====================
LLM-as-Judge evaluation of anonymized text quality using Google Gemini Flash.

Implements two evaluation dimensions:

  A) UTILITY (following Staab et al., ICLR 2025):
     - Readability (1-10): How readable is the anonymized text on its own?
     - Meaning (1-10): How well does the anonymized text preserve original meaning?
     - Hallucination (0/1): Does the anonymized text contain invented information?

  B) PRIVACY (following Schiezaro et al., Frontiers in Public Health 2026):
     - Anonymization Quality (1-10): How effectively were PII entities detected and masked?
     - Re-identification Risk (1-10): How difficult is it to re-identify individuals
       from the anonymized text? (10 = very difficult, well protected)

Results are reported per pipeline, per complexity level, and overall.

Usage:
    1. Set your Gemini API key: set GEMINI_API_KEY=your_key_here
       (or set GOOGLE_API_KEY=your_key_here)
    2. Adjust paths in USER SETTINGS below
    3. Run: python llm_judge_gemini.py

Requirements:
    pip install google-genai tqdm

"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

# Gemini model to use as judge
GEMINI_MODEL = "gemini-2.5-flash"   # or "gemini-3-flash-preview" for latest

# Paths
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_PATH  = os.path.join(BASE_DIR, "data", "label_studio", "20260302_Export_Label_Studio_Client_Notes.json")
SPLIT_IDS   = os.path.join(BASE_DIR, "results", "bert_finetuned", "split_ids.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "results", "llm_judge_gemini")

# Tag-and-replace predictions: {id, entities}
TAG_REPLACE_PREDICTIONS = {
    # Classical baselines
    "spaCy + Regex":                  os.path.join(BASE_DIR, "results", "classical_baselines", "spacy", "spacy_predictions.json"),
    "BERT (pretrained) + Regex":      os.path.join(BASE_DIR, "results", "classical_baselines", "bert", "bert_predictions.json"),
    "BERT Fine-Tuned":                os.path.join(BASE_DIR, "results", "bert_finetuned", "bert_finetuned_predictions.json"),
    "Presidio":                       os.path.join(BASE_DIR, "results", "presidio_baseline", "presidio_predictions.json"),
    # LLM tag-and-replace
    "LLM Llama-3 [few-shot +verify]": os.path.join(BASE_DIR, "results", "llm_baselines", "llm_meta_llama_3_8b_instruct_few_shot_verified_predictions.json"),
    "LLM Qwen2.5 [few-shot +verify]": os.path.join(BASE_DIR, "results", "llm_baselines", "llm_qwen2.5_7b_instruct_few_shot_verified_predictions.json"),
    "LLM SauerkrautLM [few-shot +verify]": os.path.join(BASE_DIR, "results", "llm_baselines", "llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_verified_predictions.json"),
    # LLM fine-tuned (QLoRA)
    "LLM Llama-3 [fine-tuned]":        os.path.join(BASE_DIR, "results", "llm_finetuned", "llm_finetuned_meta_llama_3_8b_instruct", "llm_finetuned_meta_llama_3_8b_instruct_predictions.json"),
    # "LLM Qwen2.5 [fine-tuned]":        os.path.join(BASE_DIR, "results", "llm_finetuned", "llm_finetuned_qwen2.5_7b_instruct", "llm_finetuned_qwen2.5_7b_instruct_predictions.json"),
    # "LLM SauerkrautLM [fine-tuned]":   os.path.join(BASE_DIR, "results", "llm_finetuned", "llm_finetuned_llama_3.1_sauerkrautlm_8b_instruct", "llm_finetuned_llama_3.1_sauerkrautlm_8b_instruct_predictions.json"),
}

# Prompt-based rewrite predictions: {id, rewritten_text}
PROMPT_REWRITE_PREDICTIONS = {
    "LLM Llama-3 [prompt few-shot]":     os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_meta_llama_3_8b_instruct_few_shot_predictions.json"),
    "LLM Qwen2.5 [prompt few-shot]":     os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_qwen2.5_7b_instruct_few_shot_predictions.json"),
    "LLM SauerkrautLM [prompt few-shot]": os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_llama_3.1_sauerkrautlm_8b_instruct_few_shot_predictions.json"),
}

MAX_DOCS = None    # None for full run, small int for testing
SAMPLE_SIZE = None  # None = evaluate ALL 638 test documents (recommended for thesis)
SEED     = 42

# Rate limiting: pause between API calls (seconds)
# Free tier: 15 RPM → need ~4s between calls
# Paid tier: 2000 RPM → 0.5s is fine
API_DELAY = 0.5    # Set to 4.5 for free tier, 0.5 for paid tier


# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import re
import time
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from tqdm import tqdm

from evaluation_utils import ALL_LABELS, load_label_studio_export


# =====================================================================
#  1. GEMINI CLIENT SETUP
# =====================================================================

def create_gemini_client():
    """Create a Gemini API client using the google-genai SDK."""
    from google import genai

    # Try to load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        # Look for .env in common locations
        for env_path in [os.path.join(BASE_DIR, ".env"), ".env", "../.env", "../../.env"]:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"  Loaded environment from: {env_path}")
                break
    except ImportError:
        pass  # python-dotenv not installed, rely on system env vars

    # The SDK automatically picks up GEMINI_API_KEY or GOOGLE_API_KEY
    # from environment variables
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n"
            "  Windows:  set GEMINI_API_KEY=your_key_here\n"
            "  Linux:    export GEMINI_API_KEY=your_key_here\n"
            "  Or add it to .env file (in project root) as: GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=api_key)
    return client


def call_gemini(client, prompt: str, model: str = GEMINI_MODEL, temperature: float = 0.1) -> str:
    """
    Call the Gemini API with a prompt and return the text response.
    Includes retry logic for rate limiting.
    """
    from google.genai import types

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            return response.text.strip() if response.text else ""

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                wait_time = (attempt + 1) * 5
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return ""

    return ""


# =====================================================================
#  2. ANONYMIZED TEXT CONSTRUCTION
# =====================================================================

def build_anonymized_text(text: str, pred_entities: List[Dict]) -> str:
    """Replace predicted entity spans with [LABEL] placeholders."""
    sorted_ents = sorted(pred_entities, key=lambda e: e["start"], reverse=True)
    anon = text
    for ent in sorted_ents:
        placeholder = f"[{ent['label']}]"
        anon = anon[:ent["start"]] + placeholder + anon[ent["end"]:]
    return anon


# =====================================================================
#  3. LLM-AS-JUDGE PROMPTS
# =====================================================================

def build_judge_prompt(original_text: str, anonymized_text: str) -> str:
    """
    Build the utility + privacy judge prompt.
    Utility dimensions follow Staab et al. (ICLR 2025).
    Privacy dimensions follow Schiezaro et al. (Frontiers in Public Health 2026).
    """
    return (
        "You are an expert text quality and privacy evaluator for German-language financial "
        "communications from a Swiss banking context. You evaluate anonymized texts "
        "by comparing them to their originals.\n\n"
        "Score the anonymized text on FIVE dimensions:\n\n"
        "── UTILITY DIMENSIONS ──\n\n"
        "READABILITY (1-10): How readable and understandable is the anonymized text on its own, "
        "WITHOUT seeing the original? Could a bank employee read and understand it?\n"
        "  1 = Unreadable, key information missing or replaced with incomprehensible tokens\n"
        "  3 = Partially readable but significant effort needed to understand\n"
        "  5 = Readable but noticeable gaps or awkward phrasing from anonymization\n"
        "  7 = Good readability with only minor issues\n"
        "  10 = Perfectly readable, natural German text\n\n"
        "MEANING (1-10): How well does the anonymized text preserve the original meaning "
        "and informational content? Focus on non-personal information: business context, "
        "transaction details, requests, action items.\n"
        "  1 = Completely different, all meaning lost\n"
        "  3 = Some meaning recognizable but heavily impacted\n"
        "  5 = Core meaning preserved but noticeably impacted\n"
        "  7 = Most meaning preserved with only minor losses\n"
        "  10 = All non-personal meaning perfectly preserved\n\n"
        "HALLUCINATION (0 or 1): Does the anonymized text contain any NEW information "
        "that was NOT in the original? (0 = contains new/invented info, 1 = no hallucination)\n\n"
        "── PRIVACY DIMENSIONS ──\n\n"
        "ANONYMIZATION_QUALITY (1-10): How effectively did the anonymization identify and "
        "mask ALL personally identifiable information (PII)? Check for names, phone numbers, "
        "email addresses, IBANs, dates of birth, ages, locations, company names, job titles, "
        "nationalities, and education details.\n"
        "  1 = Almost no PII was masked, most identifiers remain visible\n"
        "  3 = Some PII was masked but major identifiers were missed\n"
        "  5 = Most common PII was masked but some identifiers were overlooked\n"
        "  7 = Nearly all PII was masked with only minor omissions\n"
        "  10 = All PII was correctly identified and masked, no identifiers remain\n\n"
        "REIDENTIFICATION_RISK (1-10): How difficult would it be for someone to identify "
        "the specific individuals, companies, or locations from the anonymized text alone? "
        "Consider both direct identifiers AND indirect clues (e.g., unique combinations of "
        "job title + company type + location that could narrow down a person).\n"
        "  1 = Trivially easy to re-identify (most PII still present or obvious from context)\n"
        "  3 = Possible to narrow down with some effort using remaining contextual clues\n"
        "  5 = Difficult but some indirect clues remain that could aid re-identification\n"
        "  7 = Very difficult, only vague statistical guesses possible\n"
        "  10 = Impossible to re-identify, all identifying and quasi-identifying information removed\n\n"
        "═══════════════════════════════════════════════\n"
        f"ORIGINAL TEXT:\n{original_text}\n\n"
        "═══════════════════════════════════════════════\n"
        f"ANONYMIZED TEXT:\n{anonymized_text}\n\n"
        "═══════════════════════════════════════════════\n\n"
        "Respond ONLY with a JSON object, no other text:\n"
        '{"readability": <1-10>, "meaning": <1-10>, "hallucination": <0 or 1>, '
        '"anonymization_quality": <1-10>, "reidentification_risk": <1-10>, '
        '"readability_reason": "<1 sentence>", "meaning_reason": "<1 sentence>", '
        '"hallucination_reason": "<1 sentence>", '
        '"anonymization_quality_reason": "<1 sentence>", "reidentification_risk_reason": "<1 sentence>"}'
    )


# =====================================================================
#  4. RESPONSE PARSING
# =====================================================================

def parse_judge_response(response: str) -> Dict:
    """Parse the judge's JSON response into scores."""
    result = {
        "readability": None,
        "meaning": None,
        "hallucination": None,
        "anonymization_quality": None,
        "reidentification_risk": None,
        "readability_reason": "",
        "meaning_reason": "",
        "hallucination_reason": "",
        "anonymization_quality_reason": "",
        "reidentification_risk_reason": "",
        "raw_response": response,
    }

    if not response:
        return result

    # Try to extract JSON from response
    try:
        # Remove markdown code fences if present
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        # Find JSON object (greedy match from first { to last })
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())

            if "readability" in parsed:
                result["readability"] = max(1, min(10, int(parsed["readability"])))
            if "meaning" in parsed:
                result["meaning"] = max(1, min(10, int(parsed["meaning"])))
            if "hallucination" in parsed:
                result["hallucination"] = int(parsed["hallucination"])
            if "anonymization_quality" in parsed:
                result["anonymization_quality"] = max(1, min(10, int(parsed["anonymization_quality"])))
            if "reidentification_risk" in parsed:
                result["reidentification_risk"] = max(1, min(10, int(parsed["reidentification_risk"])))
            result["readability_reason"] = str(parsed.get("readability_reason", ""))
            result["meaning_reason"] = str(parsed.get("meaning_reason", ""))
            result["hallucination_reason"] = str(parsed.get("hallucination_reason", ""))
            result["anonymization_quality_reason"] = str(parsed.get("anonymization_quality_reason", ""))
            result["reidentification_risk_reason"] = str(parsed.get("reidentification_risk_reason", ""))
            return result

    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex extraction
    r_match = re.search(r'"?readability"?\s*[:=]\s*(\d+)', response, re.IGNORECASE)
    m_match = re.search(r'"?meaning"?\s*[:=]\s*(\d+)', response, re.IGNORECASE)
    h_match = re.search(r'"?hallucination"?\s*[:=]\s*(\d)', response, re.IGNORECASE)
    aq_match = re.search(r'"?anonymization_quality"?\s*[:=]\s*(\d+)', response, re.IGNORECASE)
    rr_match = re.search(r'"?reidentification_risk"?\s*[:=]\s*(\d+)', response, re.IGNORECASE)

    if r_match:
        result["readability"] = max(1, min(10, int(r_match.group(1))))
    if m_match:
        result["meaning"] = max(1, min(10, int(m_match.group(1))))
    if h_match:
        result["hallucination"] = int(h_match.group(1))
    if aq_match:
        result["anonymization_quality"] = max(1, min(10, int(aq_match.group(1))))
    if rr_match:
        result["reidentification_risk"] = max(1, min(10, int(rr_match.group(1))))

    return result


# =====================================================================
#  5. EVALUATION PIPELINE
# =====================================================================

def evaluate_pipeline(
    client,
    gold_records: List[Dict],
    anonymized_texts: Dict[int, str],
    pipeline_name: str,
) -> Dict:
    """
    Run LLM-as-judge evaluation on all documents for one pipeline.
    """
    results = []
    parse_failures = 0

    for gold in tqdm(gold_records, desc=f"Judging [{pipeline_name}]"):
        doc_id = gold["id"]
        anon_text = anonymized_texts.get(doc_id, "")
        if not anon_text:
            continue

        prompt = build_judge_prompt(gold["text"], anon_text)
        response = call_gemini(client, prompt)
        scores = parse_judge_response(response)

        if scores["readability"] is None or scores["meaning"] is None:
            parse_failures += 1

        results.append({
            "id": doc_id,
            "complexity": gold.get("meta_temp", "Unknown"),
            **scores,
        })

        # Rate limiting
        if API_DELAY > 0:
            time.sleep(API_DELAY)

    if parse_failures > 0:
        print(f"  Warning: {parse_failures}/{len(results)} responses failed to parse")

    # Aggregate
    aggregated = _aggregate_scores(results, pipeline_name)

    return {
        "pipeline": pipeline_name,
        "per_document": results,
        "aggregated": aggregated,
        "parse_failures": parse_failures,
    }


def _aggregate_scores(results: List[Dict], pipeline_name: str) -> Dict:
    """Aggregate per-document scores into means and standard deviations."""
    def _mean(vals):
        return round(sum(vals) / max(len(vals), 1), 3)

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return round((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 3)

    def _median(vals):
        if not vals:
            return 0
        s = sorted(vals)
        n = len(s)
        if n % 2 == 0:
            return round((s[n // 2 - 1] + s[n // 2]) / 2, 2)
        return s[n // 2]

    # Filter valid scores
    valid_read = [r["readability"] for r in results if r["readability"] is not None]
    valid_mean = [r["meaning"] for r in results if r["meaning"] is not None]
    valid_hall = [r["hallucination"] for r in results if r["hallucination"] is not None]
    valid_aq = [r["anonymization_quality"] for r in results if r["anonymization_quality"] is not None]
    valid_rr = [r["reidentification_risk"] for r in results if r["reidentification_risk"] is not None]

    overall = {
        "count": len(results),
        "readability_mean": _mean(valid_read),
        "readability_median": _median(valid_read),
        "readability_std": _std(valid_read),
        "meaning_mean": _mean(valid_mean),
        "meaning_median": _median(valid_mean),
        "meaning_std": _std(valid_mean),
        "hallucination_rate": round(1 - _mean(valid_hall), 3) if valid_hall else None,
        "anonymization_quality_mean": _mean(valid_aq),
        "anonymization_quality_median": _median(valid_aq),
        "anonymization_quality_std": _std(valid_aq),
        "reidentification_risk_mean": _mean(valid_rr),
        "reidentification_risk_median": _median(valid_rr),
        "reidentification_risk_std": _std(valid_rr),
    }

    # Per complexity
    by_complexity = defaultdict(list)
    for r in results:
        by_complexity[r["complexity"]].append(r)

    per_complexity = {}
    for level in ["Low", "Medium", "High"]:
        group = by_complexity.get(level, [])
        if group:
            gr = [r["readability"] for r in group if r["readability"] is not None]
            gm = [r["meaning"] for r in group if r["meaning"] is not None]
            gh = [r["hallucination"] for r in group if r["hallucination"] is not None]
            gaq = [r["anonymization_quality"] for r in group if r["anonymization_quality"] is not None]
            grr = [r["reidentification_risk"] for r in group if r["reidentification_risk"] is not None]
            per_complexity[level] = {
                "count": len(group),
                "readability_mean": _mean(gr),
                "readability_median": _median(gr),
                "readability_std": _std(gr),
                "meaning_mean": _mean(gm),
                "meaning_median": _median(gm),
                "meaning_std": _std(gm),
                "hallucination_rate": round(1 - _mean(gh), 3) if gh else None,
                "anonymization_quality_mean": _mean(gaq),
                "anonymization_quality_median": _median(gaq),
                "anonymization_quality_std": _std(gaq),
                "reidentification_risk_mean": _mean(grr),
                "reidentification_risk_median": _median(grr),
                "reidentification_risk_std": _std(grr),
            }

    return {
        "pipeline": pipeline_name,
        "overall": overall,
        "by_complexity": per_complexity,
    }


# =====================================================================
#  6. REPORT GENERATION
# =====================================================================

def format_report(all_results: List[Dict]) -> str:
    """Generate a comprehensive LLM-as-Judge report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  LLM-AS-JUDGE EVALUATION REPORT (Gemini Flash)")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Judge Model: {GEMINI_MODEL}")
    lines.append(f"  Utility: Staab et al. (ICLR 2025)")
    lines.append(f"  Privacy: Schiezaro et al. (Frontiers in Public Health 2026)")
    lines.append("=" * 78)

    # ── Summary Table ──
    lines.append(f"\n{'#' * 78}")
    lines.append("  SUMMARY: Utility & Privacy Scores")
    lines.append(f"{'#' * 78}")
    lines.append("")
    lines.append(f"  {'Pipeline':<38} {'Read':>6} {'Mean':>6} {'Halluc':>7} {'AnonQ':>6} {'ReID':>6} {'n':>5}")
    lines.append(f"  {'':38} {'(avg)':>6} {'(avg)':>6} {'rate':>7} {'(avg)':>6} {'(avg)':>6}")
    lines.append(f"  {'-' * 79}")

    for result in all_results:
        name = result["pipeline"]
        o = result["aggregated"]["overall"]
        hall_str = f"{o['hallucination_rate']:.1%}" if o["hallucination_rate"] is not None else "n/a"
        lines.append(
            f"  {name:<38} "
            f"{o['readability_mean']:>6.2f} "
            f"{o['meaning_mean']:>6.2f} "
            f"{hall_str:>7} "
            f"{o['anonymization_quality_mean']:>6.2f} "
            f"{o['reidentification_risk_mean']:>6.2f} "
            f"{o['count']:>5}"
        )

    lines.append(f"  {'-' * 79}")
    lines.append("  Read/Mean/AnonQ/ReID: 1-10 (higher = better)")
    lines.append("  Hallucination rate: % of texts with invented information (lower = better)")

    # ── Per-Complexity Breakdown ──
    for level in ["Low", "Medium", "High"]:
        has_data = any(
            level in r["aggregated"]["by_complexity"] for r in all_results
        )
        if not has_data:
            continue

        lines.append(f"\n  >>> {level} Complexity <<<")
        lines.append(f"  {'Pipeline':<38} {'Read':>6} {'Mean':>6} {'AnonQ':>6} {'ReID':>6} {'n':>5}")
        lines.append(f"  {'-' * 69}")

        for result in all_results:
            name = result["pipeline"]
            comp = result["aggregated"]["by_complexity"].get(level, {})
            if comp:
                lines.append(
                    f"  {name:<38} "
                    f"{comp['readability_mean']:>6.2f} "
                    f"{comp['meaning_mean']:>6.2f} "
                    f"{comp['anonymization_quality_mean']:>6.2f} "
                    f"{comp['reidentification_risk_mean']:>6.2f} "
                    f"{comp['count']:>5}"
                )

    # ── Detailed Per-Pipeline Stats ──
    lines.append(f"\n\n{'#' * 78}")
    lines.append("  DETAILED PER-PIPELINE STATISTICS")
    lines.append(f"{'#' * 78}")

    for result in all_results:
        name = result["pipeline"]
        o = result["aggregated"]["overall"]
        lines.append(f"\n  {'=' * 70}")
        lines.append(f"  Pipeline: {name}")
        lines.append(f"  {'=' * 70}")
        lines.append(f"  Documents: {o['count']} | Parse failures: {result['parse_failures']}")
        lines.append(f"")
        lines.append(f"  Readability: {o['readability_mean']:.2f} ± {o['readability_std']:.2f} (median: {o['readability_median']:.1f})")
        lines.append(f"  Meaning:     {o['meaning_mean']:.2f} ± {o['meaning_std']:.2f} (median: {o['meaning_median']:.1f})")
        if o["hallucination_rate"] is not None:
            lines.append(f"  Hallucination rate: {o['hallucination_rate']:.1%}")
        lines.append(f"")
        lines.append(f"  Anonymization Quality: {o['anonymization_quality_mean']:.2f} ± {o['anonymization_quality_std']:.2f} (median: {o['anonymization_quality_median']:.1f})")
        lines.append(f"  Re-identification Risk: {o['reidentification_risk_mean']:.2f} ± {o['reidentification_risk_std']:.2f} (median: {o['reidentification_risk_median']:.1f})")

        for level in ["Low", "Medium", "High"]:
            comp = result["aggregated"]["by_complexity"].get(level, {})
            if comp:
                lines.append(f"")
                lines.append(f"  {level} (n={comp['count']}):")
                lines.append(f"    Readability: {comp['readability_mean']:.2f} ± {comp['readability_std']:.2f} (median: {comp['readability_median']:.1f})")
                lines.append(f"    Meaning:     {comp['meaning_mean']:.2f} ± {comp['meaning_std']:.2f} (median: {comp['meaning_median']:.1f})")
                lines.append(f"    Anon Quality: {comp['anonymization_quality_mean']:.2f} ± {comp['anonymization_quality_std']:.2f}")
                lines.append(f"    Re-ID Risk:   {comp['reidentification_risk_mean']:.2f} ± {comp['reidentification_risk_std']:.2f}")

    return "\n".join(lines)


# =====================================================================
#  7. STRATIFIED SAMPLING
# =====================================================================

def stratified_sample(records: List[Dict], n: int) -> List[Dict]:
    """
    Sample n records stratified by complexity level (meta_temp).
    Ensures equal representation of Low, Medium, High complexity.
    """
    by_complexity = defaultdict(list)
    for r in records:
        by_complexity[r.get("meta_temp", "Unknown")].append(r)

    # Calculate per-group sample size
    groups = sorted(by_complexity.keys())
    per_group = n // len(groups)
    remainder = n % len(groups)

    sampled = []
    for i, group in enumerate(groups):
        group_records = by_complexity[group]
        group_n = per_group + (1 if i < remainder else 0)
        group_n = min(group_n, len(group_records))

        random.shuffle(group_records)
        sampled.extend(group_records[:group_n])

    random.shuffle(sampled)

    # Report distribution
    dist = defaultdict(int)
    for r in sampled:
        dist[r.get("meta_temp", "Unknown")] += 1
    print(f"  Sample distribution: {dict(dist)}")

    return sampled


# =====================================================================
#  8. MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    # ── Setup Gemini client ──
    print(f"Setting up Gemini client (model: {GEMINI_MODEL})...")
    client = create_gemini_client()

    # Test API connection
    test_response = call_gemini(client, "Reply with only: OK")
    if not test_response:
        print("ERROR: Could not connect to Gemini API. Check your API key.")
        return
    print(f"  API connection OK (response: {test_response[:20]})")

    # ── Load gold standard ──
    print(f"\nLoading data from: {INPUT_PATH}")
    gold_records = load_label_studio_export(INPUT_PATH)
    print(f"  Total records: {len(gold_records)}")

    if SPLIT_IDS:
        with open(SPLIT_IDS, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        test_ids = set(split_info["test_ids"])
        gold_records = [r for r in gold_records if r["id"] in test_ids]
        print(f"  Filtered to test split: {len(gold_records)} records")

    if MAX_DOCS:
        gold_records = gold_records[:MAX_DOCS]
        print(f"  Limited to {len(gold_records)} documents")

    if SAMPLE_SIZE and SAMPLE_SIZE < len(gold_records):
        gold_records = stratified_sample(gold_records, SAMPLE_SIZE)
        print(f"  Stratified sample: {len(gold_records)} documents "
              f"(balanced by complexity)")

    gold_by_id = {r["id"]: r for r in gold_records}

    # ── Build anonymized texts ──
    pipelines = {}

    # Tag-and-replace
    for name, path in TAG_REPLACE_PREDICTIONS.items():
        if path is None or not os.path.exists(path):
            print(f"  Skipping {name} (not found: {path})")
            continue
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        pred_by_id = {r["id"]: r.get("entities", []) for r in preds}
        anon_texts = {}
        for gold in gold_records:
            doc_id = gold["id"]
            ents = pred_by_id.get(doc_id, [])
            anon_texts[doc_id] = build_anonymized_text(gold["text"], ents)
        pipelines[name] = anon_texts
        print(f"  Loaded {name}: {len(anon_texts)} docs")

    # Prompt-based rewrites
    for name, path in PROMPT_REWRITE_PREDICTIONS.items():
        if path is None or not os.path.exists(path):
            print(f"  Skipping {name} (not found: {path})")
            continue
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        anon_texts = {r["id"]: r.get("rewritten_text", "") for r in preds
                      if r["id"] in gold_by_id}
        pipelines[name] = anon_texts
        print(f"  Loaded {name}: {len(anon_texts)} docs")

    if not pipelines:
        print("No pipelines found. Check paths.")
        return

    # ── Run evaluation ──
    all_results = []

    for pipeline_name, anon_texts in pipelines.items():
        # Skip if results already exist (resume after interruption)
        safe_name = pipeline_name.lower().replace(" ", "_").replace("+", "").replace("[", "").replace("]", "")
        per_doc_path = os.path.join(OUTPUT_DIR, f"{safe_name}_judge_scores.json")

        if os.path.exists(per_doc_path):
            print(f"\n  → SKIPPING {pipeline_name} (already exists: {per_doc_path})")
            with open(per_doc_path, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)
            existing_agg = _aggregate_scores(existing_docs, pipeline_name)
            all_results.append({
                "pipeline": pipeline_name,
                "per_document": existing_docs,
                "aggregated": existing_agg,
                "parse_failures": sum(1 for d in existing_docs if d.get("readability") is None),
            })
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {pipeline_name}")
        print(f"{'=' * 60}")

        result = evaluate_pipeline(client, gold_records, anon_texts, pipeline_name)
        all_results.append(result)

        # Save per-pipeline results immediately (in case of interruption)
        # Save without raw_response to keep file size manageable
        slim_docs = [{k: v for k, v in d.items() if k != "raw_response"} for d in result["per_document"]]
        with open(per_doc_path, "w", encoding="utf-8") as f:
            json.dump(slim_docs, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {per_doc_path}")

    # ── Generate report ──
    report = format_report(all_results)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "llm_judge_gemini_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_path}")

    # Aggregated JSON
    agg = {r["pipeline"]: r["aggregated"] for r in all_results}
    agg_path = os.path.join(OUTPUT_DIR, "llm_judge_gemini_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"  Aggregated: {agg_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()