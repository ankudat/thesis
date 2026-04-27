"""
llm_judge_gemini.py
====================
LLM-as-Judge evaluation of anonymized text quality using Google Gemini Flash,
plus an empirical adversarial inference attack run alongside the judge.

Implements two complementary evaluations:

  ─── (1) LLM-AS-JUDGE: SUBJECTIVE RATINGS (judge sees BOTH original and anonymized) ───

  A) UTILITY (following Staab et al., ICLR 2025):
     - Readability (1-10): How readable is the anonymized text on its own?
     - Meaning (1-10): How well does the anonymized text preserve original meaning?
     - Hallucination (0/1): Does the anonymized text contain invented information?

  B) PRIVACY (following Schiezaro et al., Frontiers in Public Health 2026):
     - Anonymization Quality (1-10): How effectively were PII entities detected and masked?
     - Re-identification Risk (1-10): How difficult is it to re-identify individuals
       from the anonymized text? (10 = very difficult, well protected)

  ─── (2) ADVERSARIAL INFERENCE ATTACK: EMPIRICAL MEASUREMENT ───
        (following Staab et al., ICLR 2025: "Large Language Models are Advanced Anonymizers")

  C) ATTACK PROTOCOL:
     1. Ground-truth extraction: an LLM extracts true attribute values
        (age, location, occupation, education, nationality, organization)
        from the ORIGINAL text. Cached on disk – pipeline-independent.
        Attribute set is aligned with the thesis 12-entity schema
        (Tier-3 quasi-identifiers + LOC + ORG from Tier-1).
     2. Adversarial inference: an LLM "adversary" sees ONLY the anonymized
        text and is asked to infer the same attributes via chain-of-thought.
        It returns a top-3 guess list per attribute plus a confidence score.
     3. LLM match-judging: a third call compares ground truth vs. the
        attacker's top-3 using a tolerance rubric (synonyms, ±5y for age, etc.)
        and returns MATCH / NO_MATCH / SKIP per attribute.

     Reported as per-attribute success rate (lower = better anonymization)
     and "any-attribute leaked" rate, broken down per complexity level.

Results are reported per pipeline, per complexity level, and overall.

Usage:
    1. Set your Gemini API key: set GEMINI_API_KEY=your_key_here
       (or set GOOGLE_API_KEY=your_key_here)
    2. Adjust paths in USER SETTINGS below
    3. Run: python llm_judge_gemini.py
       (judge + attack both run by default; toggle RUN_ATTACK to disable)

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
    # External API baseline
    "Datenwertsch\u00f6pfung API":        os.path.join(BASE_DIR, "results", "datenwertschoepfung_baseline", "datenwertschoepfung_predictions.json"),
}

MAX_DOCS = None    # None for full run, small int for testing
SAMPLE_SIZE = None  # None = evaluate ALL 638 test documents (recommended for thesis)
SEED     = 42

# Rate limiting: pause between API calls (seconds)
# Free tier: 15 RPM → need ~4s between calls
# Paid tier: 2000 RPM → 0.5s is fine
API_DELAY = 0.5    # Set to 4.5 for free tier, 0.5 for paid tier

# ── Adversarial inference attack settings (Staab et al., ICLR 2025) ──
# When True, runs the empirical attack alongside the LLM-as-judge ratings.
# The attack adds ~3 API calls per (document, pipeline) pair:
#   - 1 ground-truth extraction (cached: only once per document, not per pipeline)
#   - 1 adversarial inference call (anonymized text only)
#   - 1 LLM match-judging call (compares GT vs. attacker top-3 guesses)
RUN_ATTACK = True

# Quasi-identifier attributes the adversary attempts to infer.
# Aligned with the thesis 12-entity schema (TIER1_DIRECT_NER ∪ TIER3_QUASI):
#   age          ↔ AGE     (Tier 3)
#   location     ↔ LOC     (Tier 1)
#   occupation   ↔ JOB     (Tier 3)
#   education    ↔ EDU     (Tier 3)
#   nationality  ↔ NATION  (Tier 3)
#   organization ↔ ORG     (Tier 1) — employer / primary affiliation
# (Sex is intentionally excluded: it is not part of the 12-entity schema and
#  no pipeline attempts to mask it, so attacking it would yield ~100% leakage
#  with no signal for comparing pipelines.)
ATTACK_ATTRIBUTES = ["age", "location", "occupation", "education", "nationality", "organization"]

# Output paths for the attack (separate from the judge outputs)
ATTACK_OUTPUT_DIR   = os.path.join(BASE_DIR, "results", "llm_judge_gemini", "adversarial_attack")
GROUND_TRUTH_CACHE  = os.path.join(ATTACK_OUTPUT_DIR, "ground_truth_attributes.json")


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
#  8. ADVERSARIAL INFERENCE ATTACK (Staab et al., ICLR 2025)
# =====================================================================
#
# Implements the adversarial-inference protocol from
#   Staab, R., Vero, M., Balunović, M., Vechev, M.
#   "Large Language Models are Advanced Anonymizers"
#   ICLR 2025  (https://arxiv.org/abs/2402.13846)
#
# Goal: empirically measure how much PII a strong LLM can recover from an
# anonymized text *alone*, i.e. without ever seeing the original. This is
# distinct from the rated `reidentification_risk` score in section 3, which
# is a SUBJECTIVE judgement made by an LLM that sees BOTH texts.
#
# Three-step protocol per document:
#   1) Extract ground-truth attribute values from the ORIGINAL text  (cached)
#   2) Adversarial inference of the same attributes from the ANONYMIZED text
#      (chain-of-thought, top-3 guesses, confidence 1-5)
#   3) LLM match-judging of attacker's top-3 vs. ground truth, returning
#      MATCH / NO_MATCH / SKIP per attribute under a tolerance rubric.
#
# An attribute is "successfully attacked" iff MATCH. SKIP is used when the
# original text never mentioned the attribute (attack not evaluable for it).

def build_ground_truth_prompt(original_text: str) -> str:
    """Extract true attribute values from the ORIGINAL (un-anonymized) text."""
    return (
        "You are an information-extraction assistant for German banking texts.\n"
        "From the text below, extract attribute values about the main client/individual.\n"
        "Use exactly null (no quotes) when an attribute is NOT present or cannot be inferred.\n\n"
        "Attributes to extract:\n"
        "- age: integer age (e.g., 47); null if not stated\n"
        "- location: city, town, region, or canton (e.g., Zürich, Bern, Aargau); null if absent\n"
        "- occupation: job title or profession (e.g., Anwalt, Software Engineer, Pflegerin); null if absent\n"
        '- education: highest education stated (e.g., "Lehre", "Bachelor", "Master", "Doktorat"); null if absent\n'
        "- nationality: country/region of origin or citizenship (e.g., Schweiz, Deutschland, Italien); null if absent\n"
        "- organization: employer or primary organizational affiliation of the client\n"
        "  (e.g., Novartis, ABB, UBS, Stadt Zürich, Universität Zürich); null if absent.\n"
        "  Do NOT use the bank itself as the organization — extract the CLIENT's employer/affiliation only.\n\n"
        "═══════════════════════════════════════════════\n"
        f"ORIGINAL TEXT:\n{original_text}\n"
        "═══════════════════════════════════════════════\n\n"
        "Respond ONLY with a JSON object, no other text:\n"
        '{"age": <int or null>, "location": <string or null>, '
        '"occupation": <string or null>, "education": <string or null>, '
        '"nationality": <string or null>, "organization": <string or null>}'
    )


def build_attack_prompt(anonymized_text: str) -> str:
    """
    Adversarial inference prompt (Staab et al., ICLR 2025).
    The adversary sees ONLY the anonymized text and must infer attributes
    of the underlying individual using contextual clues.
    """
    return (
        "You are an expert investigator attempting to re-identify the main "
        "individual from an anonymized German banking text. The text has had "
        "personally identifiable information replaced with placeholders such as "
        "[PER], [LOC], [ORG], [JOB], [DATE], [MONEY], [IBAN], [EMAIL], [PHONE], "
        "[AGE], [NATION], [EDU].\n\n"
        "You do NOT have access to the original text. Use ONLY what remains in "
        "the anonymized text — linguistic patterns, financial details, occupational "
        "hints, geographic references, monetary amounts, dates, behavioral clues, "
        "and any indirect signals. Even when explicit identifiers are masked, "
        "residual context often leaks information about the underlying person.\n\n"
        "INSTRUCTIONS:\n"
        "1. First, identify all clues in the text (linguistic register, financial\n"
        "   amounts, products mentioned, regional dialect/idioms, professional jargon,\n"
        "   context of the conversation, life events, etc.).\n"
        "2. For EACH attribute below, reason from those clues to your best estimates.\n"
        "3. Return your TOP-3 guesses ranked by likelihood (most likely first).\n"
        "4. Rate confidence 1–5 (5 = strong textual evidence, 1 = pure guess).\n"
        "5. Cite the specific clue(s) you used in 1 sentence.\n\n"
        "Attributes to infer:\n"
        "- age: integer age (e.g., 45) or tight range (e.g., 40-50)\n"
        "- location: city / town / region / canton\n"
        "- occupation: job title or profession\n"
        '- education: educational level (e.g., "Lehre", "Bachelor", "Master")\n'
        "- nationality: country / region of origin or citizenship\n"
        "- organization: employer or primary organizational affiliation of the underlying client\n"
        "  (e.g., Novartis, ABB, UBS, Stadt Zürich, Universität Zürich).\n"
        "  Do NOT name the bank that produced the text — guess the CLIENT's employer/affiliation.\n\n"
        "═══════════════════════════════════════════════\n"
        f"ANONYMIZED TEXT:\n{anonymized_text}\n"
        "═══════════════════════════════════════════════\n\n"
        "Respond ONLY with a JSON object, no other text:\n"
        "{\n"
        '  "age":          {"guesses": [<g1>, <g2>, <g3>], "confidence": <1-5>, "reasoning": "<1 sentence>"},\n'
        '  "location":     {"guesses": [<g1>, <g2>, <g3>], "confidence": <1-5>, "reasoning": "<1 sentence>"},\n'
        '  "occupation":   {"guesses": [<g1>, <g2>, <g3>], "confidence": <1-5>, "reasoning": "<1 sentence>"},\n'
        '  "education":    {"guesses": [<g1>, <g2>, <g3>], "confidence": <1-5>, "reasoning": "<1 sentence>"},\n'
        '  "nationality":  {"guesses": [<g1>, <g2>, <g3>], "confidence": <1-5>, "reasoning": "<1 sentence>"},\n'
        '  "organization": {"guesses": [<g1>, <g2>, <g3>], "confidence": <1-5>, "reasoning": "<1 sentence>"}\n'
        "}"
    )


def build_match_judge_prompt(ground_truth: Dict, attacker_guesses: Dict) -> str:
    """
    LLM-based match judge: decide whether ANY of the attacker's top-3 guesses
    semantically matches the ground-truth value, per attribute.
    """
    gt_str = json.dumps(ground_truth, ensure_ascii=False)
    guesses_str = json.dumps(
        {a: attacker_guesses.get(a, {}).get("guesses", []) for a in ATTACK_ATTRIBUTES},
        ensure_ascii=False,
    )
    return (
        "You are evaluating an adversarial inference attack against an anonymization "
        "system. For each attribute, the attacker provided up to 3 ranked guesses "
        "based on the anonymized text alone. Decide whether ANY of the attacker's "
        "guesses semantically matches the ground-truth value extracted from the "
        "original text.\n\n"
        "MATCHING RUBRIC (per attribute):\n"
        "- age:          ±5 years tolerance; overlapping range counts (e.g. 47 vs. 45-50 → MATCH).\n"
        "- location:     same city/town OR same canton OR a clearly equivalent variant\n"
        "                (e.g. Zürich ↔ Zurich ↔ ZH → MATCH). A strict regional ancestor\n"
        "                that is much broader than the GT does NOT match\n"
        "                (e.g. 'Schweiz' for GT 'Bern' → NO_MATCH).\n"
        "- occupation:   same general profession or close job category\n"
        "                (e.g. Anwalt ↔ Jurist ↔ lawyer → MATCH; Anwalt ↔ Lehrer → NO_MATCH).\n"
        "- education:    same educational level\n"
        "                (e.g. Bachelor ↔ Hochschulabschluss → MATCH; Lehre ↔ Master → NO_MATCH).\n"
        "- nationality:  same country or close regional grouping → MATCH.\n"
        "- organization: same employer/institution OR a clearly equivalent variant\n"
        "                (e.g. UBS ↔ UBS AG ↔ UBS Switzerland → MATCH;\n"
        "                ETH Zürich ↔ ETHZ → MATCH).\n"
        "                A pure industry/sector label that is much broader than the GT does NOT match\n"
        "                (e.g. 'Pharma-Konzern' for GT 'Novartis' → NO_MATCH; 'Bank' for GT 'UBS' → NO_MATCH).\n\n"
        "OUTPUT VALUES per attribute:\n"
        '- "MATCH"    — at least one of the top-3 attacker guesses matches GT under the rubric\n'
        '- "NO_MATCH" — none of the guesses match\n'
        '- "SKIP"     — GT is null (the original did not contain this attribute), so the attack is not evaluable\n\n'
        "═══════════════════════════════════════════════\n"
        f"GROUND TRUTH (from original text):\n{gt_str}\n"
        "═══════════════════════════════════════════════\n"
        f"ATTACKER'S TOP-3 GUESSES (from anonymized text):\n{guesses_str}\n"
        "═══════════════════════════════════════════════\n\n"
        "Respond ONLY with a JSON object, no other text:\n"
        '{"age": "MATCH"/"NO_MATCH"/"SKIP", "location": "MATCH"/"NO_MATCH"/"SKIP", '
        '"occupation": "MATCH"/"NO_MATCH"/"SKIP", "education": "MATCH"/"NO_MATCH"/"SKIP", '
        '"nationality": "MATCH"/"NO_MATCH"/"SKIP", "organization": "MATCH"/"NO_MATCH"/"SKIP"}'
    )


def _extract_json_obj(response: str) -> Optional[Dict]:
    """Pull the first JSON object out of a possibly-fenced model response."""
    if not response:
        return None
    cleaned = re.sub(r"```json\s*", "", response)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError):
        return None


def parse_ground_truth_response(response: str) -> Dict:
    """Parse the flat ground-truth attribute JSON."""
    parsed = _extract_json_obj(response) or {}
    out = {}
    for a in ATTACK_ATTRIBUTES:
        v = parsed.get(a)
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a", "unknown"}:
            v = None
        out[a] = v
    return out


def parse_attack_response(response: str) -> Dict:
    """Parse the adversary response into per-attribute {guesses, confidence, reasoning}."""
    parsed = _extract_json_obj(response) or {}
    out = {}
    for a in ATTACK_ATTRIBUTES:
        p = parsed.get(a, {})
        if isinstance(p, dict):
            guesses = p.get("guesses", [])
            if not isinstance(guesses, list):
                guesses = []
            out[a] = {
                "guesses": [str(g) for g in guesses if g is not None][:3],
                "confidence": p.get("confidence"),
                "reasoning": str(p.get("reasoning", "")),
            }
        else:
            out[a] = {"guesses": [], "confidence": None, "reasoning": ""}
    return out


def parse_match_response(response: str) -> Dict[str, str]:
    """Parse the match judge response into MATCH/NO_MATCH/SKIP per attribute."""
    parsed = _extract_json_obj(response) or {}
    out = {}
    for a in ATTACK_ATTRIBUTES:
        v = str(parsed.get(a, "SKIP")).upper().strip()
        if v not in {"MATCH", "NO_MATCH", "SKIP"}:
            v = "SKIP"
        out[a] = v
    return out


def extract_ground_truth_attributes(
    client,
    gold_records: List[Dict],
    cache_path: str,
) -> Dict[int, Dict]:
    """
    Extract true attribute values from each original text. Cached on disk because
    they depend only on the original (not on any pipeline) and can be reused
    across pipelines and across re-runs.
    """
    cache: Dict[int, Dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = {int(k): v for k, v in json.load(f).items()}
        print(f"  Loaded ground-truth cache: {len(cache)} entries from {cache_path}")

    todo = [g for g in gold_records if g["id"] not in cache]
    if not todo:
        print(f"  All {len(gold_records)} ground-truth attributes already cached")
        return cache

    print(f"  Extracting ground-truth attributes for {len(todo)} documents...")
    for gold in tqdm(todo, desc="Ground truth"):
        prompt = build_ground_truth_prompt(gold["text"])
        response = call_gemini(client, prompt)
        cache[gold["id"]] = parse_ground_truth_response(response)
        if API_DELAY > 0:
            time.sleep(API_DELAY)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cache.items()}, f, indent=2, ensure_ascii=False)
    print(f"  Saved ground-truth cache: {cache_path}")

    return cache


def attack_pipeline(
    client,
    gold_records: List[Dict],
    anonymized_texts: Dict[int, str],
    ground_truth: Dict[int, Dict],
    pipeline_name: str,
) -> Dict:
    """Run adversarial inference + match judging on every doc for one pipeline."""
    results = []
    parse_failures = 0

    for gold in tqdm(gold_records, desc=f"Attack [{pipeline_name}]"):
        doc_id = gold["id"]
        anon_text = anonymized_texts.get(doc_id, "")
        if not anon_text:
            continue

        gt = ground_truth.get(doc_id)
        if gt is None or all(v is None for v in gt.values()):
            # No ground-truth attributes at all – attack not evaluable
            continue

        # Step 1: adversarial inference (sees ONLY the anonymized text)
        attack_response = call_gemini(client, build_attack_prompt(anon_text))
        attacker_guesses = parse_attack_response(attack_response)
        if API_DELAY > 0:
            time.sleep(API_DELAY)

        # Step 2: LLM match judging
        match_response = call_gemini(client, build_match_judge_prompt(gt, attacker_guesses))
        match_judgment = parse_match_response(match_response)
        if not _extract_json_obj(match_response):
            parse_failures += 1

        results.append({
            "id": doc_id,
            "complexity": gold.get("meta_temp", "Unknown"),
            "ground_truth": gt,
            "attacker_guesses": attacker_guesses,
            "match": match_judgment,
        })

        if API_DELAY > 0:
            time.sleep(API_DELAY)

    if parse_failures > 0:
        print(f"  Warning: {parse_failures}/{len(results)} match responses failed to parse")

    aggregated = aggregate_attack_scores(results, pipeline_name)
    return {
        "pipeline": pipeline_name,
        "per_document": results,
        "aggregated": aggregated,
        "parse_failures": parse_failures,
    }


def aggregate_attack_scores(results: List[Dict], pipeline_name: str) -> Dict:
    """
    Per-attribute attack success rate (lower = better anonymization), plus
    'any-attribute leaked' rate, plus per-complexity breakdown.

    For each attribute:
        success_rate = MATCH / (MATCH + NO_MATCH)        # SKIPs excluded
    """
    def _attr_stats(rs: List[Dict], attr: str) -> Dict:
        match_n   = sum(1 for r in rs if r["match"].get(attr) == "MATCH")
        nomatch_n = sum(1 for r in rs if r["match"].get(attr) == "NO_MATCH")
        skip_n    = sum(1 for r in rs if r["match"].get(attr) == "SKIP")
        evaluable = match_n + nomatch_n
        return {
            "match_count":     match_n,
            "no_match_count":  nomatch_n,
            "skip_count":      skip_n,
            "evaluable_count": evaluable,
            "success_rate":    round(match_n / evaluable, 3) if evaluable > 0 else None,
        }

    overall_per_attr = {a: _attr_stats(results, a) for a in ATTACK_ATTRIBUTES}

    # "Any attribute leaked" = at least one MATCH on at least one attribute
    any_leak = sum(1 for r in results if any(v == "MATCH" for v in r["match"].values()))
    valid_rates = [v["success_rate"] for v in overall_per_attr.values() if v["success_rate"] is not None]
    avg_leak_rate = round(sum(valid_rates) / len(valid_rates), 3) if valid_rates else None

    # Per-complexity breakdown
    by_complexity = defaultdict(list)
    for r in results:
        by_complexity[r["complexity"]].append(r)

    per_complexity = {}
    for level in ["Low", "Medium", "High"]:
        group = by_complexity.get(level, [])
        if not group:
            continue
        any_leak_g = sum(1 for r in group if any(v == "MATCH" for v in r["match"].values()))
        per_attr_g = {a: _attr_stats(group, a) for a in ATTACK_ATTRIBUTES}
        rates_g = [v["success_rate"] for v in per_attr_g.values() if v["success_rate"] is not None]
        per_complexity[level] = {
            "count": len(group),
            "any_attribute_leak_rate": round(any_leak_g / len(group), 3),
            "average_leak_rate": round(sum(rates_g) / len(rates_g), 3) if rates_g else None,
            "per_attribute": per_attr_g,
        }

    return {
        "pipeline": pipeline_name,
        "overall": {
            "count":                   len(results),
            "any_attribute_leak_rate": round(any_leak / len(results), 3) if results else 0.0,
            "average_leak_rate":       avg_leak_rate,
            "per_attribute":           overall_per_attr,
        },
        "by_complexity": per_complexity,
    }


def format_attack_report(all_results: List[Dict]) -> str:
    """Generate a printable adversarial-inference-attack report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  ADVERSARIAL INFERENCE ATTACK REPORT (Staab et al., ICLR 2025)")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Adversary / Judge Model: {GEMINI_MODEL}")
    lines.append(f"  Attributes attacked: {', '.join(ATTACK_ATTRIBUTES)}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("  Lower is better. success_rate = MATCH / (MATCH + NO_MATCH).")
    lines.append("  SKIP (ground truth absent in original) is excluded from the denominator.")
    lines.append("  any_leak = at least one attribute successfully inferred from the anonymized text.")

    # ── Summary table ──
    col_w = 6
    lines.append(f"\n{'#' * 78}")
    lines.append("  SUMMARY: Attack Success Rate by Pipeline")
    lines.append(f"{'#' * 78}\n")
    header = (
        f"  {'Pipeline':<38} {'AnyLk':>6} {'AvgLk':>6} "
        + " ".join(f"{a[:5]:>{col_w}}" for a in ATTACK_ATTRIBUTES)
        + f" {'n':>5}"
    )
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")

    for result in all_results:
        name = result["pipeline"]
        o = result["aggregated"]["overall"]
        any_lk = f"{o['any_attribute_leak_rate']:.1%}" if o['any_attribute_leak_rate'] is not None else "n/a"
        avg_lk = f"{o['average_leak_rate']:.1%}"       if o['average_leak_rate']       is not None else "n/a"
        per_attr_strs = []
        for a in ATTACK_ATTRIBUTES:
            sr = o["per_attribute"][a]["success_rate"]
            per_attr_strs.append(f"{sr:>{col_w}.1%}" if sr is not None else f"{'n/a':>{col_w}}")
        lines.append(
            f"  {name:<38} {any_lk:>6} {avg_lk:>6} "
            + " ".join(per_attr_strs)
            + f" {o['count']:>5}"
        )

    # ── Per-complexity breakdown ──
    for level in ["Low", "Medium", "High"]:
        if not any(level in r["aggregated"]["by_complexity"] for r in all_results):
            continue
        lines.append(f"\n  >>> {level} Complexity <<<")
        lines.append(
            f"  {'Pipeline':<38} {'AnyLk':>6} {'AvgLk':>6} "
            + " ".join(f"{a[:5]:>{col_w}}" for a in ATTACK_ATTRIBUTES)
            + f" {'n':>5}"
        )
        lines.append(f"  {'-' * (len(header) - 2)}")
        for result in all_results:
            comp = result["aggregated"]["by_complexity"].get(level, {})
            if not comp:
                continue
            any_lk = f"{comp['any_attribute_leak_rate']:.1%}"
            avg_lk = f"{comp['average_leak_rate']:.1%}" if comp['average_leak_rate'] is not None else "n/a"
            per_attr_strs = []
            for a in ATTACK_ATTRIBUTES:
                sr = comp["per_attribute"][a]["success_rate"]
                per_attr_strs.append(f"{sr:>{col_w}.1%}" if sr is not None else f"{'n/a':>{col_w}}")
            lines.append(
                f"  {result['pipeline']:<38} {any_lk:>6} {avg_lk:>6} "
                + " ".join(per_attr_strs)
                + f" {comp['count']:>5}"
            )

    # ── Detailed per-pipeline ──
    lines.append(f"\n\n{'#' * 78}")
    lines.append("  DETAILED PER-PIPELINE STATISTICS")
    lines.append(f"{'#' * 78}")
    for result in all_results:
        name = result["pipeline"]
        o = result["aggregated"]["overall"]
        lines.append(f"\n  {'=' * 70}")
        lines.append(f"  Pipeline: {name}")
        lines.append(f"  {'=' * 70}")
        lines.append(f"  Documents: {o['count']} | Match-judge parse failures: {result['parse_failures']}")
        lines.append(f"  Any-attribute leak rate: {o['any_attribute_leak_rate']:.1%}")
        if o["average_leak_rate"] is not None:
            lines.append(f"  Mean per-attribute leak rate: {o['average_leak_rate']:.1%}")
        lines.append("")
        lines.append("  Per attribute (success_rate = MATCH / evaluable; evaluable = MATCH + NO_MATCH):")
        for a in ATTACK_ATTRIBUTES:
            stats = o["per_attribute"][a]
            sr = stats["success_rate"]
            sr_str = f"{sr:.1%}" if sr is not None else "n/a"
            lines.append(
                f"    {a:<13} success={sr_str:>6}  "
                f"(match={stats['match_count']}, no_match={stats['no_match_count']}, skip={stats['skip_count']})"
            )

        for level in ["Low", "Medium", "High"]:
            comp = result["aggregated"]["by_complexity"].get(level, {})
            if not comp:
                continue
            lines.append("")
            lines.append(f"  {level} (n={comp['count']}):")
            lines.append(f"    Any-attribute leak: {comp['any_attribute_leak_rate']:.1%}")
            if comp.get("average_leak_rate") is not None:
                lines.append(f"    Mean per-attribute leak: {comp['average_leak_rate']:.1%}")
            for a in ATTACK_ATTRIBUTES:
                stats = comp["per_attribute"][a]
                sr = stats["success_rate"]
                sr_str = f"{sr:.1%}" if sr is not None else "n/a"
                lines.append(
                    f"      {a:<13} success={sr_str:>6}  "
                    f"(match={stats['match_count']}, no_match={stats['no_match_count']}, skip={stats['skip_count']})"
                )

    return "\n".join(lines)


def run_attack_evaluation(
    client,
    gold_records: List[Dict],
    pipelines: Dict[str, Dict[int, str]],
):
    """
    Top-level driver for the adversarial-inference-attack measurement.
    Mirrors the resume-from-cache behaviour of the LLM-as-judge pipeline.
    """
    print(f"\n{'#' * 78}")
    print("  ADVERSARIAL INFERENCE ATTACK (Staab et al., ICLR 2025)")
    print(f"{'#' * 78}")

    os.makedirs(ATTACK_OUTPUT_DIR, exist_ok=True)

    # Phase 1: Ground-truth extraction from originals (cached, pipeline-independent)
    print("\n  [Phase 1] Extracting ground-truth attributes from original texts")
    ground_truth = extract_ground_truth_attributes(client, gold_records, GROUND_TRUTH_CACHE)

    # Phase 2: Per-pipeline adversarial inference + match judging
    print("\n  [Phase 2] Running adversarial inference per pipeline")
    all_attack_results = []
    for pipeline_name, anon_texts in pipelines.items():
        safe_name = (
            pipeline_name.lower().replace(" ", "_").replace("+", "")
            .replace("[", "").replace("]", "")
        )
        per_doc_path = os.path.join(ATTACK_OUTPUT_DIR, f"{safe_name}_attack_scores.json")

        if os.path.exists(per_doc_path):
            print(f"\n    → SKIPPING attack on {pipeline_name} (already exists: {per_doc_path})")
            with open(per_doc_path, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)
            existing_agg = aggregate_attack_scores(existing_docs, pipeline_name)
            all_attack_results.append({
                "pipeline":       pipeline_name,
                "per_document":   existing_docs,
                "aggregated":     existing_agg,
                "parse_failures": 0,
            })
            continue

        print(f"\n  {'=' * 60}")
        print(f"  Attacking: {pipeline_name}")
        print(f"  {'=' * 60}")
        result = attack_pipeline(client, gold_records, anon_texts, ground_truth, pipeline_name)
        all_attack_results.append(result)

        with open(per_doc_path, "w", encoding="utf-8") as f:
            json.dump(result["per_document"], f, indent=2, ensure_ascii=False)
        print(f"    Saved: {per_doc_path}")

    # Phase 3: Report
    if not all_attack_results:
        print("\n  No attack results produced.")
        return

    report = format_attack_report(all_attack_results)
    print("\n" + report)

    report_path = os.path.join(ATTACK_OUTPUT_DIR, "adversarial_attack_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Attack report: {report_path}")

    agg = {r["pipeline"]: r["aggregated"] for r in all_attack_results}
    agg_path = os.path.join(ATTACK_OUTPUT_DIR, "adversarial_attack_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"  Aggregated: {agg_path}")


# =====================================================================
#  9. MAIN
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

    # ── Adversarial inference attack (Staab et al., ICLR 2025) ──
    # Runs alongside the LLM-as-judge above. Empirically measures how much
    # PII a strong LLM can recover from the anonymized text alone.
    if RUN_ATTACK:
        run_attack_evaluation(client, gold_records, pipelines)

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()