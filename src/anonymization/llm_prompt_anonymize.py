"""
llm_prompt_anonymize.py
========================
LLM-based DIRECT anonymization via prompt-based rewriting (RQ2a / RQ2b).

Unlike tag-and-replace (which detects PII spans and replaces with [LABEL]
placeholders), this approach instructs the LLM to rewrite the entire text
with PII removed or replaced by natural-sounding alternatives.

This enables a three-way semantic preservation comparison:
  1. Classical tag-and-replace  (spaCy/BERT → [LABEL] placeholders)
  2. LLM tag-and-replace        (Llama-3 → [LABEL] placeholders)
  3. LLM prompt-based rewrite   (Llama-3 → natural language rewrite)  ← THIS

Evaluation dimensions:
  - PII leakage: Do any ground-truth PII strings survive in the rewrite?
  - Semantic preservation: BERTScore(rewrite, original) and
    BERTScore(masked_rewrite, masked_original)
  - Qualitative: Side-by-side samples for human review

How to use:
  1. Adjust USER SETTINGS below (model, strategy, paths)
  2. Run: python llm_prompt_anonymize.py
  3. Then run semantic_preservation.py with the output predictions

Requirements:
    pip install transformers torch accelerate tqdm bitsandbytes bert-score
"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

MODEL       = "meta-llama/Meta-Llama-3-8B-Instruct"
STRATEGY    = "few-shot"      # "zero-shot" or "few-shot"
QUANTIZE    = False           # True for 13B models
MAX_DOCS    = None            # None for full run, small int for testing
SEED        = 42

# Paths
INPUT_PATH  = r"C:\thesis\data\label_studio\20260302_Export_Label_Studio_Client_Notes.json"
OUTPUT_DIR  = r"C:\thesis\results\llm_prompt_anonymize"
SPLIT_IDS   = r"C:\thesis\results\bert_finetuned\split_ids.json"

# BERTScore settings (for inline evaluation)
BERTSCORE_MODEL = "bert-base-multilingual-cased"
RUN_BERTSCORE   = True    # Set False to skip BERTScore (just generate rewrites)


# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import re
import time
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

from evaluation_utils import ALL_LABELS, load_label_studio_export


# =====================================================================
#  1. PROMPT CONSTRUCTION
# =====================================================================

def build_system_prompt() -> str:
    """
    System prompt for direct anonymization.
    The LLM rewrites the text to remove all PII while preserving meaning.
    """
    return (
        "You are an expert text anonymization system for German-language "
        "financial communications from a Swiss banking context.\n\n"
        "Your task: Rewrite the given text so that ALL personally identifiable "
        "information (PII) is removed or replaced with generic alternatives, "
        "while preserving the original meaning, structure, and informational "
        "content as closely as possible.\n\n"
        "PII categories to anonymize:\n"
        "- Person names → replace with generic references (e.g., 'ein Kunde', 'die Geschäftsführerin', 'Herr [Name]')\n"
        "- Organization names → replace with generic terms (e.g., 'ein Unternehmen', 'die Firma')\n"
        "- Locations/Addresses → replace with generic references (e.g., 'eine Stadt', 'ein Standort in der Schweiz')\n"
        "- Dates → replace with relative or vague references (e.g., 'kürzlich', 'im vergangenen Monat', 'an einem Datum')\n"
        "- Email addresses → remove or replace with '[E-Mail]'\n"
        "- Phone numbers → remove or replace with '[Telefon]'\n"
        "- IBAN/Account numbers → replace with '[Konto]'\n"
        "- Monetary amounts → keep the amounts but remove account-linking context, OR generalize (e.g., 'ein Betrag')\n"
        "- Job titles → generalize (e.g., 'eine Führungskraft', 'ein Mitarbeiter')\n"
        "- Age references → remove or generalize\n"
        "- Nationality → remove or generalize (e.g., 'ausländisch')\n"
        "- Education → generalize (e.g., 'ein Hochschulabschluss')\n\n"
        "RULES:\n"
        "1. Preserve the MEANING and INTENT of the communication.\n"
        "2. Keep the text in German.\n"
        "3. Maintain the same register and tone (formal/informal).\n"
        "4. The rewritten text should be fluent and natural.\n"
        "5. Do NOT add information that was not in the original.\n"
        "6. Do NOT omit non-PII information (keep business context, transaction details, etc.).\n"
        "7. Monetary amounts may be kept as-is if they don't identify a person alone.\n"
        "8. Output ONLY the rewritten text. No explanations, no preamble, no markdown.\n"
    )


# ── Few-Shot Examples ──

FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "Betriebsbesichtigung bei der Biofood Produzenten GmbH in Bern am 12.05.2024. "
            "Der Geschäftsführer, Herr Markus Aebischer, führte durch die neuen Anlagen. "
            "Das Unternehmen verzeichnet ein starkes Wachstum im europäischen Markt. "
            "Zur Finanzierung des weiteren Ausbaus wird eine Trade & Export Finance (TEF) "
            "Lösung für Lieferungen nach Deutschland und Österreich geprüft. Das aktuelle "
            "Volumen beträgt ca. CHF 750'000 pro Monat. Herr Aebischer ist 48 Jahre alt "
            "und hat an der ETH Zürich studiert."
        ),
        "output": (
            "Betriebsbesichtigung bei einem Lebensmittelunternehmen an einem Schweizer Standort "
            "an einem Datum im Frühjahr. Der Geschäftsführer führte durch die neuen Anlagen. "
            "Das Unternehmen verzeichnet ein starkes Wachstum im europäischen Markt. "
            "Zur Finanzierung des weiteren Ausbaus wird eine Trade & Export Finance (TEF) "
            "Lösung für Lieferungen ins Ausland geprüft. Das aktuelle "
            "Volumen beträgt ca. CHF 750'000 pro Monat. Der Geschäftsführer hat "
            "einen Hochschulabschluss."
        ),
    },
    {
        "input": (
            "Telefonat mit Frau Sandra Wyss (079 111 22 33), Buchhalterin bei der "
            "Bau-Expert GmbH am 30. April 2024. Sie informierte uns über die Saldierung "
            "des Kontos mit der CH33 0077 7888 9999 0000 1, da die entsprechende "
            "Projektgesellschaft liquidiert wurde. Der Restbetrag von CHF 12'450.50 soll "
            "auf das Hauptkonto überwiesen werden. Die rechtsverbindliche Unterschrift des "
            "Geschäftsführers Peter Schmid liegt uns vor."
        ),
        "output": (
            "Telefonat mit einer Mitarbeiterin eines Bauunternehmens. "
            "Sie informierte uns über die Saldierung "
            "eines Kontos, da die entsprechende "
            "Projektgesellschaft liquidiert wurde. Der Restbetrag von CHF 12'450.50 soll "
            "auf das Hauptkonto überwiesen werden. Die rechtsverbindliche Unterschrift des "
            "Geschäftsführers liegt uns vor."
        ),
    },
    {
        "input": (
            "Am 28. Februar 2024 fand das Eröffnungsgespräch mit der neu gegründeten "
            "Pharma Spin-Off AG statt. Der designierte CEO, Dr. Martin Fischer, ein "
            "ETH-Absolvent, benötigt diverse Firmenkonten in CHF und EUR. Die notwendigen "
            "KYC-Dokumente, inklusive Handelsregisterauszug aus dem Kanton Basel-Stadt, "
            "wurden übergeben. Kontakt für die technische Anbindung des Cash Managements "
            "ist martin.fischer@pharmaspin.ch."
        ),
        "output": (
            "An einem Datum fand das Eröffnungsgespräch mit einem neu gegründeten "
            "Pharmaunternehmen statt. Der designierte Geschäftsführer, ein "
            "Hochschulabsolvent, benötigt diverse Firmenkonten in CHF und EUR. Die notwendigen "
            "KYC-Dokumente, inklusive Handelsregisterauszug aus einem Schweizer Kanton, "
            "wurden übergeben. Die Kontaktdaten für die technische Anbindung des Cash Managements "
            "wurden hinterlegt."
        ),
    },
]


def build_user_prompt_zero_shot(text: str) -> str:
    """Zero-shot: just the text to anonymize."""
    return f"Anonymize the following text by rewriting it:\n\nOriginal: {text}\n\nAnonymized:"


def build_user_prompt_few_shot(text: str) -> str:
    """Few-shot: include examples before the target text."""
    parts = []
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(
            f"Example {i}:\n"
            f"Original: {example['input']}\n"
            f"Anonymized: {example['output']}"
        )

    parts.append(
        f"Now anonymize the following text by rewriting it:\n\n"
        f"Original: {text}\n"
        f"Anonymized:"
    )
    return "\n\n".join(parts)


# =====================================================================
#  2. MODEL LOADING (reused from llm_tag_and_replace.py)
# =====================================================================

def load_model(model_name: str, quantize_4bit: bool = False):
    """Load a HuggingFace causal LM with tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}

    if quantize_4bit:
        print("  Using 4-bit quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e9:.1f}B")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


# =====================================================================
#  3. INFERENCE
# =====================================================================

def generate_response(
    model, tokenizer, messages: List[Dict],
    max_new_tokens: int = 1024, temperature: float = 0.0,
) -> str:
    """Generate a response from the model given chat-formatted messages."""
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        input_text = f"[INST] {system}\n\n{user} [/INST]"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        outputs = model.generate(**inputs, **gen_kwargs)

    input_length = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][input_length:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


# =====================================================================
#  4. PII LEAKAGE DETECTION
# =====================================================================

def check_pii_leakage(
    rewritten_text: str,
    gold_entities: List[Dict],
    case_sensitive: bool = False,
) -> Dict:
    """
    Check whether ground-truth PII strings survive in the rewritten text.

    Returns:
        {
            "total_pii": int,
            "leaked_pii": int,
            "leakage_rate": float,
            "leaked_entities": [{"label", "text", "context"}],
            "per_category": {label: {"total", "leaked", "rate"}}
        }
    """
    leaked = []
    per_category = defaultdict(lambda: {"total": 0, "leaked": 0})

    check_text = rewritten_text if case_sensitive else rewritten_text.lower()

    for ent in gold_entities:
        label = ent["label"]
        pii_text = ent["text"]
        per_category[label]["total"] += 1

        search_text = pii_text if case_sensitive else pii_text.lower()

        # Skip very short PII (1-2 chars) as they produce false positives
        # e.g., age "58" might appear as part of a year "2058"
        if len(pii_text) <= 2:
            # For short strings, require word boundary match
            pattern = r'\b' + re.escape(search_text) + r'\b'
            found = bool(re.search(pattern, check_text))
        else:
            found = search_text in check_text

        if found:
            # Get context around the leak
            idx = check_text.find(search_text)
            if idx == -1 and len(pii_text) <= 2:
                match = re.search(r'\b' + re.escape(search_text) + r'\b', check_text)
                idx = match.start() if match else -1

            context_start = max(0, idx - 30)
            context_end = min(len(rewritten_text), idx + len(pii_text) + 30)
            context = rewritten_text[context_start:context_end]

            leaked.append({
                "label": label,
                "text": pii_text,
                "context": f"...{context}...",
            })
            per_category[label]["leaked"] += 1

    total = len(gold_entities)
    leaked_count = len(leaked)

    # Compute per-category rates
    per_cat_result = {}
    for label in sorted(per_category.keys()):
        cat = per_category[label]
        cat["rate"] = round(cat["leaked"] / max(cat["total"], 1), 4)
        per_cat_result[label] = dict(cat)

    return {
        "total_pii": total,
        "leaked_pii": leaked_count,
        "leakage_rate": round(leaked_count / max(total, 1), 4),
        "leaked_entities": leaked,
        "per_category": per_cat_result,
    }


# =====================================================================
#  5. MAIN INFERENCE LOOP
# =====================================================================

def run_prompt_anonymization(
    model, tokenizer,
    records: List[Dict],
    strategy: str = "few-shot",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Tuple[List[Dict], Dict]:
    """
    Run prompt-based anonymization on all records.

    Returns:
        results:   list of per-document results
        run_stats: timing and summary statistics
    """
    system_prompt = build_system_prompt()

    results = []
    total_time = 0.0
    all_leakage = []

    for rec in tqdm(records, desc=f"Prompt Anonymization [{strategy}]"):
        text = rec["text"]
        gold_ents = rec["entities"]
        doc_id = rec["id"]

        # Build prompt
        if strategy == "few-shot":
            user_prompt = build_user_prompt_few_shot(text)
        else:
            user_prompt = build_user_prompt_zero_shot(text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate rewrite
        start_time = time.time()
        rewritten = generate_response(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        elapsed = time.time() - start_time
        total_time += elapsed

        # Clean up common LLM artifacts
        rewritten = _clean_response(rewritten)

        # Check PII leakage
        leakage = check_pii_leakage(rewritten, gold_ents)
        all_leakage.append(leakage)

        results.append({
            "id": doc_id,
            "original_text": text,
            "rewritten_text": rewritten,
            "meta_temp": rec.get("meta_temp", "Unknown"),
            "gold_entities": gold_ents,
            "leakage": leakage,
            "inference_time": round(elapsed, 2),
            "original_length": len(text),
            "rewritten_length": len(rewritten),
            "length_ratio": round(len(rewritten) / max(len(text), 1), 4),
        })

    # Aggregate stats
    run_stats = {
        "total_records": len(records),
        "total_time": round(total_time, 1),
        "avg_time_per_doc": round(total_time / max(len(records), 1), 2),
        "docs_per_sec": round(len(records) / max(total_time, 0.01), 1),
        "avg_leakage_rate": round(
            sum(l["leakage_rate"] for l in all_leakage) / max(len(all_leakage), 1), 4
        ),
        "overall_leaked": sum(l["leaked_pii"] for l in all_leakage),
        "overall_total_pii": sum(l["total_pii"] for l in all_leakage),
        "avg_length_ratio": round(
            sum(r["length_ratio"] for r in results) / max(len(results), 1), 4
        ),
    }

    return results, run_stats


def _clean_response(text: str) -> str:
    """Remove common LLM artifacts from the rewritten text."""
    # Remove markdown code fences
    text = re.sub(r"```\w*\n?", "", text)
    # Remove "Anonymized:" prefix if the model echoed the prompt
    text = re.sub(r"^(Anonymized|Anonymisiert|Output|Rewritten):\s*", "", text, flags=re.IGNORECASE)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


# =====================================================================
#  6. REPORTING
# =====================================================================

def format_report(
    results: List[Dict],
    run_stats: Dict,
    model_name: str,
    strategy: str,
) -> str:
    """Generate a human-readable evaluation report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  LLM PROMPT-BASED ANONYMIZATION REPORT")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model: {model_name}")
    lines.append(f"  Strategy: {strategy}")
    lines.append("=" * 78)

    lines.append(f"\n  Documents: {run_stats['total_records']}")
    lines.append(f"  Total time: {run_stats['total_time']}s "
                 f"({run_stats['avg_time_per_doc']}s/doc)")
    lines.append(f"  Avg length ratio (rewrite/original): {run_stats['avg_length_ratio']:.2f}")

    # ── PII Leakage Summary ──
    lines.append(f"\n{'#' * 78}")
    lines.append("  PII LEAKAGE ANALYSIS")
    lines.append(f"{'#' * 78}")

    overall_leaked = run_stats["overall_leaked"]
    overall_total = run_stats["overall_total_pii"]
    overall_rate = overall_leaked / max(overall_total, 1)
    lines.append(f"\n  Overall: {overall_leaked}/{overall_total} PII instances leaked "
                 f"({overall_rate:.1%})")

    # Per-category leakage
    cat_totals = defaultdict(lambda: {"total": 0, "leaked": 0})
    for res in results:
        for label, cat in res["leakage"]["per_category"].items():
            cat_totals[label]["total"] += cat["total"]
            cat_totals[label]["leaked"] += cat["leaked"]

    lines.append(f"\n  {'Category':<12} {'Total':>8} {'Leaked':>8} {'Rate':>10}")
    lines.append(f"  {'-' * 42}")
    for label in sorted(cat_totals.keys()):
        t = cat_totals[label]["total"]
        l = cat_totals[label]["leaked"]
        r = l / max(t, 1)
        lines.append(f"  {label:<12} {t:>8} {l:>8} {r:>10.1%}")

    # ── Per-Complexity Breakdown ──
    lines.append(f"\n{'#' * 78}")
    lines.append("  LEAKAGE BY COMPLEXITY LEVEL")
    lines.append(f"{'#' * 78}")

    by_complexity = defaultdict(list)
    for res in results:
        by_complexity[res["meta_temp"]].append(res)

    for level in ["Low", "Medium", "High"]:
        group = by_complexity.get(level, [])
        if not group:
            continue
        leaked = sum(r["leakage"]["leaked_pii"] for r in group)
        total = sum(r["leakage"]["total_pii"] for r in group)
        rate = leaked / max(total, 1)
        avg_len_ratio = sum(r["length_ratio"] for r in group) / len(group)
        lines.append(f"\n  {level}: {len(group)} docs | "
                     f"Leakage: {leaked}/{total} ({rate:.1%}) | "
                     f"Avg length ratio: {avg_len_ratio:.2f}")

    # ── Sample Leaked PII ──
    lines.append(f"\n{'#' * 78}")
    lines.append("  SAMPLE LEAKED PII (first 20)")
    lines.append(f"{'#' * 78}")

    leak_count = 0
    for res in results:
        for leak in res["leakage"]["leaked_entities"]:
            if leak_count >= 20:
                break
            lines.append(f"  [{leak['label']:<8}] \"{leak['text']}\"  →  {leak['context']}")
            leak_count += 1
        if leak_count >= 20:
            break

    # ── Qualitative Samples ──
    lines.append(f"\n{'#' * 78}")
    lines.append("  QUALITATIVE SAMPLES (10 examples)")
    lines.append(f"{'#' * 78}")

    # Pick samples: some good (low leakage), some bad (high leakage)
    sorted_by_leakage = sorted(results, key=lambda r: r["leakage"]["leakage_rate"], reverse=True)
    samples = sorted_by_leakage[:5]  # worst
    clean = [r for r in results if r["leakage"]["leaked_pii"] == 0]
    if clean:
        random.shuffle(clean)
        samples.extend(clean[:5])  # best

    for res in samples:
        lines.append(f"\n  {'─' * 72}")
        lines.append(f"  Doc {res['id']} | {res['meta_temp']} | "
                     f"Leakage: {res['leakage']['leaked_pii']}/{res['leakage']['total_pii']} | "
                     f"Length: {res['original_length']}→{res['rewritten_length']}")
        lines.append(f"  ORIGINAL:  {res['original_text'][:300]}{'...' if len(res['original_text']) > 300 else ''}")
        lines.append(f"  REWRITTEN: {res['rewritten_text'][:300]}{'...' if len(res['rewritten_text']) > 300 else ''}")

    return "\n".join(lines)


def save_predictions_for_semantic_eval(
    results: List[Dict],
    output_path: str,
) -> None:
    """
    Save predictions in a format compatible with semantic_preservation.py.

    For prompt-based rewriting, we save the rewritten text as-is.
    The semantic_preservation.py script will need a small extension to
    handle this format (comparing rewritten text directly rather than
    constructing anonymized text from entity predictions).
    """
    output = []
    for res in results:
        output.append({
            "id": res["id"],
            "rewritten_text": res["rewritten_text"],
            "leakage": {
                "total_pii": res["leakage"]["total_pii"],
                "leaked_pii": res["leakage"]["leaked_pii"],
                "leakage_rate": res["leakage"]["leakage_rate"],
            },
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# =====================================================================
#  7. BERTSCORE EVALUATION (inline, optional)
# =====================================================================

def run_bertscore_evaluation(
    results: List[Dict],
    bertscore_model: str = "bert-base-multilingual-cased",
) -> Dict:
    """
    Compute BERTScore between original and rewritten texts.

    Returns aggregated scores overall and by complexity.
    """
    from bert_score import score

    device = "cuda" if torch.cuda.is_available() else "cpu"

    originals = [r["original_text"] for r in results]
    rewrites = [r["rewritten_text"] for r in results]

    print(f"\n  Computing BERTScore (full texts)...")
    P, R, F1 = score(
        rewrites, originals,
        model_type=bertscore_model,
        device=device,
        verbose=True,
        lang="de",
    )

    # Attach to results
    for i, res in enumerate(results):
        res["bertscore_full_p"] = round(P[i].item(), 6)
        res["bertscore_full_r"] = round(R[i].item(), 6)
        res["bertscore_full_f1"] = round(F1[i].item(), 6)

    # Aggregate
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    f1_scores = [r["bertscore_full_f1"] for r in results]

    agg = {
        "overall": {
            "bertscore_full_f1_mean": round(_mean(f1_scores), 4),
            "bertscore_full_f1_std": round(_std(f1_scores), 4),
            "bertscore_full_p_mean": round(_mean([r["bertscore_full_p"] for r in results]), 4),
            "bertscore_full_r_mean": round(_mean([r["bertscore_full_r"] for r in results]), 4),
        },
        "by_complexity": {},
    }

    by_complexity = defaultdict(list)
    for r in results:
        by_complexity[r["meta_temp"]].append(r)

    for level in ["Low", "Medium", "High"]:
        group = by_complexity.get(level, [])
        if group:
            f1s = [r["bertscore_full_f1"] for r in group]
            agg["by_complexity"][level] = {
                "count": len(group),
                "bertscore_full_f1_mean": round(_mean(f1s), 4),
                "bertscore_full_f1_std": round(_std(f1s), 4),
            }

    return agg


# =====================================================================
#  8. MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Load data ──
    print(f"Loading data from: {INPUT_PATH}")
    gold_records = load_label_studio_export(INPUT_PATH)
    print(f"  Total records: {len(gold_records)}")

    # ── Filter to test split ──
    if SPLIT_IDS:
        print(f"  Loading split IDs from: {SPLIT_IDS}")
        with open(SPLIT_IDS, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        test_ids = set(split_info["test_ids"])
        gold_records = [r for r in gold_records if r["id"] in test_ids]
        print(f"  Filtered to test split: {len(gold_records)} records")

    if MAX_DOCS is not None:
        gold_records = gold_records[:MAX_DOCS]
        print(f"  Limited to {len(gold_records)} documents")

    # ── Load model ──
    needs_4bit = QUANTIZE or "13b" in MODEL.lower()
    model, tokenizer = load_model(MODEL, quantize_4bit=needs_4bit)

    # ── Run inference ──
    results, run_stats = run_prompt_anonymization(
        model, tokenizer, gold_records,
        strategy=STRATEGY,
        max_new_tokens=1024,
        temperature=0.0,
    )

    # ── Optional BERTScore ──
    bertscore_agg = None
    if RUN_BERTSCORE:
        bertscore_agg = run_bertscore_evaluation(results, BERTSCORE_MODEL)
        run_stats["bertscore"] = bertscore_agg

    # ── Generate report ──
    report = format_report(results, run_stats, MODEL, STRATEGY)

    # Append BERTScore to report
    if bertscore_agg:
        report += f"\n\n{'#' * 78}\n"
        report += "  BERTSCORE: SEMANTIC PRESERVATION\n"
        report += f"{'#' * 78}\n"
        overall = bertscore_agg["overall"]
        report += f"\n  Overall BERTScore F1: {overall['bertscore_full_f1_mean']:.4f} ± {overall['bertscore_full_f1_std']:.4f}\n"
        report += f"  Precision: {overall['bertscore_full_p_mean']:.4f} | Recall: {overall['bertscore_full_r_mean']:.4f}\n"
        for level in ["Low", "Medium", "High"]:
            comp = bertscore_agg["by_complexity"].get(level, {})
            if comp:
                report += f"\n  {level} (n={comp['count']}): F1 = {comp['bertscore_full_f1_mean']:.4f} ± {comp['bertscore_full_f1_std']:.4f}"

    print(report)

    # ── Save outputs ──
    model_short = MODEL.split("/")[-1].lower().replace("-", "_")
    prefix = f"prompt_anon_{model_short}_{STRATEGY.replace('-', '_')}"

    # Report
    report_path = os.path.join(OUTPUT_DIR, f"{prefix}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_path}")

    # Full results (with texts)
    full_path = os.path.join(OUTPUT_DIR, f"{prefix}_full_results.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full results: {full_path}")

    # Predictions for semantic_preservation.py
    pred_path = os.path.join(OUTPUT_DIR, f"{prefix}_predictions.json")
    save_predictions_for_semantic_eval(results, pred_path)
    print(f"  Predictions: {pred_path}")

    # Run stats
    stats_path = os.path.join(OUTPUT_DIR, f"{prefix}_run_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(run_stats, f, indent=2, ensure_ascii=False)
    print(f"  Run stats: {stats_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
