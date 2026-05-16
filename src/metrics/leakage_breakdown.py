"""
leakage_breakdown.py
=====================
Compute PII leakage rate broken down by category, complexity, and tier
for ALL pipelines (tag-and-replace + prompt rewriting).

For tag-and-replace: checks if gold PII strings survive in the anonymized
text (after entity replacement with [LABEL] placeholders).
For prompt rewriting: checks if gold PII strings survive in the rewritten text.

Output:
  - JSON:  results/leakage_breakdown/leakage_breakdown.json
            (aggregate rates across all pipelines)
  - TXT:   results/leakage_breakdown/leakage_breakdown_report.txt
            (human-readable aggregate tables)
  - Per-pipeline, per-document files:
           results/leakage_breakdown/per_document/<pipeline>_leakage.json
           results/leakage_breakdown/per_document/<pipeline>_leakage.txt
            For every test document, lists which gold PII entities
            survived ("leaked") in the anonymized text and which were
            successfully masked.

Usage:
    python leakage_breakdown.py

"""

import json
import os
import re
from collections import defaultdict


def _slug(name):
    """Make a pipeline name safe for use as a filename."""
    s = name.lower()
    s = s.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "leakage_breakdown")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_PATH = os.path.join(BASE_DIR, "data", "label_studio",
                          "20260302_Export_Label_Studio_Client_Notes.json")
SPLIT_IDS = os.path.join(RESULTS_DIR, "bert_finetuned", "split_ids.json")

# ── Pipeline definitions ──────────────────────────────────────────

# Tag-and-replace: predictions files (entities with start/end/label)
TAG_REPLACE_PIPELINES = {
    "spaCy + Regex":           "classical_baselines/spacy/spacy_predictions.json",
    "BERT (pre) + Regex":      "classical_baselines/bert/bert_predictions.json",
    "Presidio":                "presidio_baseline/presidio_predictions.json",
    "BERT Fine-Tuned":         "bert_finetuned/bert_finetuned_predictions.json",
    "Llama-3 [zero-shot]":     "llm_baselines/llm_meta_llama_3_8b_instruct_zero_shot_predictions.json",
    "Llama-3 [few-shot]":      "llm_baselines/llm_meta_llama_3_8b_instruct_few_shot_predictions.json",
    "Llama-3 [fs+verify]":     "llm_baselines/llm_meta_llama_3_8b_instruct_few_shot_verified_predictions.json",
    "Qwen2.5 [zero-shot]":     "llm_baselines/llm_qwen2.5_7b_instruct_zero_shot_predictions.json",
    "Qwen2.5 [few-shot]":      "llm_baselines/llm_qwen2.5_7b_instruct_few_shot_predictions.json",
    "Qwen2.5 [fs+verify]":     "llm_baselines/llm_qwen2.5_7b_instruct_few_shot_verified_predictions.json",
    "SauerkrautLM [zero-shot]":"llm_baselines/llm_llama_3.1_sauerkrautlm_8b_instruct_zero_shot_predictions.json",
    "SauerkrautLM [few-shot]": "llm_baselines/llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_predictions.json",
    "SauerkrautLM [fs+verify]":"llm_baselines/llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_verified_predictions.json",
    "Llama-3 [fine-tuned]":    "llm_finetuned/llm_finetuned_meta_llama_3_8b_instruct/llm_finetuned_meta_llama_3_8b_instruct_predictions.json",
}

# Prompt rewriting: full_results files (with rewritten_text)
PROMPT_PIPELINES = {
    "Llama-3 [prompt]":       "llm_prompt_anonymize/prompt_anon_meta_llama_3_8b_instruct_few_shot_full_results.json",
    "Qwen2.5 [prompt]":       "llm_prompt_anonymize/prompt_anon_qwen2.5_7b_instruct_few_shot_full_results.json",
    "SauerkrautLM [prompt]":  "llm_prompt_anonymize/prompt_anon_llama_3.1_sauerkrautlm_8b_instruct_few_shot_full_results.json",
    "Datenwertsch\u00f6pfung API": "datenwertschoepfung_baseline/datenwertschoepfung_predictions.json",
}

TIER_MAP = {
    "PER": "Tier 1", "LOC": "Tier 1", "ORG": "Tier 1",
    "IBAN": "Tier 2", "EMAIL": "Tier 2", "PHONE": "Tier 2",
    "DATE": "Tier 2", "MONEY": "Tier 2",
    "JOB": "Tier 3", "AGE": "Tier 3", "NATION": "Tier 3", "EDU": "Tier 3",
}

ALL_CATEGORIES = ["PER", "LOC", "ORG", "DATE", "EMAIL", "PHONE",
                   "IBAN", "MONEY", "JOB", "AGE", "NATION", "EDU"]

ALL_LABELS = set(ALL_CATEGORIES)


def check_pii_leaked(text, pii_text):
    """Check if a PII string survives in text. Matches the matching logic
    of semantic_preservation.py exactly: case-insensitive, always with
    word-boundary anchors (\\b...\\b)."""
    search = re.escape(pii_text.lower())
    return bool(re.search(r'\b' + search + r'\b', text.lower()))


def build_anonymized_text(original_text, pred_entities):
    """Build anonymized text by replacing predicted entities with [LABEL] placeholders."""
    sorted_ents = sorted(pred_entities, key=lambda e: e["start"], reverse=True)
    anon = original_text
    for ent in sorted_ents:
        placeholder = f"[{ent['label']}]"
        anon = anon[:ent["start"]] + placeholder + anon[ent["end"]:]
    return anon


def load_gold_data():
    """Load gold standard records for test set."""
    ls_data = json.load(open(INPUT_PATH, encoding="utf-8"))
    split_ids = json.load(open(SPLIT_IDS, encoding="utf-8"))
    test_ids = set(split_ids["test_ids"])

    records = {}
    for entry in ls_data:
        if entry["id"] not in test_ids:
            continue
        entities = []
        for ann in entry.get("label", []):
            label = ann["labels"][0] if ann.get("labels") else None
            if label and label in ALL_LABELS:
                entities.append({
                    "start": ann["start"], "end": ann["end"],
                    "label": label, "text": ann["text"],
                })
        records[entry["id"]] = {
            "text": entry["text"],
            "meta_temp": entry.get("meta_temp", "Unknown"),
            "entities": entities,
        }
    return records


def compute_leakage(pipeline_name, anonymized_texts, gold_data):
    """
    Compute per-category, per-complexity, and per-tier leakage.
    anonymized_texts: dict of {doc_id: anonymized_text_string}

    Aggregation matches semantic_preservation.py: each rate is the
    mean across documents of the per-document leak rate within the
    relevant subset (overall / category / complexity / tier).
    Pooled counts (total entities, leaked entities) are also kept for
    transparency but are not used to compute the reported rate.
    """
    # Pooled counters (kept for transparency only)
    by_category = defaultdict(lambda: {"total": 0, "leaked": 0})
    by_complexity = defaultdict(lambda: {"total": 0, "leaked": 0})
    by_tier = defaultdict(lambda: {"total": 0, "leaked": 0})
    by_cat_complexity = defaultdict(lambda: defaultdict(lambda: {"total": 0, "leaked": 0}))
    overall_pool = {"total": 0, "leaked": 0}

    # Per-document rate accumulators (used to compute mean-across-docs)
    overall_doc_rates = []                              # [rate_per_doc]
    cat_doc_rates = defaultdict(list)                   # cat -> [rate_per_doc]
    complexity_doc_rates = defaultdict(list)            # complexity -> [rate_per_doc]
    tier_doc_rates = defaultdict(list)                  # tier -> [rate_per_doc]
    cat_complexity_doc_rates = defaultdict(lambda: defaultdict(list))  # cat -> complexity -> [rate]

    for doc_id, anon_text in anonymized_texts.items():
        if doc_id not in gold_data:
            continue
        gold = gold_data[doc_id]
        complexity = gold["meta_temp"]

        # Per-doc accumulators
        doc_total = 0
        doc_leaked = 0
        doc_cat_total = defaultdict(int)
        doc_cat_leaked = defaultdict(int)
        doc_tier_total = defaultdict(int)
        doc_tier_leaked = defaultdict(int)

        for ent in gold["entities"]:
            label = ent["label"]
            tier = TIER_MAP.get(label, "Unknown")
            leaked = check_pii_leaked(anon_text, ent["text"])

            # Pooled counters
            overall_pool["total"] += 1
            by_category[label]["total"] += 1
            by_complexity[complexity]["total"] += 1
            by_tier[tier]["total"] += 1
            by_cat_complexity[label][complexity]["total"] += 1
            if leaked:
                overall_pool["leaked"] += 1
                by_category[label]["leaked"] += 1
                by_complexity[complexity]["leaked"] += 1
                by_tier[tier]["leaked"] += 1
                by_cat_complexity[label][complexity]["leaked"] += 1

            # Per-doc counters
            doc_total += 1
            doc_cat_total[label] += 1
            doc_tier_total[tier] += 1
            if leaked:
                doc_leaked += 1
                doc_cat_leaked[label] += 1
                doc_tier_leaked[tier] += 1

        # Convert per-doc counts to per-doc rates and stash for averaging
        if doc_total > 0:
            r = doc_leaked / doc_total
            overall_doc_rates.append(r)
            complexity_doc_rates[complexity].append(r)
        for cat, tot in doc_cat_total.items():
            if tot > 0:
                cr = doc_cat_leaked[cat] / tot
                cat_doc_rates[cat].append(cr)
                cat_complexity_doc_rates[cat][complexity].append(cr)
        for tier, tot in doc_tier_total.items():
            if tot > 0:
                tier_doc_rates[tier].append(doc_tier_leaked[tier] / tot)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    result = {
        "pipeline": pipeline_name,
        "overall": {
            **overall_pool,
            "rate": mean(overall_doc_rates),
            "n_docs": len(overall_doc_rates),
        },
        "by_category": {
            cat: {
                **by_category[cat],
                "rate": mean(cat_doc_rates[cat]),
                "n_docs": len(cat_doc_rates[cat]),
            }
            for cat in ALL_CATEGORIES if by_category[cat]["total"] > 0
        },
        "by_complexity": {
            comp: {
                **by_complexity[comp],
                "rate": mean(complexity_doc_rates[comp]),
                "n_docs": len(complexity_doc_rates[comp]),
            }
            for comp in ["Low", "Medium", "High"]
        },
        "by_tier": {
            tier: {
                **by_tier[tier],
                "rate": mean(tier_doc_rates[tier]),
                "n_docs": len(tier_doc_rates[tier]),
            }
            for tier in ["Tier 1", "Tier 2", "Tier 3"]
        },
        "by_category_complexity": {
            cat: {
                comp: {
                    **by_cat_complexity[cat][comp],
                    "rate": mean(cat_complexity_doc_rates[cat][comp]),
                    "n_docs": len(cat_complexity_doc_rates[cat][comp]),
                }
                for comp in ["Low", "Medium", "High"]
                if by_cat_complexity[cat][comp]["total"] > 0
            }
            for cat in ALL_CATEGORIES if by_category[cat]["total"] > 0
        },
    }
    return result


def compute_per_document_leakage(pipeline_name, anonymized_texts, gold_data):
    """Return a list of per-document dicts recording exactly which gold entities
    survived (leaked) and which were successfully masked.

    Each entry:
        {
          "id": int, "complexity": str,
          "original_text": str, "anonymized_text": str,
          "total_pii": int, "leaked_pii": int, "leak_rate": float,
          "entities": [
              {"label": str, "text": str, "start": int, "end": int,
               "leaked": bool}
          ]
        }
    """
    per_doc = []
    for doc_id, gold in gold_data.items():
        if doc_id not in anonymized_texts:
            continue
        anon_text = anonymized_texts[doc_id]
        entities_with_status = []
        leaked_count = 0
        for ent in gold["entities"]:
            is_leaked = check_pii_leaked(anon_text, ent["text"])
            entities_with_status.append({
                "label": ent["label"],
                "text": ent["text"],
                "start": ent["start"],
                "end": ent["end"],
                "leaked": is_leaked,
            })
            if is_leaked:
                leaked_count += 1

        total = len(entities_with_status)
        per_doc.append({
            "id": doc_id,
            "complexity": gold["meta_temp"],
            "original_text": gold["text"],
            "anonymized_text": anon_text,
            "total_pii": total,
            "leaked_pii": leaked_count,
            "leak_rate": round(leaked_count / total, 4) if total else 0.0,
            "entities": entities_with_status,
        })
    # Sort by doc id for stable output
    per_doc.sort(key=lambda d: d["id"])
    return per_doc


def format_per_document_report(pipeline_name, per_doc):
    """Human-readable per-document leakage report for one pipeline."""
    total_pii = sum(d["total_pii"] for d in per_doc)
    total_leaked = sum(d["leaked_pii"] for d in per_doc)
    docs_with_leak = sum(1 for d in per_doc if d["leaked_pii"] > 0)
    doc_count = len(per_doc)
    overall_rate = total_leaked / total_pii if total_pii else 0.0

    lines = []
    lines.append("=" * 100)
    lines.append(f"  PII LEAKAGE — PER-DOCUMENT REPORT")
    lines.append(f"  Pipeline: {pipeline_name}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  Documents:            {doc_count}")
    lines.append(f"  Docs with any leak:   {docs_with_leak}  "
                 f"({docs_with_leak / max(doc_count, 1):.1%})")
    lines.append(f"  Total PII entities:   {total_pii}")
    lines.append(f"  Leaked entities:      {total_leaked}  ({overall_rate:.1%})")
    lines.append("")
    lines.append(f"  Symbols used below:  * LEAKED   . masked")
    lines.append("")

    for d in per_doc:
        rate = d["leaked_pii"] / max(d["total_pii"], 1)
        header = (f"--- Doc {d['id']} [{d['complexity']}] | "
                  f"{d['leaked_pii']}/{d['total_pii']} leaked ({rate:.1%}) ---")
        lines.append(header)
        lines.append(f"  ORIG: {d['original_text'][:400]}"
                     + ("..." if len(d['original_text']) > 400 else ""))
        lines.append(f"  ANON: {d['anonymized_text'][:400]}"
                     + ("..." if len(d['anonymized_text']) > 400 else ""))
        lines.append("")
        # Show only leaked entities when the rate is low, all when rate > 0
        for ent in d["entities"]:
            mark = "*" if ent["leaked"] else "."
            status = "LEAKED" if ent["leaked"] else "masked"
            lines.append(f"    {mark} {status:<7} {ent['label']:<7} "
                         f"\"{ent['text'][:60]}\"")
        lines.append("")
    return "\n".join(lines)


def save_per_pipeline_details(pipeline_name, per_doc):
    """Write both .json and .txt per-document files for one pipeline."""
    sub_dir = os.path.join(OUTPUT_DIR, "per_document")
    os.makedirs(sub_dir, exist_ok=True)
    slug = _slug(pipeline_name)

    # JSON (structured)
    json_path = os.path.join(sub_dir, f"{slug}_leakage.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline": pipeline_name,
            "documents": per_doc,
        }, f, indent=2, ensure_ascii=False)

    # TXT (human-readable)
    report = format_per_document_report(pipeline_name, per_doc)
    txt_path = os.path.join(sub_dir, f"{slug}_leakage.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    return json_path, txt_path


def format_report(all_results):
    """Generate a comprehensive leakage breakdown report."""
    lines = []
    lines.append("=" * 100)
    lines.append("  PII LEAKAGE BREAKDOWN REPORT")
    lines.append("  All 17 pipelines — by category, complexity, and tier")
    lines.append("=" * 100)

    # ── Overall summary table ──
    lines.append("\n" + "#" * 100)
    lines.append("  OVERALL LEAKAGE RATES")
    lines.append("#" * 100)
    lines.append(f"\n  {'Pipeline':<35} {'Leaked':>8} {'Total':>8} {'Rate':>8}")
    lines.append("  " + "-" * 62)
    for r in all_results:
        o = r["overall"]
        lines.append(f"  {r['pipeline']:<35} {o['leaked']:>8} {o['total']:>8} {o['rate']:>8.1%}")

    # ── By category table ──
    lines.append("\n" + "#" * 100)
    lines.append("  LEAKAGE BY CATEGORY")
    lines.append("#" * 100)

    # Header
    cat_header = f"  {'Pipeline':<28}"
    for cat in ALL_CATEGORIES:
        cat_header += f" {cat:>7}"
    lines.append(f"\n{cat_header}")
    lines.append("  " + "-" * (28 + 8 * len(ALL_CATEGORIES)))

    for r in all_results:
        row = f"  {r['pipeline']:<28}"
        for cat in ALL_CATEGORIES:
            if cat in r["by_category"]:
                rate = r["by_category"][cat]["rate"]
                row += f" {rate:>6.1%}"
            else:
                row += f"    {'—':>3}"
        lines.append(row)

    # ── By tier ──
    lines.append("\n" + "#" * 100)
    lines.append("  LEAKAGE BY TIER")
    lines.append("#" * 100)
    lines.append(f"\n  {'Pipeline':<35} {'Tier 1':>10} {'Tier 2':>10} {'Tier 3':>10}")
    lines.append("  " + "-" * 68)
    for r in all_results:
        t1 = r["by_tier"].get("Tier 1", {}).get("rate", 0)
        t2 = r["by_tier"].get("Tier 2", {}).get("rate", 0)
        t3 = r["by_tier"].get("Tier 3", {}).get("rate", 0)
        lines.append(f"  {r['pipeline']:<35} {t1:>10.1%} {t2:>10.1%} {t3:>10.1%}")

    # ── By complexity ──
    lines.append("\n" + "#" * 100)
    lines.append("  LEAKAGE BY COMPLEXITY")
    lines.append("#" * 100)
    lines.append(f"\n  {'Pipeline':<35} {'Low':>10} {'Medium':>10} {'High':>10}")
    lines.append("  " + "-" * 68)
    for r in all_results:
        lo = r["by_complexity"].get("Low", {}).get("rate", 0)
        md = r["by_complexity"].get("Medium", {}).get("rate", 0)
        hi = r["by_complexity"].get("High", {}).get("rate", 0)
        lines.append(f"  {r['pipeline']:<35} {lo:>10.1%} {md:>10.1%} {hi:>10.1%}")

    return "\n".join(lines)


def main():
    print("Loading gold data...")
    gold_data = load_gold_data()
    print(f"  {len(gold_data)} test documents loaded")

    all_results = []

    # ── Tag-and-replace pipelines ──
    print("\nProcessing tag-and-replace pipelines...")
    for pipeline_name, pred_path in TAG_REPLACE_PIPELINES.items():
        fpath = os.path.join(RESULTS_DIR, pred_path)
        if not os.path.exists(fpath):
            print(f"  WARNING: Missing {fpath}")
            continue

        preds = json.load(open(fpath, encoding="utf-8"))
        pred_by_id = {p["id"]: p for p in preds}

        # Build anonymized texts
        anon_texts = {}
        for doc_id, gold in gold_data.items():
            if doc_id in pred_by_id:
                anon_texts[doc_id] = build_anonymized_text(
                    gold["text"], pred_by_id[doc_id]["entities"])

        result = compute_leakage(pipeline_name, anon_texts, gold_data)
        all_results.append(result)
        print(f"  {pipeline_name:<35} leak={result['overall']['rate']:.1%}")

        per_doc = compute_per_document_leakage(pipeline_name, anon_texts, gold_data)
        save_per_pipeline_details(pipeline_name, per_doc)

    # ── Prompt rewriting pipelines ──
    print("\nProcessing prompt rewriting pipelines...")
    for pipeline_name, pred_path in PROMPT_PIPELINES.items():
        fpath = os.path.join(RESULTS_DIR, pred_path)
        if not os.path.exists(fpath):
            print(f"  WARNING: Missing {fpath}")
            continue

        data = json.load(open(fpath, encoding="utf-8"))
        # Some pipelines use string IDs, gold data uses int IDs — coerce to int
        anon_texts = {}
        for r in data:
            try:
                rid = int(r["id"])
            except (TypeError, ValueError):
                rid = r["id"]
            if rid in gold_data:
                anon_texts[rid] = r.get("rewritten_text", "")

        result = compute_leakage(pipeline_name, anon_texts, gold_data)
        all_results.append(result)
        print(f"  {pipeline_name:<35} leak={result['overall']['rate']:.1%}")

        per_doc = compute_per_document_leakage(pipeline_name, anon_texts, gold_data)
        save_per_pipeline_details(pipeline_name, per_doc)

    # ── Save JSON ──
    json_path = os.path.join(OUTPUT_DIR, "leakage_breakdown.json")
    json.dump({r["pipeline"]: r for r in all_results},
              open(json_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  JSON: {json_path}")

    # ── Save report ──
    report = format_report(all_results)
    report_path = os.path.join(OUTPUT_DIR, "leakage_breakdown_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # ── Print report ──
    print(f"\n{report}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
