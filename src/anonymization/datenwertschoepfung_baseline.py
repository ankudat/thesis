"""
datenwertschoepfung_baseline.py
================================
Datenwertschöpfung Anonymizer API baseline.

Sends documents to the Datenwertschöpfung anonymization API and evaluates
using PII leakage rate (same methodology as the prompt-based rewriting
pipelines). The API returns anonymized text with placeholders — no entity
offsets are provided, making leakage rate the appropriate evaluation metric.

API options: names, numbers, e-mails, addresses.

Output:
  - Full results with leakage analysis (per document, per category, per complexity)
  - Side-by-side comparison (original vs anonymized)
  - Run statistics

Usage:
    python datenwertschoepfung_baseline.py

Requirements:
    pip install requests python-dotenv tqdm

"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

INPUT_PATH  = r"C:\thesis\data\label_studio\20260302_Export_Label_Studio_Client_Notes.json"
OUTPUT_DIR  = r"C:\thesis\results\datenwertschoepfung_baseline"
SPLIT_IDS   = r"C:\thesis\results\bert_finetuned\split_ids.json"
LIMIT       = None   # Set to 5 for quick test, None for full run

API_URL     = "https://185.229.90.29/anonymize"
API_OPTIONS = ["names", "numbers", "e-mails", "addresses"]

# Delay between API calls (seconds)
DELAY       = 0.3

# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import re
import time
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add anonymization directory to path for evaluation_utils
import sys
sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import ALL_LABELS, load_label_studio_export

load_dotenv()

# =====================================================================
#  1. API CLIENT
# =====================================================================

def get_api_token():
    token = os.environ.get("DATENWERTSCHOEPFUNG_TOKEN")
    if not token:
        raise ValueError(
            "No API token found. Set DATENWERTSCHOEPFUNG_TOKEN in .env file."
        )
    return token


def call_anonymizer_api(text: str, token: str, max_retries: int = 3) -> Dict:
    """Call the Datenwertschöpfung anonymizer API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {
        "consider_numwords": False,
        "options": API_OPTIONS,
        "text": text,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, headers=headers, json=data,
                verify=False, timeout=30
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {response.status_code}: {response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"  Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return {"error": "API call failed", "anonymized_text": text}


# =====================================================================
#  2. PII LEAKAGE CHECK (same as llm_prompt_anonymize.py)
# =====================================================================

def check_pii_leakage(anonymized_text: str, gold_entities: List[Dict]) -> Dict:
    """
    Check which ground-truth PII strings survive in the anonymized text.
    Uses word boundary matching for short strings (<=5 chars) to avoid
    false positives from substring matches.
    """
    leaked = []
    per_category = defaultdict(lambda: {"total": 0, "leaked": 0})
    check_text = anonymized_text.lower()

    for ent in gold_entities:
        label = ent["label"]
        pii_text = ent["text"]
        per_category[label]["total"] += 1
        search_text = pii_text.lower()

        # Always use word boundary matching to avoid false positives
        # from German compound words (e.g., "Inhaber" in "Alleininhaber")
        found = bool(re.search(r'\b' + re.escape(search_text) + r'\b', check_text))

        if found:
            idx = check_text.find(search_text)
            if idx == -1:
                m = re.search(r'\b' + re.escape(search_text) + r'\b', check_text)
                idx = m.start() if m else 0
            ctx_s = max(0, idx - 30)
            ctx_e = min(len(anonymized_text), idx + len(pii_text) + 30)
            leaked.append({
                "label": label,
                "text": pii_text,
                "context": f"...{anonymized_text[ctx_s:ctx_e]}..."
            })
            per_category[label]["leaked"] += 1

    per_cat = {
        l: {**c, "rate": round(c["leaked"] / max(c["total"], 1), 4)}
        for l, c in sorted(per_category.items())
    }

    return {
        "total_pii": len(gold_entities),
        "leaked_pii": len(leaked),
        "leakage_rate": round(len(leaked) / max(len(gold_entities), 1), 4),
        "leaked_entities": leaked,
        "per_category": per_cat,
    }


# =====================================================================
#  3. MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Datenwertschöpfung Anonymizer Baseline")
    print(f"  API: {API_URL}")
    print(f"  Options: {API_OPTIONS}")
    print(f"  Evaluation: PII leakage rate (same as prompt rewriting)")
    print("=" * 70)

    token = get_api_token()
    print(f"  API token loaded")

    # Load gold standard
    gold_records = load_label_studio_export(INPUT_PATH)
    print(f"  Total records: {len(gold_records)}")

    if SPLIT_IDS and os.path.exists(SPLIT_IDS):
        split_data = json.load(open(SPLIT_IDS, encoding="utf-8"))
        test_ids = set(split_data["test_ids"])
        gold_records = [r for r in gold_records if r["id"] in test_ids]
        print(f"  Filtered to test split: {len(gold_records)} records")

    if LIMIT:
        gold_records = gold_records[:LIMIT]
        print(f"  Limited to {LIMIT} documents")

    # Process each document
    results = []
    total_time = 0
    api_errors = 0
    all_leakage = []

    for rec in tqdm(gold_records, desc="  Anonymizing"):
        start_time = time.time()
        response = call_anonymizer_api(rec["text"], token)
        elapsed = time.time() - start_time
        total_time += elapsed

        if "error" in response:
            api_errors += 1

        anonymized_text = response.get("anonymized_text", rec["text"])

        # Check PII leakage
        leakage = check_pii_leakage(anonymized_text, rec["entities"])
        all_leakage.append(leakage)

        results.append({
            "id": rec["id"],
            "meta_temp": rec.get("meta_temp", "Unknown"),
            "original_text": rec["text"],
            "anonymized_text": anonymized_text,
            "gold_entities": rec["entities"],
            "leakage": leakage,
            "inference_time": round(elapsed, 3),
        })

        time.sleep(DELAY)

    n = len(gold_records)

    # ── Run statistics ──
    run_stats = {
        "total_records": n,
        "total_time": round(total_time, 1),
        "avg_time_per_doc": round(total_time / max(n, 1), 2),
        "api_errors": api_errors,
        "avg_leakage_rate": round(
            sum(l["leakage_rate"] for l in all_leakage) / max(len(all_leakage), 1), 4),
        "overall_leaked": sum(l["leaked_pii"] for l in all_leakage),
        "overall_total_pii": sum(l["total_pii"] for l in all_leakage),
    }

    # ── Save files ──

    # Full results
    full_path = os.path.join(OUTPUT_DIR, "datenwertschoepfung_full_results.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Run stats
    stats_path = os.path.join(OUTPUT_DIR, "datenwertschoepfung_run_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(run_stats, f, indent=2, ensure_ascii=False)

    # Predictions (for compatibility with semantic_preservation.py)
    predictions = [{"id": r["id"], "rewritten_text": r["anonymized_text"]} for r in results]
    pred_path = os.path.join(OUTPUT_DIR, "datenwertschoepfung_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # ── Generate report ──
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  DATENWERTSCHÖPFUNG ANONYMIZER — EVALUATION REPORT")
    report_lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"  API: {API_URL}")
    report_lines.append(f"  Options: {API_OPTIONS}")
    report_lines.append("=" * 80)
    report_lines.append("")

    ol = run_stats["overall_leaked"]
    ot = run_stats["overall_total_pii"]
    report_lines.append(f"  Documents: {n}")
    report_lines.append(f"  API errors: {api_errors}")
    report_lines.append(f"  Total time: {total_time:.1f}s ({total_time/max(n,1):.2f}s/doc)")
    report_lines.append(f"  Overall leakage: {ol}/{ot} ({ol/max(ot,1):.1%})")
    report_lines.append("")

    # Per-category leakage
    cat_totals = defaultdict(lambda: {"total": 0, "leaked": 0})
    for r in results:
        for label, counts in r["leakage"]["per_category"].items():
            cat_totals[label]["total"] += counts["total"]
            cat_totals[label]["leaked"] += counts["leaked"]

    report_lines.append("  PII LEAKAGE BY CATEGORY:")
    report_lines.append(f"  {'Category':<10} {'Leaked':>8} {'Total':>8} {'Rate':>8}")
    report_lines.append("  " + "-" * 38)
    for label in sorted(cat_totals.keys()):
        t = cat_totals[label]["total"]
        lk = cat_totals[label]["leaked"]
        rate = lk / max(t, 1)
        report_lines.append(f"  {label:<10} {lk:>8} {t:>8} {rate:>8.1%}")
    report_lines.append("")

    # Per-complexity leakage
    report_lines.append("  PII LEAKAGE BY COMPLEXITY:")
    report_lines.append(f"  {'Complexity':<10} {'Leaked':>8} {'Total':>8} {'Rate':>8}")
    report_lines.append("  " + "-" * 38)
    for comp in ["Low", "Medium", "High"]:
        comp_results = [r for r in results if r["meta_temp"] == comp]
        if comp_results:
            lk = sum(r["leakage"]["leaked_pii"] for r in comp_results)
            tt = sum(r["leakage"]["total_pii"] for r in comp_results)
            rate = lk / max(tt, 1)
            report_lines.append(f"  {comp:<10} {lk:>8} {tt:>8} {rate:>8.1%}")
    report_lines.append("")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "datenwertschoepfung_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ── Full document log (with original + API output + leakage) ──
    log_lines = []
    log_lines.append("=" * 90)
    log_lines.append("  DATENWERTSCHÖPFUNG ANONYMIZER — FULL DOCUMENT LOG")
    log_lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 90)

    for r in results:
        log_lines.append("")
        log_lines.append("-" * 90)
        log_lines.append(f"  Doc {r['id']} [{r['meta_temp']}] — "
                         f"Leakage: {r['leakage']['leaked_pii']}/{r['leakage']['total_pii']} "
                         f"({r['leakage']['leakage_rate']:.0%})")
        log_lines.append("-" * 90)

        log_lines.append(f"\n  ORIGINAL TEXT:")
        log_lines.append(f"  {r['original_text']}")

        log_lines.append(f"\n  API ANONYMIZED TEXT:")
        log_lines.append(f"  {r['anonymized_text']}")

        log_lines.append(f"\n  GOLD ENTITIES ({len(r['gold_entities'])}):")
        for e in r["gold_entities"]:
            log_lines.append(f"    [{e['label']:>6}] \"{e['text']}\"")

        if r["leakage"]["leaked_entities"]:
            log_lines.append(f"\n  LEAKED ENTITIES ({r['leakage']['leaked_pii']}):")
            for lk in r["leakage"]["leaked_entities"]:
                log_lines.append(f"    [{lk['label']:>6}] \"{lk['text']}\"")
                log_lines.append(f"           Context: {lk['context']}")
        else:
            log_lines.append(f"\n  NO LEAKAGE — all PII successfully anonymized")

    log_path = os.path.join(OUTPUT_DIR, "datenwertschoepfung_full_document_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # ── Side-by-side comparison ──
    side_lines = []
    side_lines.append("=" * 90)
    side_lines.append("  SIDE-BY-SIDE: Original vs. API Anonymized")
    side_lines.append("=" * 90)

    for r in results:
        side_lines.append("")
        side_lines.append(f"--- Doc {r['id']} [{r['meta_temp']}] | "
                          f"Leak: {r['leakage']['leaked_pii']}/{r['leakage']['total_pii']} ---")
        side_lines.append(f"ORIG: {r['original_text']}")
        side_lines.append(f"ANON: {r['anonymized_text']}")

    side_path = os.path.join(OUTPUT_DIR, "datenwertschoepfung_side_by_side.txt")
    with open(side_path, "w", encoding="utf-8") as f:
        f.write("\n".join(side_lines))

    # ── Print summary ──
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print(f"  Files: full_results.json, run_stats.json, predictions.json,")
    print(f"         report.txt, full_document_log.txt, side_by_side.txt")
    print(f"\n{'='*70}")
    print(f"  OVERALL PII LEAKAGE: {ol}/{ot} = {ol/max(ot,1):.1%}")
    print(f"{'='*70}")

    print(f"\n  By category:")
    for label in sorted(cat_totals.keys()):
        t = cat_totals[label]["total"]
        lk = cat_totals[label]["leaked"]
        rate = lk / max(t, 1)
        bar = "#" * int(rate * 20)
        print(f"    {label:<8} {lk:>4}/{t:<4} = {rate:>6.1%}  {bar}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
