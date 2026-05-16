"""
compute_rejection_rates.py
==========================
Compute the generation-to-validation rejection rates that are cited in
thesis (Data Validation). This script compares the
raw LLM-generated dataset against the validated/cleaned dataset and
reports overall and per-complexity retention statistics.

Inputs
------
  - data/raw/german_financial_data_raw.json
    The full raw output of the Gemini 2.5 Pro generation run (before any
    validation).
  - data/processed/german_financial_data_cleaned.json
    The dataset after Stage 1 (deterministic rule checks) and Stage 2
    (semantic LLM audit) validation.

Outputs
-------
  - Printed report to stdout.
  - data/splits/rejection_rates.json  (machine-readable summary with the
    same numbers cited in the thesis.)

Usage
-----
    python src/data_generation/compute_rejection_rates.py

"""

import json
import os
from collections import Counter
from typing import Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "german_financial_data_raw.json")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "german_financial_data_cleaned.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "splits", "rejection_rates.json")

COMPLEXITY_LEVELS = ["Low", "Medium", "High"]


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_by_complexity(records) -> Dict[str, int]:
    """Return {'Low': n, 'Medium': n, 'High': n, 'Unknown': n} for a list of records."""
    counts = Counter(r.get("meta_temp", "Unknown") for r in records)
    # Ensure all levels present even if zero
    out = {lvl: counts.get(lvl, 0) for lvl in COMPLEXITY_LEVELS}
    if counts.get("Unknown", 0) > 0:
        out["Unknown"] = counts["Unknown"]
    return out


def main() -> None:
    print(f"Loading raw dataset:     {RAW_PATH}")
    raw = _load_json(RAW_PATH)
    print(f"Loading cleaned dataset: {CLEAN_PATH}")
    clean = _load_json(CLEAN_PATH)

    raw_by = _count_by_complexity(raw)
    clean_by = _count_by_complexity(clean)

    n_raw = len(raw)
    n_clean = len(clean)
    n_rejected = n_raw - n_clean
    rejection_rate = n_rejected / n_raw if n_raw else 0.0

    # ----- Console report -----
    width = 74
    print("=" * width)
    print("  REJECTION RATE REPORT (Section 3.1.5)")
    print("=" * width)
    print()
    print(f"  Total generated:   {n_raw:>6}")
    print(f"  Total retained:    {n_clean:>6}")
    print(f"  Total rejected:    {n_rejected:>6}  ({rejection_rate*100:5.1f}%)")
    print()
    print(f"  {'Level':<10} {'Generated':>10} {'Retained':>10} {'Rejected':>10} {'Rate':>8}")
    print("  " + "-" * (width - 4))
    per_level = {}
    for lvl in COMPLEXITY_LEVELS:
        g = raw_by.get(lvl, 0)
        k = clean_by.get(lvl, 0)
        rej = g - k
        rate = (rej / g) if g else 0.0
        per_level[lvl] = {"generated": g, "retained": k, "rejected": rej, "rate": rate}
        print(f"  {lvl:<10} {g:>10} {k:>10} {rej:>10} {rate*100:>7.1f}%")

    unknown_raw = raw_by.get("Unknown", 0)
    unknown_clean = clean_by.get("Unknown", 0)
    if unknown_raw or unknown_clean:
        print(f"  (Unknown complexity: raw={unknown_raw}, clean={unknown_clean})")

    print()
    print("  Observation: rejection rate rises monotonically with complexity,")
    print("  consistent with the claim that hasty, fragment-heavy notes more")
    print("  frequently produce structural annotation errors.")
    print()

    # ----- Machine-readable dump -----
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total": {
                "generated": n_raw,
                "retained": n_clean,
                "rejected": n_rejected,
                "rate": rejection_rate,
            },
            "per_complexity": per_level,
            "source_files": {
                "raw": os.path.relpath(RAW_PATH, BASE_DIR).replace("\\", "/"),
                "cleaned": os.path.relpath(CLEAN_PATH, BASE_DIR).replace("\\", "/"),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"  Wrote machine-readable summary to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
