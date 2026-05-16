"""
export_data_splits.py
=====================
Export the Label Studio annotated dataset as train / dev / test files in
both JSON and Excel (xlsx) formats, matching the split used for all
downstream evaluations in the thesis (see results/bert_finetuned/split_ids.json).

Inputs
------
  - data/label_studio/20260302_Export_Label_Studio_Client_Notes.json
    The full 2,542-document annotated corpus exported from Label Studio.
  - results/bert_finetuned/split_ids.json
    The canonical train/dev/test split (1270 / 634 / 638).

Outputs (written to data/splits/)
---------------------------------
  - {train,dev,test}.json
        Full records (id, text, raw_text, label, complexity).
  - {train,dev,test}.xlsx
        Tabular per-document view: id, complexity, char/word counts,
        total entity count, per-category entity counts, truncated preview.
  - split_summary.xlsx
        Summary statistics: docs per split, docs per complexity, average
        text length, total entities per category per split.

Usage
-----
    python src/data_generation/export_data_splits.py

"""

import json
import os
from collections import Counter
from typing import Dict, List

import pandas as pd


# =====================================================================
#  PATHS
# =====================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FULL_DATA_PATH = os.path.join(
    BASE_DIR, "data", "label_studio",
    "20260302_Export_Label_Studio_Client_Notes.json",
)
SPLIT_IDS_PATH = os.path.join(
    BASE_DIR, "results", "bert_finetuned", "split_ids.json",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "splits")

# Preview text length in the Excel tabular view
PREVIEW_CHARS = 200

# Fixed category order (12 PII categories across Tiers 1, 2, 3)
CATEGORY_ORDER = [
    # Tier 1 — Direct NER
    "PER", "LOC", "ORG",
    # Tier 2 — Structured
    "DATE", "EMAIL", "PHONE", "IBAN", "MONEY",
    # Tier 3 — Quasi-identifiers
    "JOB", "AGE", "NATION", "EDU",
]


# =====================================================================
#  HELPERS
# =====================================================================

def _entity_category(entity: Dict) -> str:
    """Label Studio stores the category under 'labels' (list of str)."""
    labels = entity.get("labels") or []
    return labels[0] if labels else "UNKNOWN"


def _mask_text(text: str, entities: List[Dict]) -> str:
    """Replace every annotated PII span with ``[LABEL]``.

    Spans are applied in reverse order by ``start`` so earlier offsets stay
    valid as we rewrite the string. Overlapping spans (rare in Label Studio
    exports) are handled by skipping any span fully contained in an already
    processed region.
    """
    if not text or not entities:
        return text
    # Sort descending by start; ties on start sorted by larger end first
    sorted_ents = sorted(
        entities,
        key=lambda e: (-int(e.get("start", 0)), -int(e.get("end", 0))),
    )
    masked = text
    last_start = None
    for e in sorted_ents:
        start = int(e.get("start", 0))
        end = int(e.get("end", 0))
        if start >= end or end > len(masked):
            continue
        # If this span falls inside an already processed (later-starting)
        # region, skip it to avoid double replacement.
        if last_start is not None and end > last_start:
            continue
        label = _entity_category(e)
        masked = masked[:start] + f"[{label}]" + masked[end:]
        last_start = start
    return masked


def _record_to_row(record: Dict) -> Dict:
    """Flatten one annotated record into a tabular row."""
    text = record.get("text", "") or ""
    entities = record.get("label", []) or []
    cat_counts = Counter(_entity_category(e) for e in entities)
    masked = _mask_text(text, entities)

    row = {
        "id": record.get("id"),
        "complexity": record.get("meta_temp"),
        "num_chars": len(text),
        "num_words": len(text.split()),
        "num_entities": len(entities),
    }
    for cat in CATEGORY_ORDER:
        row[f"n_{cat}"] = cat_counts.get(cat, 0)
    row["preview"] = (text[:PREVIEW_CHARS] + "…") if len(text) > PREVIEW_CHARS else text
    row["original_text"] = text
    row["masked_text"] = masked
    return row


def _json_safe_record(record: Dict) -> Dict:
    """Pick the fields a downstream consumer actually needs."""
    return {
        "id": record.get("id"),
        "complexity": record.get("meta_temp"),
        "text": record.get("text", ""),
        "raw_text": record.get("raw_text", ""),
        "entities": record.get("label", []) or [],
    }


def _build_summary(splits: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Per-split, per-complexity summary counts + entity category totals."""
    rows = []
    for split_name, records in splits.items():
        total_entities = 0
        cat_totals = Counter()
        complexity_counts = Counter()
        char_lengths = []
        for rec in records:
            complexity_counts[rec.get("meta_temp")] += 1
            char_lengths.append(len(rec.get("text", "") or ""))
            for e in rec.get("label", []) or []:
                cat = _entity_category(e)
                cat_totals[cat] += 1
                total_entities += 1

        row = {
            "split": split_name,
            "n_docs": len(records),
            "n_low": complexity_counts.get("Low", 0),
            "n_medium": complexity_counts.get("Medium", 0),
            "n_high": complexity_counts.get("High", 0),
            "avg_chars_per_doc": (sum(char_lengths) / len(char_lengths)) if char_lengths else 0.0,
            "total_entities": total_entities,
            "entities_per_doc": (total_entities / len(records)) if records else 0.0,
        }
        for cat in CATEGORY_ORDER:
            row[f"total_{cat}"] = cat_totals.get(cat, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    # round averages to 1 decimal for readability
    df["avg_chars_per_doc"] = df["avg_chars_per_doc"].round(1)
    df["entities_per_doc"] = df["entities_per_doc"].round(2)
    return df


# =====================================================================
#  MAIN
# =====================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading full dataset from: {FULL_DATA_PATH}")
    with open(FULL_DATA_PATH, encoding="utf-8") as f:
        all_records = json.load(f)
    print(f"  {len(all_records)} records")

    print(f"Loading split definition from: {SPLIT_IDS_PATH}")
    with open(SPLIT_IDS_PATH, encoding="utf-8") as f:
        split_def = json.load(f)
    print(f"  train={len(split_def['train_ids'])} "
          f"dev={len(split_def['dev_ids'])} "
          f"test={len(split_def['test_ids'])} "
          f"(seed={split_def.get('seed')})")

    by_id = {r["id"]: r for r in all_records}

    splits = {
        "train": [by_id[i] for i in split_def["train_ids"] if i in by_id],
        "dev":   [by_id[i] for i in split_def["dev_ids"]   if i in by_id],
        "test":  [by_id[i] for i in split_def["test_ids"]  if i in by_id],
    }

    # ----- JSON files -----
    for split_name, records in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        payload = [_json_safe_record(r) for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  wrote {path} ({len(payload)} records)")

    # ----- Excel per-split tabular view -----
    for split_name, records in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{split_name}.xlsx")
        df = pd.DataFrame([_record_to_row(r) for r in records])
        # Ensure consistent column order
        ordered_cols = (
            ["id", "complexity", "num_chars", "num_words", "num_entities"]
            + [f"n_{c}" for c in CATEGORY_ORDER]
            + ["preview", "original_text", "masked_text"]
        )
        df = df[ordered_cols]
        df.to_excel(path, sheet_name=split_name, index=False)
        print(f"  wrote {path} ({len(df)} rows × {len(df.columns)} cols)")

    # ----- Summary spreadsheet -----
    summary_df = _build_summary(splits)
    summary_path = os.path.join(OUTPUT_DIR, "split_summary.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        # also write the raw split_ids for reproducibility
        pd.DataFrame({
            "split": (["train"] * len(split_def["train_ids"])
                      + ["dev"]   * len(split_def["dev_ids"])
                      + ["test"]  * len(split_def["test_ids"])),
            "doc_id": (list(split_def["train_ids"])
                       + list(split_def["dev_ids"])
                       + list(split_def["test_ids"])),
        }).to_excel(writer, sheet_name="doc_ids", index=False)
    print(f"  wrote {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
