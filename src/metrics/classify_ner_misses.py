"""
classify_ner_misses.py
======================
Classifies NER false negatives into three categories per pipeline and writes
a combined report plus per-pipeline CSV listings.

  1. BOUNDARY disagreement (same label, overlapping span)
     -> Model detected the entity but drew a different boundary.
  2. LABEL disagreement (different label, overlapping span)
     -> Model detected the entity but assigned a different type
        (e.g., JOB vs. EDU on "ETH-Ingenieur").
  3. TRUE OMISSION (no overlapping prediction)
     -> Model produced nothing for this region; the entity survives
        unmasked in the output.

Inputs are read directly from the canonical JSON files:
  - Predictions JSON per pipeline: list of {id, entities:[{start,end,label,text}]}
  - Gold (Label Studio export)   : list of {id, text, label:[{start,end,text,labels:[L]}]}
  - Split-IDs JSON               : {train_ids, dev_ids, test_ids}

Outputs:
  - <OUTPUT_DIR>/classify_ner_misses_report.txt   (human-readable combined report)
  - <OUTPUT_DIR>/<pipeline_slug>_misses.csv       (per-pipeline FN listing)

"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GOLD_PATH  = os.path.join(BASE_DIR, "data", "label_studio",
                          "20260302_Export_Label_Studio_Client_Notes.json")
SPLIT_IDS  = os.path.join(BASE_DIR, "results", "bert_finetuned", "split_ids.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "miss_classification")
SPLIT_NAME = "test"   # "train" | "dev" | "test"

# Tag-and-replace pipelines: {pipeline name: predictions JSON path}.
# Prompt-rewrite and external-API pipelines are excluded because they
# do not emit entity spans.
PREDICTIONS = {
    # Classical baselines
    "spaCy + Regex":                       os.path.join(BASE_DIR, "results", "classical_baselines", "spacy", "spacy_predictions.json"),
    "BERT (pretrained) + Regex":           os.path.join(BASE_DIR, "results", "classical_baselines", "bert", "bert_predictions.json"),
    "Presidio":                            os.path.join(BASE_DIR, "results", "presidio_baseline", "presidio_predictions.json"),
    # Fine-tuned encoder
    "BERT Fine-Tuned":                     os.path.join(BASE_DIR, "results", "bert_finetuned", "bert_finetuned_predictions.json"),
    # LLM tag-and-replace
    "LLM Llama-3 [few-shot +verify]":      os.path.join(BASE_DIR, "results", "llm_baselines", "llm_meta_llama_3_8b_instruct_few_shot_verified_predictions.json"),
    "LLM Qwen2.5 [few-shot +verify]":      os.path.join(BASE_DIR, "results", "llm_baselines", "llm_qwen2.5_7b_instruct_few_shot_verified_predictions.json"),
    "LLM SauerkrautLM [few-shot +verify]": os.path.join(BASE_DIR, "results", "llm_baselines", "llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_verified_predictions.json"),
    # LLM fine-tuned (QLoRA)
    "LLM Llama-3 [fine-tuned]":            os.path.join(BASE_DIR, "results", "llm_finetuned", "llm_finetuned_meta_llama_3_8b_instruct", "llm_finetuned_meta_llama_3_8b_instruct_predictions.json"),
}

# Number of example misses to include in the report per category
EXAMPLES_PER_CATEGORY = 10

# Write per-pipeline CSV listings of every false negative with its category
WRITE_CSV = True


# =====================================================================
#  IMPLEMENTATION
# =====================================================================

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Iterable


# --------------------------------------------------------------------- #
# Data classes                                                          #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Entity:
    """A single labelled span within a document."""
    label: str
    text: str
    start: int
    end: int

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Document:
    """All extracted entities from one document, partitioned by outcome."""
    doc_id: str
    exact: list = field(default_factory=list)
    false_negatives: list = field(default_factory=list)
    false_positives: list = field(default_factory=list)


@dataclass
class Classification:
    """The result of classifying a single false negative."""
    doc_id: str
    gold: Entity
    category: str  # "boundary" | "label" | "omission"
    overlapping_fps: list = field(default_factory=list)


# --------------------------------------------------------------------- #
# Input parsing                                                         #
# --------------------------------------------------------------------- #

def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_predictions(
    predictions_path: str,
    gold_path: str,
    split_ids_path: str,
    split_name: str = "test",
) -> list:
    """Match predictions to gold and return a list of Document objects.

    Exact match : same (start, end, label).
    False neg.  : gold span with no exact-matching prediction.
    False pos.  : predicted span with no exact-matching gold annotation.
    """
    predictions = _load_json(predictions_path)
    gold = _load_json(gold_path)
    split_data = _load_json(split_ids_path)

    key = f"{split_name}_ids"
    if key not in split_data:
        raise KeyError(f"split-ids JSON has no key {key!r}")
    keep_ids: set = set(split_data[key])

    pred_by_id = {item["id"]: item.get("entities", []) for item in predictions}
    gold_by_id = {item["id"]: item.get("label", []) or [] for item in gold}

    documents: list = []
    for doc_id in sorted(d for d in keep_ids):
        gold_entities = gold_by_id.get(doc_id, [])
        pred_entities = pred_by_id.get(doc_id, [])

        gold_map: dict = {}
        for g in gold_entities:
            labels = g.get("labels") or []
            if not labels:
                continue
            gold_map[(g["start"], g["end"], labels[0])] = g

        pred_map: dict = {}
        for p in pred_entities:
            pred_map[(p["start"], p["end"], p["label"])] = p

        exact_keys = gold_map.keys() & pred_map.keys()
        fn_keys = gold_map.keys() - exact_keys
        fp_keys = pred_map.keys() - exact_keys

        doc = Document(doc_id=str(doc_id))
        for k in exact_keys:
            g = gold_map[k]
            doc.exact.append(Entity(label=k[2], text=g.get("text", ""), start=k[0], end=k[1]))
        for k in fn_keys:
            g = gold_map[k]
            doc.false_negatives.append(Entity(label=k[2], text=g.get("text", ""), start=k[0], end=k[1]))
        for k in fp_keys:
            p = pred_map[k]
            doc.false_positives.append(Entity(label=k[2], text=p.get("text", ""), start=k[0], end=k[1]))
        documents.append(doc)

    return documents


# --------------------------------------------------------------------- #
# Classification                                                        #
# --------------------------------------------------------------------- #

def classify_false_negative(fn: Entity, fps):
    """Decide the category for one gold false negative within its document.

    Priority:
        1. Same-label overlapping FP -> 'boundary'
        2. Otherwise, any overlapping FP -> 'label'
        3. Otherwise -> 'omission'
    """
    same_label_overlaps = [fp for fp in fps if fp.label == fn.label and fn.overlaps(fp)]
    if same_label_overlaps:
        return "boundary", same_label_overlaps

    diff_label_overlaps = [fp for fp in fps if fp.label != fn.label and fn.overlaps(fp)]
    if diff_label_overlaps:
        return "label", diff_label_overlaps

    return "omission", []


def classify_all(documents):
    results = []
    for doc in documents:
        for fn in doc.false_negatives:
            category, overlapping = classify_false_negative(fn, doc.false_positives)
            results.append(Classification(
                doc_id=doc.doc_id, gold=fn,
                category=category, overlapping_fps=overlapping,
            ))
    return results


# --------------------------------------------------------------------- #
# Report formatting                                                     #
# --------------------------------------------------------------------- #

CATEGORY_LABELS = {
    "boundary": "Boundary disagreement (same label, overlapping span)",
    "label":    "Label disagreement (overlapping span, different label)",
    "omission": "True omission (no overlapping prediction)",
}


def format_pipeline_section(pipeline_name: str, documents, results) -> str:
    """Build the per-pipeline report block."""
    buf = StringIO()
    total_gold = sum(len(d.exact) + len(d.false_negatives) for d in documents)
    total_tp = sum(len(d.exact) for d in documents)
    total_fn = sum(len(d.false_negatives) for d in documents)
    total_fp = sum(len(d.false_positives) for d in documents)

    buf.write("##############################################################################\n")
    buf.write(f"  {pipeline_name}\n")
    buf.write("##############################################################################\n\n")

    buf.write(f"  Documents parsed : {len(documents)}\n")
    buf.write(f"  Gold entities    : {total_gold}\n")
    buf.write(f"  Exact matches    : {total_tp}\n")
    buf.write(f"  False negatives  : {total_fn}\n")
    buf.write(f"  False positives  : {total_fp}\n")
    if total_gold:
        buf.write(f"  Strict recall    : {total_tp / total_gold:.4f}\n")
    buf.write("\n")

    by_category = Counter(r.category for r in results)
    buf.write("  False negatives by category\n")
    buf.write("  " + "-" * 72 + "\n")
    for key in ("boundary", "label", "omission"):
        n = by_category.get(key, 0)
        share = (n / total_fn * 100) if total_fn else 0.0
        buf.write(f"  {CATEGORY_LABELS[key]:<55} {n:>4} ({share:>4.1f}%)\n")
    buf.write("\n")

    gold_by_label: Counter = Counter()
    for doc in documents:
        for entity in doc.exact + doc.false_negatives:
            gold_by_label[entity.label] += 1

    misses_by_label: dict = defaultdict(Counter)
    for r in results:
        misses_by_label[r.gold.label][r.category] += 1

    buf.write(f"  {'Label':<10}{'Gold':>6}{'Miss':>6}{'Recall':>9}"
              f"{'Bound.':>8}{'Label':>8}{'Omiss.':>8}\n")
    buf.write("  " + "-" * 55 + "\n")
    for label, gold_n in sorted(gold_by_label.items(), key=lambda kv: -kv[1]):
        counts = misses_by_label.get(label, Counter())
        miss_n = sum(counts.values())
        recall = (gold_n - miss_n) / gold_n if gold_n else 0
        buf.write(
            f"  {label:<10}{gold_n:>6}{miss_n:>6}{recall*100:>8.1f}%"
            f"{counts.get('boundary', 0):>8}"
            f"{counts.get('label', 0):>8}"
            f"{counts.get('omission', 0):>8}\n"
        )
    buf.write("\n")

    return buf.getvalue()


def format_examples_section(pipeline_name: str, results, limit: int) -> str:
    """Build a short example dump for each category, for manual inspection."""
    buf = StringIO()
    buf.write(f"  Examples (up to {limit} per category) for {pipeline_name}\n")
    buf.write("  " + "-" * 72 + "\n")
    for category in ("boundary", "label", "omission"):
        filtered = [r for r in results if r.category == category]
        buf.write(f"\n  >>> {CATEGORY_LABELS[category]} ({len(filtered)} total) <<<\n")
        for r in filtered[:limit]:
            buf.write(f"    Doc {r.doc_id}: GOLD [{r.gold.label}] '{r.gold.text}'"
                      f"  ({r.gold.start}:{r.gold.end})\n")
            for fp in r.overlapping_fps:
                buf.write(f"          MODEL [{fp.label}] '{fp.text}'"
                          f"  ({fp.start}:{fp.end})\n")
        if len(filtered) > limit:
            buf.write(f"    ... and {len(filtered) - limit} more\n")
    buf.write("\n")
    return buf.getvalue()


def format_summary_table(per_pipeline_stats: list) -> str:
    """Cross-pipeline comparison table (placed at the top of the report)."""
    buf = StringIO()
    buf.write("##############################################################################\n")
    buf.write("  SUMMARY: Composition of missed entities, all pipelines\n")
    buf.write("##############################################################################\n\n")
    buf.write(f"  {'Pipeline':<40}{'Gold':>6}{'FN':>6}{'Recall':>9}"
              f"{'Bound.':>8}{'Label':>8}{'Omiss.':>8}\n")
    buf.write("  " + "-" * 85 + "\n")
    for s in per_pipeline_stats:
        recall = (s["total_gold"] - s["total_fn"]) / s["total_gold"] if s["total_gold"] else 0
        buf.write(
            f"  {s['name']:<40}"
            f"{s['total_gold']:>6}{s['total_fn']:>6}{recall*100:>8.1f}%"
            f"{s['boundary']:>8}{s['label']:>8}{s['omission']:>8}\n"
        )
    buf.write("\n")
    buf.write("  Recall is strict (exact span and label match).\n")
    buf.write("  Bound./Label/Omiss. counts of false negatives by category.\n")
    buf.write("\n")
    return buf.getvalue()


# --------------------------------------------------------------------- #
# CSV writer                                                            #
# --------------------------------------------------------------------- #

def _slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def write_csv(results, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "category", "gold_label", "gold_text",
            "gold_start", "gold_end",
            "overlapping_fp_labels", "overlapping_fp_texts",
        ])
        for r in results:
            writer.writerow([
                r.doc_id, r.category, r.gold.label, r.gold.text,
                r.gold.start, r.gold.end,
                "|".join(fp.label for fp in r.overlapping_fps),
                "|".join(fp.text for fp in r.overlapping_fps),
            ])


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GOLD_PATH):
        print(f"Gold file not found: {GOLD_PATH}", file=sys.stderr)
        return 1
    if not os.path.exists(SPLIT_IDS):
        print(f"Split-IDs file not found: {SPLIT_IDS}", file=sys.stderr)
        return 1

    sections: list = []
    per_pipeline_stats: list = []
    example_sections: list = []

    sections.append(
        "==============================================================================\n"
        "  NER MISS-CLASSIFICATION REPORT\n"
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Split: {SPLIT_NAME}\n"
        f"  Gold: {os.path.relpath(GOLD_PATH, BASE_DIR)}\n"
        f"  Split-IDs: {os.path.relpath(SPLIT_IDS, BASE_DIR)}\n"
        "  Category definitions:\n"
        "    Boundary disagreement: same label, overlapping span\n"
        "    Label disagreement   : different label, overlapping span\n"
        "    True omission        : no overlapping prediction\n"
        "==============================================================================\n\n"
    )

    for pipeline_name, pred_path in PREDICTIONS.items():
        print(f"Processing {pipeline_name} ...")
        if not os.path.exists(pred_path):
            msg = f"  Skipping {pipeline_name}: predictions file not found ({pred_path})\n\n"
            print(msg)
            sections.append(msg)
            continue

        try:
            documents = parse_predictions(
                predictions_path=pred_path,
                gold_path=GOLD_PATH,
                split_ids_path=SPLIT_IDS,
                split_name=SPLIT_NAME,
            )
        except Exception as exc:
            msg = f"  Skipping {pipeline_name}: {type(exc).__name__}: {exc}\n\n"
            print(msg)
            sections.append(msg)
            continue

        results = classify_all(documents)
        sections.append(format_pipeline_section(pipeline_name, documents, results))
        example_sections.append(format_examples_section(pipeline_name, results, EXAMPLES_PER_CATEGORY))

        by_category = Counter(r.category for r in results)
        per_pipeline_stats.append({
            "name": pipeline_name,
            "total_gold": sum(len(d.exact) + len(d.false_negatives) for d in documents),
            "total_fn": sum(len(d.false_negatives) for d in documents),
            "boundary": by_category.get("boundary", 0),
            "label": by_category.get("label", 0),
            "omission": by_category.get("omission", 0),
        })

        if WRITE_CSV:
            csv_path = os.path.join(OUTPUT_DIR, f"{_slugify(pipeline_name)}_misses.csv")
            write_csv(results, csv_path)
            print(f"  Wrote {len(results)} miss rows to {csv_path}")

    # Compose final report: header, cross-pipeline summary, then per-pipeline detail,
    # then example dumps at the end (useful for manual inspection but lengthy).
    report = sections[0] + format_summary_table(per_pipeline_stats) + "".join(sections[1:])
    report += "##############################################################################\n"
    report += "  APPENDIX: Example misses per category, per pipeline\n"
    report += "##############################################################################\n\n"
    report += "".join(example_sections)

    print()
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classify_ner_misses_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
