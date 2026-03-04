"""
evaluation_utils.py
====================
Shared evaluation utilities for the thesis:
  - Data loading from Label Studio export
  - Entity-level matching (strict & relaxed)
  - Precision / Recall / F1 computation per category and overall
  - Tiered evaluation (direct, structured, quasi-identifiers)
  - Report generation

Author: André Kuhn – Master Thesis (MScIDS, HSLU)
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set


# ─────────────────────────────────────────────
#  1. ENTITY TIER DEFINITIONS
# ─────────────────────────────────────────────
# The thesis groups PII entities into three detection-difficulty tiers.
# This lets us measure where classical NLP tools fall short vs. LLMs.

# Tier 1: Standard NER categories that spaCy and BERT handle well
TIER1_DIRECT_NER: Set[str] = {"PER", "LOC", "ORG"}

# Tier 2: Structured identifiers detectable via regular expressions
TIER2_STRUCTURED: Set[str] = {"IBAN", "EMAIL", "PHONE", "DATE", "MONEY"}

# Tier 3: Quasi-identifiers that require contextual understanding (LLM territory)
TIER3_QUASI: Set[str] = {"JOB", "AGE", "NATION", "EDU"}

# Combined set of every label used in the annotation schema
ALL_LABELS: Set[str] = TIER1_DIRECT_NER | TIER2_STRUCTURED | TIER3_QUASI

# Human-readable tier names mapped to their label sets.
# Used by `evaluate_tiered()` to produce a per-tier report.
TIER_MAP: Dict[str, Set[str]] = {
    "Tier 1 – Direct NER": TIER1_DIRECT_NER,
    "Tier 2 – Structured (Regex)": TIER2_STRUCTURED,
    "Tier 3 – Quasi-Identifiers": TIER3_QUASI,
    "All Categories": ALL_LABELS,
}

# spaCy's German model uses different label names than our schema.
# This map translates them.  `None` means "ignore this label".
SPACY_LABEL_MAP: Dict[str, Optional[str]] = {
    "PER": "PER",
    "PERSON": "PER",     # some spaCy models use PERSON instead of PER
    "LOC": "LOC",
    "GPE": "LOC",        # GPE (geopolitical entity) maps to our LOC
    "ORG": "ORG",
    "MISC": None,        # MISC is too ambiguous to map reliably
}


# ─────────────────────────────────────────────
#  2. DATA LOADING
# ─────────────────────────────────────────────

def load_label_studio_export(filepath: str) -> List[Dict]:
    """
    Load a Label Studio JSON export and normalise it into a simple format.

    Each returned record looks like:
        {
            "id":        42,
            "text":      "Sehr geehrter Herr Müller ...",
            "meta_temp": "High",                         # generation style
            "entities":  [
                {"start": 20, "end": 26, "label": "PER", "text": "Müller"},
                ...
            ]
        }

    Only entities whose label is in ALL_LABELS are kept; everything else
    (e.g. a stray "MISC" annotation) is silently dropped.
    """
    with open(filepath, "r", encoding="utf-8") as file_handle:
        raw_export = json.load(file_handle)

    records = []

    for entry in raw_export:
        # Collect every valid entity annotation from this document
        entities = []
        for annotation in entry.get("label", []):
            # Label Studio stores labels as a list; we only use the first one
            label = annotation["labels"][0] if annotation.get("labels") else None
            if label and label in ALL_LABELS:
                entities.append({
                    "start": annotation["start"],
                    "end":   annotation["end"],
                    "label": label,
                    "text":  annotation["text"],
                })

        records.append({
            "id":        entry["id"],
            "text":      entry["text"],
            "meta_temp": entry.get("meta_temp", "Unknown"),
            "entities":  entities,
        })

    return records


# ─────────────────────────────────────────────
#  3. ENTITY-LEVEL MATCHING
# ─────────────────────────────────────────────

def match_entities_strict(
    gold_entities: List[Dict],
    pred_entities: List[Dict],
    label_filter: Optional[Set[str]] = None,
) -> Tuple[int, int, int]:
    """
    Strict matching: a prediction is correct only if *both* its character
    span (start, end) and its label match a gold entity exactly.

    Args:
        gold_entities:  ground-truth entity dicts (start, end, label, text)
        pred_entities:  predicted entity dicts
        label_filter:   if given, only entities with these labels are compared

    Returns:
        (true_positives, false_positives, false_negatives)
    """

    def _to_span_set(entities: List[Dict]) -> Set[Tuple[int, int, str]]:
        """Convert a list of entity dicts to a set of (start, end, label) tuples."""
        spans = set()
        for entity in entities:
            label = entity["label"]
            if label_filter and label not in label_filter:
                continue
            spans.add((entity["start"], entity["end"], label))
        return spans

    gold_spans = _to_span_set(gold_entities)
    pred_spans = _to_span_set(pred_entities)

    true_positives  = len(gold_spans & pred_spans)       # in both sets
    false_positives = len(pred_spans - gold_spans)       # predicted but not in gold
    false_negatives = len(gold_spans - pred_spans)       # in gold but not predicted

    return true_positives, false_positives, false_negatives


def match_entities_relaxed(
    gold_entities: List[Dict],
    pred_entities: List[Dict],
    label_filter: Optional[Set[str]] = None,
    overlap_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Relaxed matching: the label must still match exactly, but the character
    spans only need to overlap by at least `overlap_threshold` of the
    *shorter* span.  This is useful because NER models often predict
    slightly different boundaries (e.g. "Dr. Müller" vs. "Müller").

    Each gold and each predicted entity can participate in at most one match.

    Returns:
        (true_positives, false_positives, false_negatives)
    """

    def _apply_filter(entities: List[Dict]) -> List[Dict]:
        if label_filter is None:
            return entities
        return [e for e in entities if e["label"] in label_filter]

    filtered_golds = _apply_filter(gold_entities)
    filtered_preds = _apply_filter(pred_entities)

    matched_gold_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()

    for gold_idx, gold in enumerate(filtered_golds):
        best_overlap_ratio = 0.0
        best_pred_idx: Optional[int] = None

        for pred_idx, pred in enumerate(filtered_preds):
            # Skip already-matched predictions (one-to-one matching)
            if pred_idx in matched_pred_indices:
                continue

            # Labels must agree
            if gold["label"] != pred["label"]:
                continue

            # Calculate character-level overlap
            overlap_start = max(gold["start"], pred["start"])
            overlap_end   = min(gold["end"],   pred["end"])
            if overlap_start >= overlap_end:
                continue  # no overlap at all

            overlap_length = overlap_end - overlap_start
            shorter_span   = min(
                gold["end"] - gold["start"],
                pred["end"] - pred["start"],
            )
            if shorter_span == 0:
                continue

            ratio = overlap_length / shorter_span
            if ratio >= overlap_threshold and ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_pred_idx = pred_idx

        # If we found a valid match for this gold entity, record it
        if best_pred_idx is not None:
            matched_gold_indices.add(gold_idx)
            matched_pred_indices.add(best_pred_idx)

    true_positives  = len(matched_gold_indices)
    false_positives = len(filtered_preds) - len(matched_pred_indices)
    false_negatives = len(filtered_golds) - len(matched_gold_indices)

    return true_positives, false_positives, false_negatives


# ─────────────────────────────────────────────
#  4. METRICS COMPUTATION
# ─────────────────────────────────────────────

def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1-score from raw counts.

    Returns a dict like:
        {"precision": 0.85, "recall": 0.90, "f1": 0.87, "tp": 17, "fp": 3, "fn": 2}
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


def evaluate_predictions(
    gold_records: List[Dict],
    pred_records: List[Dict],
    matching: str = "strict",
    label_filter: Optional[Set[str]] = None,
) -> Dict:
    """
    Evaluate predicted entities against the gold standard across all documents.

    Args:
        gold_records:  list of {id, text, entities: [{start, end, label, text}]}
        pred_records:  list of {id, entities: [{start, end, label}]}  (same IDs)
        matching:      "strict" (exact span) or "relaxed" (overlap-based)
        label_filter:  if given, only these labels are evaluated

    Returns:
        Dict with:
            - "per_category": {label: {precision, recall, f1, tp, fp, fn}}
            - "overall":      {precision, recall, f1, tp, fp, fn}
            - "matching":     "strict" or "relaxed"
    """
    # Build a lookup so we can find predictions by document ID
    predictions_by_id = {
        record["id"]: record.get("entities", [])
        for record in pred_records
    }

    # Choose the right matching function
    match_fn = match_entities_strict if matching == "strict" else match_entities_relaxed

    # Which labels are we scoring?
    active_labels = label_filter if label_filter else ALL_LABELS

    # Accumulators: per-category and overall
    category_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    overall_counts  = {"tp": 0, "fp": 0, "fn": 0}

    # Walk through every document
    for gold_record in gold_records:
        doc_id      = gold_record["id"]
        gold_ents   = gold_record["entities"]
        pred_ents   = predictions_by_id.get(doc_id, [])

        # Score each label individually
        for label in active_labels:
            tp, fp, fn = match_fn(gold_ents, pred_ents, label_filter={label})
            category_counts[label]["tp"] += tp
            category_counts[label]["fp"] += fp
            category_counts[label]["fn"] += fn

        # Score all active labels together for the "overall" row
        tp, fp, fn = match_fn(gold_ents, pred_ents, label_filter=active_labels)
        overall_counts["tp"] += tp
        overall_counts["fp"] += fp
        overall_counts["fn"] += fn

    # Convert raw counts to precision / recall / F1
    per_category_metrics = {
        label: compute_prf(counts["tp"], counts["fp"], counts["fn"])
        for label, counts in sorted(category_counts.items())
    }

    return {
        "per_category": per_category_metrics,
        "overall":      compute_prf(overall_counts["tp"], overall_counts["fp"], overall_counts["fn"]),
        "matching":     matching,
    }


def evaluate_tiered(
    gold_records: List[Dict],
    pred_records: List[Dict],
    matching: str = "strict",
) -> Dict:
    """
    Run evaluation once per tier (Tier 1, 2, 3, and All).

    Returns a dict keyed by tier name, e.g.:
        {
            "Tier 1 – Direct NER":         { ... evaluation results ... },
            "Tier 2 – Structured (Regex)":  { ... },
            ...
        }
    """
    return {
        tier_name: evaluate_predictions(gold_records, pred_records, matching=matching, label_filter=tier_labels)
        for tier_name, tier_labels in TIER_MAP.items()
    }


# ─────────────────────────────────────────────
#  5. REPORT FORMATTING
# ─────────────────────────────────────────────

def format_results_table(results: Dict, title: str = "") -> str:
    """
    Format a single evaluation result (one tier, one matching mode)
    as a human-readable text table.
    """
    lines = []

    if title:
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 70}")

    # Table header
    lines.append(f"  Matching: {results['matching']}")
    lines.append(f"  {'Category':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    lines.append(f"  {'-' * 62}")

    # One row per category
    for label, metrics in sorted(results["per_category"].items()):
        lines.append(
            f"  {label:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6}"
        )

    # Overall row (separated by a line)
    lines.append(f"  {'-' * 62}")
    overall = results["overall"]
    lines.append(
        f"  {'OVERALL':<12} {overall['precision']:>10.4f} {overall['recall']:>10.4f} "
        f"{overall['f1']:>10.4f} {overall['tp']:>6} {overall['fp']:>6} {overall['fn']:>6}"
    )

    return "\n".join(lines)


def format_tiered_report(tiered_results: Dict, pipeline_name: str = "") -> str:
    """
    Combine per-tier tables into one complete evaluation report.
    """
    lines = []
    lines.append(f"\n{'#' * 70}")
    lines.append(f"  EVALUATION REPORT: {pipeline_name}")
    lines.append(f"{'#' * 70}")

    for tier_name, results in tiered_results.items():
        lines.append(format_results_table(results, title=tier_name))

    return "\n".join(lines)


def save_results_json(results: Dict, filepath: str) -> None:
    """Save evaluation results to a JSON file (creating directories as needed)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
#  6. ERROR ANALYSIS & LOGGING
# ─────────────────────────────────────────────

def _build_context_snippet(
    text: str,
    start: int,
    end: int,
    entity_text: str,
    style: str,
    context_chars: int = 40,
) -> str:
    """
    Build a short context window around an entity for human-readable error logs.
    Example output:  'geehrter Herr  >>[Müller]<<  aus Zürich  (Style: High)'
    """
    before = text[max(0, start - context_chars): start].replace("\n", " ")
    after  = text[end: min(len(text), end + context_chars)].replace("\n", " ")
    return f"{before} >>[{entity_text}]<< {after}  (Style: {style})"


def reconstruct_tagged_text(text: str, entities: List[Dict]) -> str:
    """
    Re-insert XML-style tags into plain text for visual debugging.
    Example:  "Herr <PER>Müller</PER> aus <LOC>Zürich</LOC>"

    Entities are inserted back-to-front so that earlier offsets remain valid.
    """
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    tagged = text

    for ent in sorted_entities:
        start = ent["start"]
        end   = ent["end"]
        label = ent["label"]
        tagged = tagged[:start] + f"<{label}>" + tagged[start:end] + f"</{label}>" + tagged[end:]

    return tagged


def generate_error_samples(
    gold_records: List[Dict],
    pred_records: List[Dict],
    num_samples: int = 15,
) -> str:
    """
    Extract concrete text examples of false positives and false negatives
    by comparing strict entity boundaries.  Returns a formatted string
    ready to be included in the evaluation report.

    Each example shows ~40 characters of context around the entity,
    with the entity itself highlighted in >> [text] << markers.
    """
    report_lines = []
    report_lines.append("\n" + "#" * 70)
    report_lines.append("  DEEP DIVE: SAMPLE ERROR ANALYSIS (STRICT MATCHING)")
    report_lines.append("#" * 70 + "\n")

    false_positives_collected = []
    false_negatives_collected = []

    for gold, pred in zip(gold_records, pred_records):
        text  = gold.get("text", "")
        style = gold.get("meta_temp", "Unknown")

        # Build comparable sets: (start, end, label) → surface text
        gold_span_map = {
            (e["start"], e["end"], e["label"]): e["text"]
            for e in gold.get("entities", [])
        }
        pred_span_map = {
            (e["start"], e["end"], e["label"]): e["text"]
            for e in pred.get("entities", [])
        }

        # A prediction not in gold = false positive
        for span, surface_text in pred_span_map.items():
            if span not in gold_span_map:
                start, end, label = span
                context = _build_context_snippet(text, start, end, surface_text, style)
                false_positives_collected.append((label, context))

        # A gold entity not in predictions = false negative
        for span, surface_text in gold_span_map.items():
            if span not in pred_span_map:
                start, end, label = span
                context = _build_context_snippet(text, start, end, surface_text, style)
                false_negatives_collected.append((label, context))

    # Format the collected errors (capped at num_samples)
    report_lines.append("=== FALSE POSITIVES (Model incorrectly highlighted these, or boundaries are wrong) ===")
    if not false_positives_collected:
        report_lines.append("  None found!")
    for label, context in false_positives_collected[:num_samples]:
        report_lines.append(f"  [{label:<6}] ...{context}...")

    report_lines.append("\n=== FALSE NEGATIVES (Model entirely missed these, or boundaries are wrong) ===")
    if not false_negatives_collected:
        report_lines.append("  None found!")
    for label, context in false_negatives_collected[:num_samples]:
        report_lines.append(f"  [{label:<6}] ...{context}...")

    return "\n".join(report_lines)


def generate_full_document_log(gold_records: List[Dict], pred_records: List[Dict]) -> str:
    """
    Create a document-by-document breakdown showing exactly what the model
    predicted vs. the gold standard.  Includes reconstructed XML text for
    side-by-side visual comparison.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("  FULL DOCUMENT-BY-DOCUMENT COMPARISON (STRICT MATCHING)")
    lines.append("=" * 80 + "\n")

    for gold, pred in zip(gold_records, pred_records):
        doc_id = gold.get("id", "Unknown")
        style  = gold.get("meta_temp", "Unknown")
        text   = gold.get("text", "").replace("\n", " ")

        # Gold: use pre-existing raw_content if available, otherwise reconstruct
        gold_tagged = gold.get("raw_content")
        if not gold_tagged:
            gold_tagged = reconstruct_tagged_text(text, gold.get("entities", []))
        gold_tagged = gold_tagged.replace("\n", " ")

        # Pred: always reconstruct from predictions
        pred_tagged = reconstruct_tagged_text(text, pred.get("entities", [])).replace("\n", " ")

        lines.append(f"--- Document ID: {doc_id} | Style: {style} ---")
        lines.append(f"  [GOLD STANDARD] : {gold_tagged}")
        lines.append(f"  [MODEL OUTPUT]  : {pred_tagged}\n")

        # Build comparable span sets
        gold_spans = {
            (e["start"], e["end"], e["label"]): e["text"]
            for e in gold.get("entities", [])
        }
        pred_spans = {
            (e["start"], e["end"], e["label"]): e["text"]
            for e in pred.get("entities", [])
        }

        exact_matches   = set(gold_spans.keys()) & set(pred_spans.keys())
        false_negatives = set(gold_spans.keys()) - set(pred_spans.keys())
        false_positives = set(pred_spans.keys()) - set(gold_spans.keys())

        # Exact matches
        lines.append("  [✓] EXACT MATCHES (Model got these perfectly):")
        if not exact_matches:
            lines.append("      (None)")
        for span in sorted(exact_matches, key=lambda s: s[0]):
            lines.append(f"      [{span[2]:<6}] {gold_spans[span]} (chars {span[0]}:{span[1]})")

        # False negatives
        lines.append("  [✗] FALSE NEGATIVES (Gold standard wanted these, model missed them):")
        if not false_negatives:
            lines.append("      (None)")
        for span in sorted(false_negatives, key=lambda s: s[0]):
            lines.append(f"      [{span[2]:<6}] {gold_spans[span]} (chars {span[0]}:{span[1]})")

        # False positives
        lines.append("  [!] FALSE POSITIVES (Model hallucinated these, or boundaries are wrong):")
        if not false_positives:
            lines.append("      (None)")
        for span in sorted(false_positives, key=lambda s: s[0]):
            lines.append(f"      [{span[2]:<6}] {pred_spans[span]} (chars {span[0]}:{span[1]})")

        lines.append("\n" + "-" * 80 + "\n")

    return "\n".join(lines)


def generate_category_error_report(gold_records: List[Dict], pred_records: List[Dict]) -> str:
    """
    Aggregate false positives and false negatives by category (PER, ORG, …)
    and rank by frequency.  This reveals *systematic* model errors — e.g.
    if the model consistently misses a particular organisation name.
    """
    # Structure: { label: { "FP": { surface_text: count }, "FN": { … } } }
    category_errors = defaultdict(lambda: {"FP": defaultdict(int), "FN": defaultdict(int)})

    for gold, pred in zip(gold_records, pred_records):
        gold_spans = {
            (e["start"], e["end"], e["label"]): e["text"]
            for e in gold.get("entities", [])
        }
        pred_spans = {
            (e["start"], e["end"], e["label"]): e["text"]
            for e in pred.get("entities", [])
        }

        missed_spans      = set(gold_spans.keys()) - set(pred_spans.keys())
        hallucinated_spans = set(pred_spans.keys()) - set(gold_spans.keys())

        for span in missed_spans:
            category_errors[span[2]]["FN"][gold_spans[span]] += 1

        for span in hallucinated_spans:
            category_errors[span[2]]["FP"][pred_spans[span]] += 1

    # Format the report
    lines = []
    lines.append("=" * 80)
    lines.append("  SYSTEMATIC ERROR ANALYSIS BY CATEGORY (FREQUENCY RANKED)")
    lines.append("=" * 80)
    lines.append("  This report shows the most frequent strict-match errors.")
    lines.append("  Compare the Top FNs vs Top FPs in a category to spot boundary issues.\n")

    for label in sorted(category_errors.keys()):
        lines.append(f"\n{'#' * 30} CATEGORY: {label} {'#' * 30}")

        # Top 25 false negatives
        lines.append("\n  [✗] TOP 25 FALSE NEGATIVES (Gold standard wanted these, model missed them):")
        top_fns = sorted(category_errors[label]["FN"].items(), key=lambda x: x[1], reverse=True)[:25]
        if not top_fns:
            lines.append("      (None)")
        for surface_text, count in top_fns:
            lines.append(f"      {count:4d}x : '{surface_text}'")

        # Top 25 false positives
        lines.append("\n  [!] TOP 25 FALSE POSITIVES (Model hallucinated these, or bounds are wrong):")
        top_fps = sorted(category_errors[label]["FP"].items(), key=lambda x: x[1], reverse=True)[:25]
        if not top_fps:
            lines.append("      (None)")
        for surface_text, count in top_fps:
            lines.append(f"      {count:4d}x : '{surface_text}'")

        lines.append("\n" + "-" * 80)

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  7. QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal smoke test: one document, one entity, perfect prediction → F1 = 1.0
    gold = [{"id": 1, "text": "Test", "entities": [
        {"start": 0, "end": 4, "label": "PER", "text": "Test"}
    ]}]
    pred = [{"id": 1, "entities": [
        {"start": 0, "end": 4, "label": "PER", "text": "Test"}
    ]}]

    result = evaluate_predictions(gold, pred)
    assert result["overall"]["f1"] == 1.0, "Self-test failed!"
    print("evaluation_utils.py self-test passed.")
