"""
classical_baseline.py
======================
Classical Anonymization Baselines for the thesis.

Implements TWO classical baselines, each combined with regex for structured entities:

  Baseline A: spaCy (de_core_news_lg) + Regex
  Baseline B: BERT German NER (token classification) + Regex

Both baselines are limited to Tier 1 + Tier 2 entities.
Neither can detect Tier 3 quasi-identifiers (JOB, AGE, NATION, EDU).

Recommended BERT NER models for German (choose via --bert-model):
  - "fhswf/bert_de_ner"                                German BERT, GermEval + CoNLL
  - "mschiesser/ner-bert-german"                        German BERT NER
  - "Davlan/bert-base-multilingual-cased-ner-hrl"       Multilingual BERT NER
  - "dslim/bert-large-NER"                              Large multilingual NER

Usage:
    python classical_baseline.py --input <label_studio_export.json> --output-dir <results_dir>
    python classical_baseline.py --input <export.json> --bert-model fhswf/bert_de_ner --limit 100

Requirements:
    pip install transformers torch spacy tqdm
    python -m spacy download de_core_news_lg

Author: André Kuhn – Master Thesis (MScIDS, HSLU)
"""

import json
import re
import os
import argparse
import time
import gc
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm

import torch

from evaluation_utils import (
    load_label_studio_export,
    evaluate_tiered,
    format_tiered_report,
    save_results_json,
    generate_error_samples,
    generate_full_document_log,
    generate_category_error_report,
    SPACY_LABEL_MAP,
)


# ─────────────────────────────────────────────
#  1. CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_SPACY_MODEL = "de_core_news_lg"
DEFAULT_BERT_MODEL  = "fhswf/bert_de_ner"

# Mapping from BERT NER label schemes to our unified schema.
# Most German BERT NER models use IOB2 tagging (B-PER, I-PER, B-LOC, …).
# We strip the B-/I- prefix in `map_bert_label()` and then look up the
# base label here.  `None` means "ignore this label".
BERT_LABEL_MAP: Dict[str, Optional[str]] = {
    "PER": "PER",
    "PERSON": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "MISC": None,       # too ambiguous to map reliably
    "OTH": None,        # some models use OTH for miscellaneous
    "O": None,          # the "outside" tag — not an entity
}


# ─────────────────────────────────────────────
#  2. REGEX PATTERNS FOR STRUCTURED ENTITIES
# ─────────────────────────────────────────────
# These patterns target Swiss / German financial documents.
# Each label maps to a list of compiled regex patterns;
# all patterns for a label are tried and the union of matches is used.

REGEX_PATTERNS = {
    # Swiss IBANs:  CH + 2 check digits + 5 groups of 4 digits + 1-2 trailing digits
    "IBAN": [
        re.compile(r"\bCH\s?\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?[\d]{1,2}\b"),
    ],

    # Standard email addresses
    "EMAIL": [
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ],

    # Swiss phone numbers in international (+41 …) and local (0xx …) formats
    "PHONE": [
        re.compile(r"\+41[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
        re.compile(r"\b0\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"),
        re.compile(r"\b0\d{2}\s\d{3}\s\d{2}\s\d{2}\b"),
    ],

    # German-format dates, named months, quarters, and standalone years
    "DATE": [
        re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),                            # 01.03.2025
        re.compile(                                                                # 1. Januar 2025
            r"\b\d{1,2}\.\s?(?:Januar|Februar|März|April|Mai|Juni|Juli|"
            r"August|September|Oktober|November|Dezember)\s?\d{2,4}\b"
        ),
        re.compile(                                                                # Januar 2025
            r"\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|"
            r"September|Oktober|November|Dezember)\s\d{4}\b"
        ),
        re.compile(r"\bQ[1-4]\s?\d{4}\b"),                                        # Q3 2025
        re.compile(r"\b(?:19|20)\d{2}\b"),                                         # standalone years
    ],

    # Currency amounts (CHF, EUR, USD, GBP) with optional multipliers (Mio, Mrd, k)
    "MONEY": [
        # Currency code first: "CHF 1'250.00 Mio."
        re.compile(r"\b(?:CHF|EUR|USD|GBP)\s?[\d'.,]+(?:\s?(?:Mio|Mrd|k|K|Tsd)\.?)?(?!\w)"),
        # Amount first: "1'250.00 CHF"  (leading digit required to avoid ". CHF" false positives)
        re.compile(r"\b\d[\d'.,]*\s?(?:CHF|EUR|USD|GBP)(?:\s?(?:Mio|Mrd|k|K|Tsd)\.?)?(?!\w)"),
        # Bare currency code (e.g. "in CHF")
        re.compile(r"\b(?:CHF|EUR|USD|GBP)\b"),
        # Shorthand amounts like "250k"
        re.compile(r"\b\d+(?:['.,]\d+)?[kK]\b"),
    ],
}


# ─────────────────────────────────────────────
#  3. REGEX PIPELINE (shared by both baselines)
# ─────────────────────────────────────────────

def run_regex_ner(text: str) -> List[Dict]:
    """
    Scan `text` with every regex pattern and return a flat list of
    entity dicts {start, end, label, text}.
    """
    entities = []

    for label, patterns in REGEX_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                entities.append({
                    "start": match.start(),
                    "end":   match.end(),
                    "label": label,
                    "text":  match.group(),
                })

    return entities


# ─────────────────────────────────────────────
#  4. OVERLAP RESOLUTION
# ─────────────────────────────────────────────

def resolve_overlaps(entities: List[Dict]) -> List[Dict]:
    """
    When NER and regex produce overlapping spans for the same text region,
    keep only the longest span.  On ties, the entity that appears first in
    the input list wins — this is why callers put regex entities *before*
    NER entities (giving regex priority on exact ties).

    Algorithm:
      1. Sort entities by start position (earliest first), breaking ties by
         span length (longest first).
      2. Walk through and accept an entity only if it doesn't overlap with
         any already-accepted entity.

    Returns a new list sorted by start position.
    """
    if not entities:
        return []

    sorted_entities = sorted(
        entities,
        key=lambda e: (e["start"], -(e["end"] - e["start"])),  # earliest start, longest span
    )

    accepted: List[Dict] = []

    for candidate in sorted_entities:
        # Check if this candidate overlaps with any already-accepted entity
        has_overlap = any(
            candidate["start"] < existing["end"] and candidate["end"] > existing["start"]
            for existing in accepted
        )
        if not has_overlap:
            accepted.append(candidate)

    return sorted(accepted, key=lambda e: e["start"])


# ─────────────────────────────────────────────
#  5. BASELINE A:  spaCy + Regex
# ─────────────────────────────────────────────

def load_spacy_model(model_name: str):
    """Load and return a spaCy language model."""
    import spacy

    print(f"Loading spaCy model: {model_name}")
    nlp = spacy.load(model_name)
    print(f"  Pipeline components: {nlp.pipe_names}")
    return nlp


def run_spacy_ner(nlp, text: str) -> List[Dict]:
    """
    Run spaCy NER on `text` and translate labels to our schema
    using SPACY_LABEL_MAP.  Unknown or ignored labels are dropped.
    """
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        mapped_label = SPACY_LABEL_MAP.get(ent.label_)
        if mapped_label is not None:
            entities.append({
                "start": ent.start_char,
                "end":   ent.end_char,
                "label": mapped_label,
                "text":  ent.text,
            })

    return entities


def spacy_pipeline(nlp, text: str) -> List[Dict]:
    """
    Full Baseline A pipeline: spaCy NER + regex, with overlap resolution.
    Regex entities are placed first so they win tiebreakers.
    """
    spacy_entities = run_spacy_ner(nlp, text)
    regex_entities = run_regex_ner(text)
    return resolve_overlaps(regex_entities + spacy_entities)


# ─────────────────────────────────────────────
#  6. BASELINE B:  BERT NER + Regex
# ─────────────────────────────────────────────

def load_bert_ner_model(model_name: str, device: int = -1):
    """
    Load a BERT-based NER model via the HuggingFace `pipeline` abstraction.

    Args:
        model_name: HuggingFace model ID (e.g. "fhswf/bert_de_ner")
        device:     GPU index (0, 1, …) or -1 for CPU.
                    If -1 and a GPU is available, automatically uses GPU 0.
    """
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )

    print(f"Loading BERT NER model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForTokenClassification.from_pretrained(model_name, trust_remote_code=True)

    # Auto-select GPU if available and no explicit device was requested
    if device == -1 and torch.cuda.is_available():
        device = 0

    ner_pipe = hf_pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=device,
        # "first" keeps the label of the first sub-token in a word, avoiding
        # fragmented predictions from WordPiece tokenisation
        aggregation_strategy="first",
    )

    # Print diagnostic info about the model
    id2label = model.config.id2label
    print(f"  Model labels: {list(id2label.values())}")
    print(f"  Device: {'GPU' if device >= 0 else 'CPU'}")

    return ner_pipe


def map_bert_label(raw_label: str) -> Optional[str]:
    """
    Translate a raw BERT NER label to our unified schema.

    Steps:
      1. Strip IOB2 prefixes (B-, I-, E-, S-)
      2. Strip GermEval-specific suffixes ("deriv", "part")
      3. Look up the cleaned label in BERT_LABEL_MAP

    Returns the mapped label, or None if it should be ignored.
    """
    base_label = raw_label

    # 1. Remove IOB prefix
    for prefix in ("B-", "I-", "E-", "S-"):
        if base_label.startswith(prefix):
            base_label = base_label[len(prefix):]
            break

    # 2. Remove GermEval suffixes (e.g. "PERderiv" → "PER")
    base_label = base_label.replace("deriv", "").replace("part", "")

    # 3. Look up — try as-is first, then uppercase
    return BERT_LABEL_MAP.get(base_label, BERT_LABEL_MAP.get(base_label.upper()))


def run_bert_ner(ner_pipe, text: str) -> List[Dict]:
    """
    Run the HuggingFace NER pipeline on `text` and map outputs to our schema.
    Returns a list of entity dicts; returns an empty list on errors.
    """
    try:
        raw_results = ner_pipe(text)
    except Exception as error:
        print(f"  BERT NER error: {error}")
        return []

    entities = []

    for item in raw_results:
        raw_label = item.get("entity_group", item.get("entity", ""))
        mapped_label = map_bert_label(raw_label)

        if mapped_label is None:
            continue

        start = item["start"]
        end   = item["end"]

        entities.append({
            "start": start,
            "end":   end,
            "label": mapped_label,
            "text":  text[start:end],
        })

    return entities


def bert_pipeline(ner_pipe, text: str) -> List[Dict]:
    """
    Full Baseline B pipeline: BERT NER + regex, with overlap resolution.
    Regex entities are placed first so they win tiebreakers.
    """
    bert_entities  = run_bert_ner(ner_pipe, text)
    regex_entities = run_regex_ner(text)
    return resolve_overlaps(regex_entities + bert_entities)


# ─────────────────────────────────────────────
#  7. RUN A SINGLE BASELINE
# ─────────────────────────────────────────────

def run_baseline(
    name: str,
    predict_fn,
    gold_records: List[Dict],
    output_dir: str,
) -> Dict:
    """
    Run one baseline end-to-end on all records:
      1. Generate predictions for every document
      2. Group documents by temperature / style metadata
      3. Evaluate each group (strict + relaxed matching, all tiers)
      4. Save predictions, reports, and error analyses to disk

    Returns a summary dict with timing, results, and predictions.
    """
    baseline_dir = os.path.join(output_dir, name)
    os.makedirs(baseline_dir, exist_ok=True)

    # ── Step 1: Generate predictions ──
    print(f"\n  Running predictions ({name})...")
    start_time = time.time()

    pred_records = []
    for record in tqdm(gold_records, desc=f"{name} NER"):
        predicted_entities = predict_fn(record["text"])
        pred_records.append({
            "id":       record["id"],
            "entities": predicted_entities,
        })

    elapsed      = time.time() - start_time
    docs_per_sec = len(gold_records) / max(elapsed, 0.01)
    print(f"  Completed in {elapsed:.1f}s ({docs_per_sec:.0f} docs/sec)")

    # Save raw predictions to JSON
    predictions_path = os.path.join(baseline_dir, f"{name}_predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2, ensure_ascii=False)

    # ── Step 2: Group documents by temperature / style ──
    temp_groups = defaultdict(lambda: {"gold": [], "pred": []})

    # "Overall" always contains all documents
    temp_groups["Overall"]["gold"] = gold_records
    temp_groups["Overall"]["pred"] = pred_records

    # Additionally group by the meta_temp field
    for gold_rec, pred_rec in zip(gold_records, pred_records):
        temperature_label = gold_rec.get("meta_temp", "Unknown")
        temp_groups[temperature_label]["gold"].append(gold_rec)
        temp_groups[temperature_label]["pred"].append(pred_rec)

    # ── Step 3: Evaluate each group ──
    all_results    = {}
    report_content = []

    report_content.append(f"{name.upper()} Baseline Evaluation Report")
    report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Total Records: {len(gold_records)}")
    report_content.append(f"Inference Time: {elapsed:.1f}s ({docs_per_sec:.0f} docs/sec)\n")

    # Process groups in a deterministic order
    preferred_order = ["Overall", "Low", "Medium", "High"]
    for extra_key in temp_groups.keys():
        if extra_key not in preferred_order:
            preferred_order.append(extra_key)

    for temp_label in preferred_order:
        if temp_label not in temp_groups or not temp_groups[temp_label]["gold"]:
            continue

        group_gold = temp_groups[temp_label]["gold"]
        group_pred = temp_groups[temp_label]["pred"]
        num_docs   = len(group_gold)

        all_results[temp_label] = {}
        report_content.append(f"\n{'=' * 70}")
        report_content.append(f"  EVALUATION SUBSET: {temp_label.upper()} (Records: {num_docs})")
        report_content.append(f"{'=' * 70}\n")

        for matching_mode in ("strict", "relaxed"):
            tiered_results = evaluate_tiered(group_gold, group_pred, matching=matching_mode)
            all_results[temp_label][matching_mode] = tiered_results

            # Print the "Overall" group to the terminal (others only go to the report file)
            if temp_label == "Overall":
                print(format_tiered_report(tiered_results, f"{name} ({matching_mode.upper()} matching)"))

            report_content.append(
                format_tiered_report(tiered_results, f"{name} - {temp_label} ({matching_mode.upper()} matching)")
            )
            report_content.append("\n")

    # ── Step 4: Save all outputs ──

    # Structured results as JSON
    results_path = os.path.join(baseline_dir, f"{name}_evaluation_results.json")
    save_results_json(all_results, results_path)

    # Append a sample error deep-dive to the report
    deep_dive_text = generate_error_samples(gold_records, pred_records, num_samples=15)
    report_content.append(deep_dive_text)

    # Save the main human-readable report
    report_path = os.path.join(baseline_dir, f"{name}_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    print(f"  Report: {report_path}")

    # Full document-by-document comparison log
    print(f"  Generating full document-by-document log...")
    full_log_text = generate_full_document_log(gold_records, pred_records)
    full_log_path = os.path.join(baseline_dir, f"{name}_full_document_log.txt")
    with open(full_log_path, "w", encoding="utf-8") as f:
        f.write(full_log_text)

    # Category-level frequency error report
    print(f"  Generating category frequency error report...")
    category_error_text = generate_category_error_report(gold_records, pred_records)
    category_error_path = os.path.join(baseline_dir, f"{name}_category_error_analysis.txt")
    with open(category_error_path, "w", encoding="utf-8") as f:
        f.write(category_error_text)

    return {
        "name":         name,
        "elapsed":      elapsed,
        "results":      all_results,
        "pred_records": pred_records,
    }


# ─────────────────────────────────────────────
#  8. SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────

def print_and_save_comparison(baseline_results: Dict, output_dir: str) -> None:
    """
    When both baselines have been run, produce a side-by-side comparison
    of their strict-matching F1 scores across all tiers and temperature groups.
    """
    comparison_lines = []
    comparison_lines.append(f"\n{'=' * 60}")
    comparison_lines.append(f"  BASELINE COMPARISON (Strict Matching)")
    comparison_lines.append(f"{'=' * 60}")

    tier_names = [
        "Tier 1 – Direct NER",
        "Tier 2 – Structured (Regex)",
        "Tier 3 – Quasi-Identifiers",
        "All Categories",
    ]

    for temp_label in ("Overall", "Low", "Medium", "High"):
        # Skip if either model doesn't have this temperature subset
        spacy_has = temp_label in baseline_results["spacy"]["results"]
        bert_has  = temp_label in baseline_results["bert"]["results"]
        if not (spacy_has and bert_has):
            continue

        comparison_lines.append(f"\n  >>> SUBSET: {temp_label.upper()} <<<")
        comparison_lines.append(f"  {'Metric':<25} {'spaCy + Regex':>15} {'BERT + Regex':>15}")
        comparison_lines.append(f"  {'-' * 57}")

        # F1 per tier
        for tier_name in tier_names:
            spacy_f1 = (baseline_results["spacy"]["results"][temp_label]["strict"]
                        .get(tier_name, {}).get("overall", {}).get("f1", 0))
            bert_f1  = (baseline_results["bert"]["results"][temp_label]["strict"]
                        .get(tier_name, {}).get("overall", {}).get("f1", 0))
            comparison_lines.append(f"  {tier_name:<25} {spacy_f1:>15.4f} {bert_f1:>15.4f}")

        # Per-category breakdown only for the "Overall" subset (to avoid a wall of text)
        if temp_label == "Overall":
            comparison_lines.append(f"\n  {'Category (Overall)':<18} {'spaCy P/R/F1':>18} {'BERT P/R/F1':>18}")
            comparison_lines.append(f"  {'-' * 57}")

            for label in ("PER", "LOC", "ORG"):
                spacy_cat = (baseline_results["spacy"]["results"]["Overall"]["strict"]
                             ["Tier 1 – Direct NER"]["per_category"].get(label, {}))
                bert_cat  = (baseline_results["bert"]["results"]["Overall"]["strict"]
                             ["Tier 1 – Direct NER"]["per_category"].get(label, {}))

                spacy_str = (f"{spacy_cat.get('precision', 0):.2f}/"
                             f"{spacy_cat.get('recall', 0):.2f}/"
                             f"{spacy_cat.get('f1', 0):.2f}")
                bert_str  = (f"{bert_cat.get('precision', 0):.2f}/"
                             f"{bert_cat.get('recall', 0):.2f}/"
                             f"{bert_cat.get('f1', 0):.2f}")

                comparison_lines.append(f"  {label:<18} {spacy_str:>18} {bert_str:>18}")

    comparison_lines.append(f"\n{'=' * 60}")
    comparison_lines.append(f"  Tier 3 (Quasi-ID) F1 = 0.0 for both — as expected.")
    comparison_lines.append(f"  Results saved to: {output_dir}")
    comparison_lines.append(f"{'=' * 60}")

    comparison_text = "\n".join(comparison_lines)

    # Print to terminal
    print(comparison_text)

    # Save to file
    comparison_path = os.path.join(output_dir, "baseline_comparison_summary.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(comparison_text)
    print(f"  Comparison summary saved to: {comparison_path}")


# ─────────────────────────────────────────────
#  9. MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classical NER Baselines: spaCy + BERT (both with Regex)"
    )
    parser.add_argument(
        "--input", type=str,
        default=r"C:\thesis\data\label_studio\20260222_Export_Label_Studio_Client_Notes.json",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=r"C:\thesis\results\classical_baselines",
    )
    parser.add_argument(
        "--spacy-model", type=str, default=DEFAULT_SPACY_MODEL,
        help="spaCy model name (default: de_core_news_lg)",
    )
    parser.add_argument(
        "--bert-model", type=str, default=DEFAULT_BERT_MODEL,
        help="HuggingFace BERT NER model ID (default: fhswf/bert_de_ner)",
    )
    parser.add_argument(
        "--skip-spacy", action="store_true",
        help="Skip the spaCy baseline",
    )
    parser.add_argument(
        "--skip-bert", action="store_true",
        help="Skip the BERT baseline",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of records (for quick testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load annotated data ──
    print(f"Loading data from: {args.input}")
    gold_records = load_label_studio_export(args.input)
    if args.limit:
        gold_records = gold_records[: args.limit]
    print(f"  Loaded {len(gold_records)} records\n")

    baseline_results = {}

    # ── BASELINE A: spaCy + Regex ──
    if not args.skip_spacy:
        print(f"{'=' * 60}")
        print(f"  BASELINE A: spaCy ({args.spacy_model}) + Regex")
        print(f"{'=' * 60}")

        nlp = load_spacy_model(args.spacy_model)
        result_a = run_baseline(
            name="spacy",
            predict_fn=lambda text: spacy_pipeline(nlp, text),
            gold_records=gold_records,
            output_dir=args.output_dir,
        )
        baseline_results["spacy"] = result_a

        # Free memory before loading the next model
        del nlp
        gc.collect()

    # ── BASELINE B: BERT NER + Regex ──
    if not args.skip_bert:
        print(f"\n{'=' * 60}")
        print(f"  BASELINE B: BERT NER ({args.bert_model}) + Regex")
        print(f"{'=' * 60}")

        ner_pipe = load_bert_ner_model(args.bert_model)
        result_b = run_baseline(
            name="bert",
            predict_fn=lambda text: bert_pipeline(ner_pipe, text),
            gold_records=gold_records,
            output_dir=args.output_dir,
        )
        baseline_results["bert"] = result_b

        # Free memory
        del ner_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Side-by-side comparison (only if both baselines were run) ──
    if len(baseline_results) == 2:
        print_and_save_comparison(baseline_results, args.output_dir)


if __name__ == "__main__":
    main()
