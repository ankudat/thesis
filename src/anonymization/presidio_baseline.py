"""
presidio_baseline.py
=====================
Microsoft Presidio Analyzer baseline for PII detection in German financial texts.

Presidio combines NER models with regex-based pattern recognizers and context-aware
confidence boosting. This makes it an interesting additional baseline because:
  - It's a widely-used, production-grade PII detection framework
  - It combines NER + regex in a single unified pipeline (vs. your separate approaches)
  - It has built-in recognizers for EMAIL, PHONE, IBAN, etc.

For German, Presidio uses spaCy's German model for NER (PER, LOC, ORG) and
pattern-based recognizers for structured entities. We add custom recognizers
for Swiss-specific patterns (IBAN, MONEY, PHONE) and Tier 3 quasi-identifiers.

Usage:
    python presidio_baseline.py

Requirements:
    pip install presidio-analyzer==2.2.361 presidio-anonymizer==2.2.361 spacy==3.8.0 tqdm
    python -m spacy download de_core_news_lg  # installs de_core_news_lg-3.8.0

"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

INPUT_PATH  = r"C:\thesis\data\label_studio\20260302_Export_Label_Studio_Client_Notes.json"
OUTPUT_DIR  = r"C:\thesis\results\presidio_baseline"
SPLIT_IDS   = r"C:\thesis\results\bert_finetuned\split_ids.json"
LIMIT       = None   # Set to 5 for quick test, None for full run
SEED        = 42

# spaCy model for German NER (used as Presidio's NLP engine)
SPACY_MODEL = "de_core_news_lg"

# Minimum confidence score for Presidio detections
SCORE_THRESHOLD = 0.3


# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import re
import time
import random
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from tqdm import tqdm

from evaluation_utils import (
    ALL_LABELS,
    load_label_studio_export,
    evaluate_tiered,
    format_tiered_report,
    save_results_json,
    generate_error_samples,
    generate_full_document_log,
    generate_category_error_report,
)


# =====================================================================
#  1. PRESIDIO LABEL MAPPING
# =====================================================================
# Presidio uses its own entity type names. We map them to the thesis schema.

PRESIDIO_TO_THESIS = {
    # NER-based (from spaCy German model via Presidio)
    "PERSON":           "PER",
    "LOCATION":         "LOC",
    "ORGANIZATION":     "ORG",       # Presidio may not always emit this
    "NRP":              "NATION",    # Nationality, Religious, Political group

    # Pattern-based (Presidio built-in + custom)
    "EMAIL_ADDRESS":    "EMAIL",
    "PHONE_NUMBER":     "PHONE",
    "IBAN_CODE":        "IBAN",
    "DATE_TIME":        "DATE",

    # Custom recognizers we add below
    "CH_IBAN":          "IBAN",
    "CH_PHONE":         "PHONE",
    "MONEY":            "MONEY",
    "JOB_TITLE":        "JOB",
    "AGE":              "AGE",
    "NATIONALITY":      "NATION",
    "EDUCATION":        "EDU",
}

# Presidio entities we want to request (built-in + custom)
ENTITIES_TO_DETECT = list(set(PRESIDIO_TO_THESIS.keys()))


# =====================================================================
#  2. CUSTOM RECOGNIZERS FOR SWISS GERMAN FINANCIAL TEXT
# =====================================================================

def create_custom_recognizers():
    """
    Create custom Presidio recognizers for Swiss-specific PII patterns
    and Tier 3 quasi-identifiers that Presidio doesn't detect out of the box.
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers = []

    # ── Swiss IBAN ──
    # Format: CH followed by 2 check digits + up to 21 alphanumeric chars
    # Often written with spaces: CH12 0034 5678 9101 1112 3
    ch_iban_pattern = Pattern(
        name="ch_iban",
        regex=r"\bCH\d{2}[\s]?(?:\d{4}[\s]?){4,5}\d{1,4}\b",
        score=0.9,
    )
    recognizers.append(PatternRecognizer(
        supported_entity="CH_IBAN",
        patterns=[ch_iban_pattern],
        supported_language="de",
        context=["IBAN", "Konto", "Kontonummer", "Bankkonto", "Bankverbindung"],
    ))

    # ── Swiss Phone Numbers ──
    # +41 xx xxx xx xx or 0xx xxx xx xx (with various spacing)
    ch_phone_patterns = [
        Pattern(
            name="ch_phone_intl",
            regex=r"\+41[\s]?\d{2}[\s]?\d{3}[\s]?\d{2}[\s]?\d{2}",
            score=0.85,
        ),
        Pattern(
            name="ch_phone_local",
            regex=r"\b0\d{2}[\s]?\d{3}[\s]?\d{2}[\s]?\d{2}\b",
            score=0.7,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="CH_PHONE",
        patterns=ch_phone_patterns,
        supported_language="de",
        context=["Telefon", "Tel", "Natel", "Handy", "Mobilnummer", "Telefonnummer",
                 "telefonisch", "anrufen", "erreichbar"],
    ))

    # ── Monetary Amounts ──
    # CHF/EUR/USD followed by amounts, or amounts followed by currency
    # Handles: CHF 500'000, EUR 2.8 Mio., USD 850'000, CHF 12'450.50
    money_patterns = [
        Pattern(
            name="money_currency_amount",
            regex=r"\b(?:CHF|EUR|USD|GBP)\s*\d[\d''.,\s]*(?:Mio\.?|Mrd\.?|k)?\b",
            score=0.85,
        ),
        Pattern(
            name="money_amount_currency",
            regex=r"\b\d[\d''.,\s]*\s*(?:CHF|EUR|USD|GBP|Franken|Euro)\b",
            score=0.8,
        ),
        Pattern(
            name="money_standalone_currency",
            regex=r"\b(?:CHF|EUR|USD|GBP)\b",
            score=0.3,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="MONEY",
        patterns=money_patterns,
        supported_language="de",
        context=["Betrag", "Summe", "Volumen", "Umsatz", "Gewinn", "EBITDA",
                 "Investition", "Finanzierung", "Hypothek", "Kredit", "überweisen",
                 "zahlen", "Zahlung", "kosten", "Preis", "Wert"],
    ))

    # ── Date Patterns ──
    # German dates: 01.05.2024, 1. Mai 2024, Mai 2024, Ende Monat, nächste Woche
    date_patterns = [
        Pattern(
            name="date_numeric",
            regex=r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
            score=0.9,
        ),
        Pattern(
            name="date_german_month",
            regex=r"\b\d{1,2}\.?\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*\d{2,4}\b",
            score=0.85,
        ),
        Pattern(
            name="date_month_year",
            regex=r"\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b",
            score=0.7,
        ),
        Pattern(
            name="date_relative",
            regex=r"\b(?:Ende\s+(?:Monat|Woche|Jahr)|nächste[rns]?\s+(?:Woche|Monat|Jahr)|letzten?\s+(?:Woche|Monat|Jahr))\b",
            score=0.6,
        ),
        Pattern(
            name="date_duration",
            regex=r"\b\d+\s*(?:Monate[n]?|Jahre[n]?|Wochen|Tage[n]?)\b",
            score=0.5,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="DATE_TIME",
        patterns=date_patterns,
        supported_language="de",
        context=["Datum", "am", "bis", "vom", "seit", "ab", "per", "Termin",
                 "Frist", "Laufzeit", "Auszahlung"],
    ))

    # ── Job Titles (Tier 3) ──
    job_patterns = [
        Pattern(
            name="job_title_de",
            regex=(
                r"\b(?:CEO|CFO|COO|CTO|CIO|CMO|"
                r"Geschäftsführer(?:in)?|"
                r"Geschäftsleitungsmitglied|"
                r"Leiter(?:in)?\s+\w+|"
                r"Direktor(?:in)?|"
                r"Verwaltungsrat(?:spräsident(?:in)?)?|"
                r"Inhaber(?:in)?|"
                r"Buchhalter(?:in)?|"
                r"Projektleiter(?:in)?|"
                r"Analyst(?:in)?|"
                r"Berater(?:in)?|"
                r"Architekt(?:in)?|"
                r"Ingenieur(?:in)?|"
                r"Spezialist(?:in)?|"
                r"Sachbearbeiter(?:in)?|"
                r"Assistentin|Assistent|"
                r"Treuhänder(?:in)?|"
                r"Export\s*Manager(?:in)?|"
                r"(?:FX|TEF|Treasury)[- ]?Spezialist(?:in)?)\b"
            ),
            score=0.7,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="JOB_TITLE",
        patterns=job_patterns,
        supported_language="de",
        context=["als", "Position", "Funktion", "Rolle", "tätig", "arbeitet",
                 "der", "die", "designierte", "zuständig"],
    ))

    # ── Age (Tier 3) ──
    age_patterns = [
        Pattern(
            name="age_years",
            regex=r"\b\d{1,3}\s*(?:Jahre?\s*(?:alt)?|jährig(?:e[rns]?)?|-jährig(?:e[rns]?)?)\b",
            score=0.8,
        ),
        Pattern(
            name="age_standalone",
            regex=r"\b(?:Alter|Alter:)\s*\d{1,3}\b",
            score=0.8,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="AGE",
        patterns=age_patterns,
        supported_language="de",
        context=["alt", "Alter", "geboren", "Jahrgang", "jährig"],
    ))

    # ── Nationality (Tier 3) ──
    nationality_patterns = [
        Pattern(
            name="nationality_de",
            regex=(
                r"\b(?:deutsch(?:e[rns]?)?|"
                r"französisch(?:e[rns]?)?|"
                r"italienisch(?:e[rns]?)?|"
                r"österreichisch(?:e[rns]?)?|"
                r"schweizerisch(?:e[rns]?)?|"
                r"Schweizer(?:in)?|"
                r"britisch(?:e[rns]?)?|"
                r"amerikanisch(?:e[rns]?)?|"
                r"chinesisch(?:e[rns]?)?|"
                r"japanisch(?:e[rns]?)?|"
                r"türkisch(?:e[rns]?)?|"
                r"spanisch(?:e[rns]?)?|"
                r"portugiesisch(?:e[rns]?)?|"
                r"niederländisch(?:e[rns]?)?|"
                r"belgisch(?:e[rns]?)?|"
                r"polnisch(?:e[rns]?)?|"
                r"russisch(?:e[rns]?)?|"
                r"koreanisch(?:e[rns]?)?|"
                r"indisch(?:e[rns]?)?|"
                r"brasilianisch(?:e[rns]?)?|"
                r"chilenisch(?:e[rns]?)?|"
                r"europäisch(?:e[rns]?)?)\b"
            ),
            score=0.6,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="NATIONALITY",
        patterns=nationality_patterns,
        supported_language="de",
        context=["Staatsbürger", "Staatsangehörigkeit", "Nationalität", "Bürger",
                 "Pass", "Herkunft"],
    ))

    # ── Education (Tier 3) ──
    edu_patterns = [
        Pattern(
            name="education_institution",
            regex=(
                r"\b(?:ETH(?:\s*(?:Zürich|Zuerich))?|"
                r"HSG|EPFL|"
                r"Universität\s+\w+|"
                r"Hochschule\s+\w+|"
                r"Fachhochschule\s+\w+)\b"
            ),
            score=0.7,
        ),
        Pattern(
            name="education_degree",
            regex=(
                r"\b(?:MBA|PhD|Dr\.|"
                r"ETH-Absolvent(?:in)?|"
                r"HSG-Absolvent(?:in)?|"
                r"EPFL-Absolvent(?:in)?|"
                r"Hochschulabschluss|"
                r"Hochschulabsolvent(?:in)?|"
                r"(?:Holzbau|Maschinen|Elektro|Bau)-?Ingenieur(?:in)?\s+(?:FH|ETH|HF))\b"
            ),
            score=0.7,
        ),
    ]
    recognizers.append(PatternRecognizer(
        supported_entity="EDUCATION",
        patterns=edu_patterns,
        supported_language="de",
        context=["studiert", "Abschluss", "Absolvent", "Studium", "gelernt",
                 "ausgebildet", "Ausbildung"],
    ))

    return recognizers


# =====================================================================
#  3. ANALYZER SETUP
# =====================================================================

def create_analyzer():
    """
    Create a Presidio AnalyzerEngine configured for German with
    spaCy NLP engine and custom recognizers.
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Configure NLP engine with German spaCy model
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "de", "model_name": SPACY_MODEL},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Create registry with German as the supported language
    registry = RecognizerRegistry(supported_languages=["de"])

    # Load predefined recognizers — these will default to English,
    # so we need to re-register the useful ones for German
    # Do NOT call registry.load_predefined_recognizers() as it adds "en" recognizers

    # Add spaCy-based NER recognizer for German
    from presidio_analyzer.predefined_recognizers import SpacyRecognizer
    spacy_recognizer = SpacyRecognizer(supported_language="de")
    registry.add_recognizer(spacy_recognizer)

    # Add pattern-based recognizers re-instantiated for German
    from presidio_analyzer import Pattern, PatternRecognizer

    # Email recognizer for German
    email_pattern = Pattern(
        name="email",
        regex=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        score=0.9,
    )
    registry.add_recognizer(PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[email_pattern],
        supported_language="de",
        context=["E-Mail", "Mail", "Email", "Mailadresse", "per Mail"],
    ))

    # IBAN recognizer for German (general + Swiss)
    iban_pattern = Pattern(
        name="iban_general",
        regex=r"\b[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){2,7}[\dA-Z]{1,4}\b",
        score=0.85,
    )
    registry.add_recognizer(PatternRecognizer(
        supported_entity="IBAN_CODE",
        patterns=[iban_pattern],
        supported_language="de",
        context=["IBAN", "Konto", "Kontonummer", "Bankverbindung"],
    ))

    # URL recognizer for German
    url_pattern = Pattern(
        name="url",
        regex=r"\bhttps?://[^\s]+\b",
        score=0.6,
    )
    registry.add_recognizer(PatternRecognizer(
        supported_entity="URL",
        patterns=[url_pattern],
        supported_language="de",
    ))

    # Phone number recognizer for German/Swiss
    phone_patterns = [
        Pattern(name="phone_intl", regex=r"\+\d{1,3}[\s\-]?\d[\d\s\-]{6,14}\d", score=0.7),
        Pattern(name="phone_local", regex=r"\b0\d{2}[\s]?\d{3}[\s]?\d{2}[\s]?\d{2}\b", score=0.6),
    ]
    registry.add_recognizer(PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=phone_patterns,
        supported_language="de",
        context=["Telefon", "Tel", "Natel", "Handy", "Mobilnummer",
                 "Telefonnummer", "telefonisch", "anrufen", "erreichbar", "Nummer"],
    ))

    # Add custom recognizers for Swiss/German patterns and Tier 3
    custom_recognizers = create_custom_recognizers()
    for recognizer in custom_recognizers:
        registry.add_recognizer(recognizer)

    # Verify no English-only recognizers leaked in
    print(f"  Registry supported languages: {registry.supported_languages}")

    # Create analyzer
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["de"],
    )

    # Print supported entities
    supported = analyzer.get_supported_entities(language="de")
    print(f"  Supported entities for German: {sorted(supported)}")

    return analyzer


# =====================================================================
#  4. PREDICTION PIPELINE
# =====================================================================

def presidio_predict(
    analyzer,
    text: str,
    score_threshold: float = 0.3,
) -> List[Dict]:
    """
    Run Presidio Analyzer on a single text and return entities
    mapped to the thesis schema.

    Returns:
        List of {start, end, label, text, score, presidio_type}
    """
    results = analyzer.analyze(
        text=text,
        language="de",
        score_threshold=score_threshold,
    )

    entities = []
    seen_spans = set()

    for result in results:
        # Map Presidio entity type to thesis label
        thesis_label = PRESIDIO_TO_THESIS.get(result.entity_type)
        if thesis_label is None:
            continue  # Skip entity types we don't evaluate
        if thesis_label not in ALL_LABELS:
            continue

        start = result.start
        end = result.end
        span_key = (start, end, thesis_label)

        # Deduplicate: Presidio can return overlapping results from
        # different recognizers for the same span
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        entities.append({
            "start": start,
            "end": end,
            "label": thesis_label,
            "text": text[start:end],
            "score": round(result.score, 4),
            "presidio_type": result.entity_type,
        })

    # Sort by start offset
    entities.sort(key=lambda e: e["start"])

    # Resolve overlapping spans: keep higher-scoring entity
    entities = _resolve_overlaps(entities)

    return entities


def _resolve_overlaps(entities: List[Dict]) -> List[Dict]:
    """
    Resolve overlapping entity spans by keeping the higher-confidence one.
    When spans partially overlap, keep both if they don't share >50% of chars.
    """
    if not entities:
        return entities

    # Sort by start, then by length descending
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    resolved = []

    for ent in sorted_ents:
        overlaps = False
        for existing in resolved:
            # Check overlap
            overlap_start = max(ent["start"], existing["start"])
            overlap_end = min(ent["end"], existing["end"])
            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start
                shorter = min(ent["end"] - ent["start"], existing["end"] - existing["start"])
                if shorter > 0 and overlap_len / shorter > 0.5:
                    # Significant overlap — keep the one with higher score
                    if ent.get("score", 0) > existing.get("score", 0):
                        resolved.remove(existing)
                        resolved.append(ent)
                    overlaps = True
                    break

        if not overlaps:
            resolved.append(ent)

    resolved.sort(key=lambda e: e["start"])
    return resolved


# =====================================================================
#  5. EVALUATION & REPORTING
# =====================================================================

def run_presidio_baseline(
    analyzer,
    gold_records: List[Dict],
    output_dir: str,
    score_threshold: float = 0.3,
) -> Dict:
    """
    Run Presidio on all records and evaluate against ground truth.
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_records = []
    total_time = 0.0

    for rec in tqdm(gold_records, desc="Presidio Inference"):
        start_time = time.time()
        entities = presidio_predict(analyzer, rec["text"], score_threshold)
        elapsed = time.time() - start_time
        total_time += elapsed

        pred_records.append({
            "id": rec["id"],
            "entities": entities,
        })

    docs_per_sec = len(gold_records) / max(total_time, 0.01)

    # ── Build report ──
    report_content = []
    report_content.append(f"Presidio Analyzer Evaluation Report")
    report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"NLP Model: {SPACY_MODEL}")
    report_content.append(f"Score Threshold: {score_threshold}")
    report_content.append(f"Total Records: {len(gold_records)}")
    report_content.append(f"Inference Time: {total_time:.1f}s ({docs_per_sec:.0f} docs/sec)")
    report_content.append("")

    all_results = {}

    # Group by temperature/complexity
    temp_groups = defaultdict(lambda: {"gold": [], "pred": []})
    temp_groups["Overall"]["gold"] = gold_records
    temp_groups["Overall"]["pred"] = pred_records

    pred_by_id = {r["id"]: r for r in pred_records}
    for g in gold_records:
        temp = g.get("meta_temp", "Unknown")
        temp_groups[temp]["gold"].append(g)
        temp_groups[temp]["pred"].append(pred_by_id[g["id"]])

    for temp_label in ["Overall", "Low", "Medium", "High"]:
        if temp_label not in temp_groups or not temp_groups[temp_label]["gold"]:
            continue

        group_gold = temp_groups[temp_label]["gold"]
        group_pred = temp_groups[temp_label]["pred"]

        all_results[temp_label] = {}
        report_content.append(f"\n{'=' * 70}")
        report_content.append(f"  EVALUATION SUBSET: {temp_label.upper()} (Records: {len(group_gold)})")
        report_content.append(f"{'=' * 70}\n")

        for matching_mode in ("strict", "relaxed"):
            tiered_results = evaluate_tiered(group_gold, group_pred, matching=matching_mode)
            all_results[temp_label][matching_mode] = tiered_results

            pipeline_name = f"Presidio - {temp_label} ({matching_mode.upper()} matching)"

            if temp_label == "Overall":
                print(format_tiered_report(tiered_results, pipeline_name))

            report_content.append(format_tiered_report(tiered_results, pipeline_name))
            report_content.append("\n")

    # ── Save outputs ──
    prefix = "presidio"

    # Report
    report_path = os.path.join(output_dir, f"{prefix}_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    # Append error analysis
    deep_dive = generate_error_samples(gold_records, pred_records, num_samples=15)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + deep_dive)
    print(f"\n  Report: {report_path}")

    # Full document log
    full_log = generate_full_document_log(gold_records, pred_records)
    log_path = os.path.join(output_dir, f"{prefix}_full_document_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_log)
    print(f"  Document log: {log_path}")

    # Category error analysis
    cat_errors = generate_category_error_report(gold_records, pred_records)
    cat_path = os.path.join(output_dir, f"{prefix}_category_error_analysis.txt")
    with open(cat_path, "w", encoding="utf-8") as f:
        f.write(cat_errors)
    print(f"  Category errors: {cat_path}")

    # Predictions JSON (for semantic_preservation.py)
    pred_path = os.path.join(output_dir, f"{prefix}_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2, ensure_ascii=False)
    print(f"  Predictions: {pred_path}")

    # Results JSON
    results_path = os.path.join(output_dir, f"{prefix}_evaluation_results.json")
    save_results_json(all_results, results_path)
    print(f"  Results JSON: {results_path}")

    return {
        "name": "presidio",
        "elapsed": total_time,
        "results": all_results,
        "pred_records": pred_records,
    }


# =====================================================================
#  6. MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

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

    if LIMIT:
        gold_records = gold_records[:LIMIT]
        print(f"  Limited to {len(gold_records)} records")

    # ── Setup Presidio ──
    print(f"\n{'=' * 60}")
    print(f"  Setting up Presidio Analyzer")
    print(f"  NLP Model: {SPACY_MODEL}")
    print(f"{'=' * 60}")

    analyzer = create_analyzer()

    # ── Run evaluation ──
    result = run_presidio_baseline(
        analyzer=analyzer,
        gold_records=gold_records,
        output_dir=OUTPUT_DIR,
        score_threshold=SCORE_THRESHOLD,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
