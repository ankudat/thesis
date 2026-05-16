"""
app.py
======
Web demo for German PII anonymization based on the thesis pipelines.

Five approaches run side by side on user-pasted text:
  1. REGEX        -- structured PII (IBAN, EMAIL, PHONE, DATE, MONEY)
  2. BERT         -- pretrained German NER (fhswf/bert_de_ner) for PER/LOC/ORG
                     plus the regex stage layered on top
  3. LLM TAG      -- Llama-3-8B-Instruct, zero-shot tag-and-replace
                     (covers all twelve PII categories incl. Tier-3
                      quasi-identifiers: JOB, AGE, NATION, EDU). Output is
                     the original text with PII spans replaced by [LABEL].
  4. LLM REWRITE  -- same Llama-3 model with a different prompt; rewrites
                     the text into a fluent anonymized version (e.g.
                     "Markus Steiner, CFO" -> "ein Mitglied der
                     Geschaeftsleitung"). Best readability, but higher
                     hallucination and PII-leakage risk -- see thesis
                     RQ2a/RQ2b.
  5. CASCADE      -- regex -> BERT -> LLM applied in sequence; each stage
                     contributes only entities that the earlier stages
                     missed. Earlier stages are trusted on overlap, so the
                     fast/deterministic detectors win wherever they fire,
                     and the LLM only adds genuinely new spans (typically
                     Tier-3 quasi-identifiers). This is the recommended
                     production architecture: maximum coverage, lowest
                     average latency.

Run:
    cd demo
    python app.py
    # then open http://127.0.0.1:5000 in a browser

See README.md for full installation instructions (including how to
authenticate with Hugging Face, which is required for the Llama-3
pipelines).

Notes
-----
* Models are lazy-loaded on first request to keep startup fast.
* The LLM stage requires ~16 GB VRAM in fp16 or ~6 GB with 4-bit
  quantization. Set QUANTIZE_LLM_4BIT below to True if your GPU is
  tight on memory.
* If a stage fails (e.g. no GPU, model not downloaded) the other two
  still work -- the failure is reported as JSON in the response.
"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

LLM_MODEL_NAME      = "meta-llama/Meta-Llama-3-8B-Instruct"
BERT_MODEL_NAME     = "fhswf/bert_de_ner"
QUANTIZE_LLM_4BIT   = False     # set True for ~6 GB VRAM mode
HOST                = "127.0.0.1"
PORT                = 5000

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import re
import time
from typing import List, Dict, Optional

from flask import Flask, request, jsonify, render_template

# Lazy globals -- populated on first call to each pipeline
_bert_pipeline = None
_llm_model = None
_llm_tokenizer = None


# =====================================================================
#  1. REGEX PIPELINE  (Tier-2 structured PII)
# =====================================================================

REGEX_PATTERNS = {
    "IBAN": [
        re.compile(r"\bCH\s?\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{1,2}\b"),
    ],
    "EMAIL": [
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ],
    "PHONE": [
        re.compile(r"\+41[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
        re.compile(r"\b0\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"),
    ],
    "DATE": [
        re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),
        re.compile(
            r"\b\d{1,2}\.\s?(?:Januar|Februar|März|April|Mai|Juni|Juli|"
            r"August|September|Oktober|November|Dezember)\s?\d{2,4}\b"
        ),
        re.compile(
            r"\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|"
            r"September|Oktober|November|Dezember)\s\d{4}\b"
        ),
        re.compile(r"\bQ[1-4]\s?\d{4}\b"),
        re.compile(r"\b(?:19|20)\d{2}\b"),
    ],
    "MONEY": [
        re.compile(r"\b(?:CHF|EUR|USD|GBP)\s?[\d'.,]+(?:\s?(?:Mio|Mrd|k|K|Tsd)\.?)?(?!\w)"),
        re.compile(r"\b\d[\d'.,]*\s?(?:CHF|EUR|USD|GBP)(?:\s?(?:Mio|Mrd|k|K|Tsd)\.?)?(?!\w)"),
        re.compile(r"\b\d+(?:['.,]\d+)?[kK]\b"),
    ],
}


def detect_regex(text: str) -> List[Dict]:
    """Run all regex patterns; resolve overlaps so nested matches collapse
    (e.g. '2024' inside '15.05.2024' is dropped in favour of the longer span)."""
    entities = []
    for label, patterns in REGEX_PATTERNS.items():
        for pattern in patterns:
            for m in pattern.finditer(text):
                entities.append({
                    "start": m.start(),
                    "end":   m.end(),
                    "label": label,
                    "text":  m.group(),
                })
    return resolve_overlaps(entities)


# =====================================================================
#  2. BERT PIPELINE  (Tier-1 PER / LOC / ORG  +  regex on top)
# =====================================================================

BERT_LABEL_MAP = {
    "PER":  "PER",  "PERSON":  "PER",
    "LOC":  "LOC",  "LOCATION": "LOC",  "GPE": "LOC",
    "ORG":  "ORG",  "ORGANIZATION": "ORG",
    "MISC": None,   # drop -- not a PII category in our schema
}


def _ensure_bert():
    """Lazy-load the BERT NER pipeline (cached after first call)."""
    global _bert_pipeline
    if _bert_pipeline is not None:
        return
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              pipeline)
    print(f"[BERT] Loading {BERT_MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    mdl = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_NAME)
    _bert_pipeline = pipeline(
        "ner", model=mdl, tokenizer=tok, aggregation_strategy="simple",
        device=-1,  # CPU; change to 0 to force GPU
    )
    print("[BERT] Ready.")


def _detect_bert_only(text: str) -> List[Dict]:
    """Run BERT NER alone, no regex layer. Used by both the standalone
    BERT pipeline (with regex layered on top) and the cascade pipeline
    (where regex has already run as the first stage)."""
    _ensure_bert()
    raw = _bert_pipeline(text)
    bert_ents = []
    for r in raw:
        label_in = r["entity_group"].upper().replace("B-", "").replace("I-", "")
        label = BERT_LABEL_MAP.get(label_in)
        if not label:
            continue
        bert_ents.append({
            "start": int(r["start"]),
            "end":   int(r["end"]),
            "label": label,
            "text":  text[int(r["start"]):int(r["end"])],
        })
    return bert_ents


def detect_bert(text: str) -> List[Dict]:
    """Run BERT NER then layer regex on top, then resolve overlaps."""
    return resolve_overlaps(detect_regex(text) + _detect_bert_only(text))


def resolve_overlaps(entities: List[Dict]) -> List[Dict]:
    """Keep the longest span on overlapping ranges; ties go to first-seen."""
    if not entities:
        return []
    entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    kept = []
    for ent in entities:
        if kept and ent["start"] < kept[-1]["end"]:
            continue   # overlaps the previously kept (longer) span
        kept.append(ent)
    return kept


# =====================================================================
#  3. LLM PIPELINE  (Llama-3, zero-shot tag-and-replace)
# =====================================================================

ENTITY_LABELS = ["PER", "LOC", "ORG", "DATE", "EMAIL", "PHONE",
                 "IBAN", "MONEY", "JOB", "AGE", "NATION", "EDU"]


def _ensure_llm():
    """Lazy-load the LLM (cached after first call)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[LLM] Loading {LLM_MODEL_NAME} (this can take a minute) ...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if _llm_tokenizer.pad_token is None:
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token

    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if QUANTIZE_LLM_4BIT:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        )
    _llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **kw)
    _llm_model.eval()
    print("[LLM] Ready.")


def _llm_system_prompt() -> str:
    return (
        "You are an expert Named Entity Recognition (NER) system specialized "
        "in identifying personally identifiable information (PII) in "
        "German-language financial communications from a Swiss banking "
        "context.\n\n"
        "Task: Given a text, COPY the entire text exactly and mark ALL "
        "entities by wrapping them with @@LABEL and ## tokens.\n\n"
        f"Entity categories: {', '.join(ENTITY_LABELS)}\n\n"
        "RULES:\n"
        "1. Copy the ENTIRE input text character by character.\n"
        "2. Wrap each entity with @@LABEL before and ## after.\n"
        "   Example: 'Herr Markus Steiner, CEO' -> "
        "'Herr @@PER Markus Steiner##, @@JOB CEO##'\n"
        "3. Tag EVERY occurrence including repeats.\n"
        "4. For German compounds (e.g. HSG-Absolventin), tag the FULL compound.\n"
        "5. Nationality adjectives (deutscher, schweizerisch) belong to NATION.\n"
        "6. Person names should NOT include titles (Herr, Frau, Dr.).\n"
        "7. If no entities, just copy the text unchanged.\n"
        "8. Output ONLY the marked text. No explanations.\n"
    )


_TAG_RE = re.compile(r"@@([A-Z]+)\s+(.+?)\s*##")


def _parse_marked_output(marked: str, original: str) -> List[Dict]:
    """Parse @@LABEL ...## output back into entity dicts on the ORIGINAL text."""
    ents = []
    cursor = 0
    for m in _TAG_RE.finditer(marked):
        label = m.group(1)
        if label not in ENTITY_LABELS:
            continue
        ent_text = m.group(2).strip()
        idx = original.find(ent_text, cursor)
        if idx == -1:
            idx = original.find(ent_text)
        if idx == -1:
            continue
        ents.append({
            "start": idx,
            "end":   idx + len(ent_text),
            "label": label,
            "text":  ent_text,
        })
        cursor = idx + len(ent_text)
    return resolve_overlaps(ents)


def _llm_generate(messages, max_new_tokens: int) -> str:
    """Shared chat-style generation helper used by both LLM pipelines."""
    import torch
    prompt = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = _llm_tokenizer(prompt, return_tensors="pt").to(_llm_model.device)
    with torch.no_grad():
        out = _llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=_llm_tokenizer.pad_token_id,
        )
    return _llm_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def detect_llm(text: str) -> List[Dict]:
    """Run Llama-3 zero-shot tag-and-replace and parse the output."""
    _ensure_llm()
    messages = [
        {"role": "system", "content": _llm_system_prompt()},
        {"role": "user",
         "content": f"Mark all PII entities in the following text:\n\n"
                    f"Input: {text}\nOutput:"},
    ]
    generated = _llm_generate(messages,
                              max_new_tokens=min(1024, int(len(text) * 1.5) + 256))
    return _parse_marked_output(generated, text)


# =====================================================================
#  3b. LLM REWRITE PIPELINE  (Llama-3, prompt-based rewriting)
# =====================================================================

def _llm_rewrite_system_prompt() -> str:
    return (
        "You are an expert text anonymization system for German-language "
        "financial communications from a Swiss banking context.\n\n"
        "Your task: Rewrite the given text so that ALL personally identifiable "
        "information (PII) is removed or replaced with generic alternatives, "
        "while preserving the original meaning, structure, and informational "
        "content as closely as possible.\n\n"
        "PII categories to anonymize:\n"
        "- Person names -> generic references (e.g. 'ein Kunde', 'die Geschaeftsfuehrerin')\n"
        "- Organization names -> generic terms (e.g. 'ein Unternehmen', 'die Firma')\n"
        "- Locations/Addresses -> generic references (e.g. 'eine Stadt', 'ein Standort in der Schweiz')\n"
        "- Dates -> relative or vague references (e.g. 'kuerzlich', 'an einem Datum')\n"
        "- Email addresses -> remove or replace with '[E-Mail]'\n"
        "- Phone numbers -> remove or replace with '[Telefon]'\n"
        "- IBAN/Account numbers -> replace with '[Konto]'\n"
        "- Monetary amounts -> generic references (e.g. 'einen Betrag', 'eine Summe')\n"
        "- Job titles -> generalize (e.g. 'eine Fuehrungskraft', 'ein Mitarbeiter')\n"
        "- Age references -> remove or generalize\n"
        "- Nationality -> remove or generalize\n"
        "- Education -> generalize (e.g. 'ein Hochschulabschluss')\n\n"
        "RULES:\n"
        "1. Preserve the MEANING and INTENT of the communication.\n"
        "2. Keep the text in German.\n"
        "3. Maintain the same register and tone.\n"
        "4. The rewritten text should be fluent and natural.\n"
        "5. Do NOT add information that was not in the original.\n"
        "6. Do NOT omit non-PII information.\n"
        "7. Output ONLY the rewritten text. No explanations, no markdown.\n"
    )


def detect_rewrite(text: str) -> Dict:
    """Run Llama-3 prompt-based rewriting. Returns the rewritten text
    (no entity-level output, since the model produces fresh prose)."""
    _ensure_llm()
    messages = [
        {"role": "system", "content": _llm_rewrite_system_prompt()},
        {"role": "user",
         "content": f"Anonymize the following text by rewriting it:\n\n"
                    f"Original: {text}\n\nAnonymized:"},
    ]
    generated = _llm_generate(messages,
                              max_new_tokens=min(1024, int(len(text) * 1.8) + 256))
    return {"rewritten": generated.strip()}


# =====================================================================
#  3c. CASCADE PIPELINE  (regex -> BERT -> LLM, earlier stages win)
# =====================================================================

def _overlaps_any(candidate: Dict, kept: List[Dict]) -> bool:
    """True if `candidate`'s span intersects any already-kept entity."""
    cs, ce = candidate["start"], candidate["end"]
    for k in kept:
        if cs < k["end"] and ce > k["start"]:
            return True
    return False


def detect_cascade(text: str) -> List[Dict]:
    """
    Sequential cascade. Each stage adds only entities the earlier stages
    missed -- earlier stages are trusted on any overlap.

    Order is chosen by speed and trust:
      1. regex   -- ~1 ms, ~100% precision on the patterns it covers
      2. BERT    -- ~40 ms, high precision on PER/LOC/ORG
      3. LLM     -- ~8 s,   only one that detects Tier-3 (JOB/AGE/NATION/EDU)

    Each entity is tagged with a `source` field so the UI can show which
    stage caught it. The result is the union of all unique detections.
    """
    kept: List[Dict] = []

    # Stage 1: regex (always wins on its own patterns)
    for ent in detect_regex(text):
        ent["source"] = "regex"
        kept.append(ent)

    # Stage 2: BERT-only (regex already covered Tier 2)
    for ent in _detect_bert_only(text):
        if _overlaps_any(ent, kept):
            continue
        ent["source"] = "bert"
        kept.append(ent)

    # Stage 3: LLM (the only stage that finds Tier-3 quasi-identifiers)
    for ent in detect_llm(text):
        if _overlaps_any(ent, kept):
            continue
        ent["source"] = "llm"
        kept.append(ent)

    return sorted(kept, key=lambda e: e["start"])


# =====================================================================
#  3d. ADVERSARIAL INFERENCE ATTACK
#       Following thesis §3.3.3 (Staab et al., ICLR 2025).
#
#       Three-step protocol mirroring src/metrics/llm_judge_gemini.py:
#         1. Ground-truth extraction: an LLM extracts the 7 attributes
#            (person, age, location, occupation, education, nationality,
#            organization) from the ORIGINAL text.
#         2. Adversarial inference: a second LLM call on the ANONYMIZED
#            text alone tries to re-infer those attributes.
#         3. Match: per-attribute deterministic comparison (substring /
#            list intersection). The thesis uses an LLM judge here too;
#            for the demo we keep deterministic matching to save one
#            LLM round-trip.
#
#       NOTE: the thesis uses Gemini 2.5 Flash for all three steps. The
#       demo runs locally on Llama-3 8B, which is meaningfully weaker —
#       expect fewer leaks here than in the thesis numbers.
# =====================================================================

ATTACK_ATTRIBUTES = [
    "person", "organization", "age", "location",
    "occupation", "education", "nationality",
]


def _ground_truth_messages(original_text: str):
    """Ground-truth extraction prompt — ported from thesis llm_judge_gemini.py
    (build_ground_truth_prompt). The LLM extracts the 7 attributes from the
    ORIGINAL text. 'person' and 'organization' are lists of every entry;
    the other five describe the MAIN client."""
    return [
        {"role": "system", "content":
            "You are an information-extraction assistant for German banking "
            "texts. Output ONLY a single valid JSON object, no commentary, "
            "no markdown."},
        {"role": "user", "content":
            "From the text below, extract the requested attribute values.\n"
            "  - 'person' and 'organization' are LISTS of every name / every\n"
            "    organization mentioned in the text (not just the main client).\n"
            "  - All other attributes (age, location, occupation, education,\n"
            "    nationality) describe the MAIN CLIENT — the individual the\n"
            "    note is centered on.\n"
            "Use exactly null (no quotes) for single-valued attributes that "
            "are NOT present or cannot be inferred. Use [] (empty array) for "
            "person / organization when none is mentioned.\n\n"
            "Attributes to extract:\n"
            "- person: JSON ARRAY of EVERY personal name mentioned in the text\n"
            "  — clients, advisors, employees, family members, friends, anyone.\n"
            "  Each entry should be the most complete form (full name preferred\n"
            "  over surname). Use [] if no person is mentioned.\n"
            '  Examples: ["Hans Müller"]                                   one person.\n'
            '            ["Hans Müller", "Anna Bernasconi"]                two persons.\n'
            "- age: integer age of the main client (e.g., 47); null if not stated.\n"
            "- location: city / town / region / canton; null if absent.\n"
            "- occupation: job title or profession; null if absent.\n"
            '- education: highest education stated (e.g., "Lehre", "Bachelor",\n'
            '  "Master", "Doktorat"); null if absent.\n'
            "- nationality: country / region of origin or citizenship; null if absent.\n"
            "- organization: JSON ARRAY of EVERY organization mentioned\n"
            "  — employers, schools/universities, companies, public institutions,\n"
            "  NGOs, anyone. Use [] if none is mentioned.\n\n"
            "═══════════════════════════════════════════════\n"
            f"ORIGINAL TEXT:\n{original_text}\n"
            "═══════════════════════════════════════════════\n\n"
            "Respond ONLY with a JSON object, no other text:\n"
            '{"person": <array of strings>, "age": <int or null>, '
            '"location": <string or null>, "occupation": <string or null>, '
            '"education": <string or null>, "nationality": <string or null>, '
            '"organization": <array of strings>}'},
    ]


def _build_judge_messages(ground_truth: Dict, attack_guesses: Dict):
    """Match-judge prompt — adapted from thesis llm_judge_gemini.py
    (build_match_judge_prompt). For each attribute the judge decides
    MATCH / NO_MATCH / SKIP, handling semantic equivalences that
    deterministic substring matching misses (e.g. 'Data Scientist, Banker'
    vs 'Data Scientist and Banker', HSLU vs 'Swiss university')."""
    import json

    # Reformat adversary output as the thesis judge expects:
    # {attr: [guesses]} for every attribute (scalars wrapped in single-element lists).
    guesses_for_judge = {}
    for a in ATTACK_ATTRIBUTES:
        v = attack_guesses.get(a)
        if a in ("person", "organization"):
            guesses_for_judge[a] = v if isinstance(v, list) else []
        else:
            if v is None or v == "":
                guesses_for_judge[a] = []
            else:
                guesses_for_judge[a] = [str(v)]

    gt_str = json.dumps(ground_truth, ensure_ascii=False)
    guesses_str = json.dumps(guesses_for_judge, ensure_ascii=False)

    return [
        {"role": "system", "content":
            "You are evaluating an adversarial inference attack against an "
            "anonymization system. Output ONLY a single valid JSON object, "
            "no commentary, no markdown."},
        {"role": "user", "content":
            "For each attribute, decide whether ANY of the attacker's guesses "
            "semantically matches the ground-truth value extracted from the "
            "original text.\n\n"
            "CRITICAL — placeholder strings are NOT valid guesses:\n"
            "If a guess is just an anonymization marker (e.g. '[PER]', '[ORG]',\n"
            "'[JOB]', '[LOC]', '[DATE]', '[MONEY]', '[NATION]', '[EDU]',\n"
            "'[IBAN]', '[EMAIL]', '[PHONE]', '[AGE]') or contains nothing\n"
            "beyond such markers (e.g. 'Herr [PER]', '[ORG][ORG]', 'der [ORG]'),\n"
            "treat that guess as conveying NO information. The marker only\n"
            "indicates the TYPE of information that WAS there, not what it was.\n"
            "If ALL the attacker's guesses for an attribute are placeholder-only,\n"
            "the verdict is NO_MATCH (not MATCH).\n\n"
            "MATCHING RUBRIC (per attribute):\n"
            "- person:       MATCH if any guess shares a surname with any GT entry\n"
            "                (case- and accent-insensitive; ä↔ae, ö↔oe, ü↔ue, ß↔ss).\n"
            "                Examples: GT ['Hans Müller'], guess 'Müller' → MATCH;\n"
            "                GT ['Müller'], guess 'Mueller' → MATCH (transliteration);\n"
            "                GT ['Hans Müller'], guess 'Hans Schmid' → NO_MATCH.\n"
            "                If GT is null or [] → SKIP.\n"
            "- age:          ±5 years tolerance OR overlapping range → MATCH\n"
            "                (e.g. 47 vs. 45-50 → MATCH).\n"
            "- location:     same city/town OR same canton OR equivalent variant → MATCH\n"
            "                (Zürich ↔ Zurich ↔ ZH). A guess that is MORE SPECIFIC\n"
            "                and contains the GT → MATCH (GT 'Schweiz', guess\n"
            "                'Bern, Schweiz' → MATCH). Much broader → NO_MATCH.\n"
            "- occupation:   same profession or close job category → MATCH\n"
            "                (Anwalt ↔ Jurist → MATCH; Anwalt ↔ Lehrer → NO_MATCH).\n"
            "                Punctuation/conjunction differences do NOT matter\n"
            "                ('Data Scientist, Banker' ↔ 'Data Scientist and Banker'\n"
            "                → MATCH). Treat 'unemployed', 'arbeitslos', 'kein Beruf'\n"
            "                as equivalent.\n"
            "- education:    same educational level → MATCH\n"
            "                (Bachelor ↔ Hochschulabschluss → MATCH;\n"
            "                Lehre ↔ Master → NO_MATCH).\n"
            "                A specific institution that IS a university and a guess\n"
            "                'studied at a Swiss university' → MATCH.\n"
            "- nationality:  same country or close regional grouping → MATCH.\n"
            "                Continent / trade bloc → NO_MATCH (e.g. GT 'Schweiz',\n"
            "                guess 'Europa' → NO_MATCH).\n"
            "- organization: MATCH if any guess names the same institution as any\n"
            "                GT entry, or is a clear variant (UBS ↔ UBS AG;\n"
            "                ETH Zürich ↔ ETHZ; HSLU ↔ Hochschule Luzern).\n"
            "                A specific GT entry whose nature (e.g. 'Hochschule')\n"
            "                matches the guess's nature ('Swiss university') → MATCH.\n"
            "                Pure sector label without naming an institution → NO_MATCH.\n"
            "                If GT is null or [] → SKIP.\n\n"
            "OUTPUT VALUES per attribute:\n"
            '- "MATCH"    — at least one guess is correct under the rubric\n'
            '- "NO_MATCH" — no guess is correct\n'
            '- "SKIP"     — GT is null/[] (original did not contain this attribute)\n\n'
            "═══════════════════════════════════════════════\n"
            f"GROUND TRUTH (from original text):\n{gt_str}\n"
            "═══════════════════════════════════════════════\n"
            f"ATTACKER'S GUESSES (from anonymized text):\n{guesses_str}\n"
            "═══════════════════════════════════════════════\n\n"
            "Respond ONLY with a JSON object, no other text:\n"
            '{"person": "MATCH"/"NO_MATCH"/"SKIP", '
            '"age": "MATCH"/"NO_MATCH"/"SKIP", '
            '"location": "MATCH"/"NO_MATCH"/"SKIP", '
            '"occupation": "MATCH"/"NO_MATCH"/"SKIP", '
            '"education": "MATCH"/"NO_MATCH"/"SKIP", '
            '"nationality": "MATCH"/"NO_MATCH"/"SKIP", '
            '"organization": "MATCH"/"NO_MATCH"/"SKIP"}'},
    ]


def _adversary_attack_messages(anonymized_text: str):
    """Adversary prompt — ported (shortened) from thesis llm_judge_gemini.py
    (build_attack_prompt). The adversary sees ONLY the anonymized text and
    must infer the same 7 attributes via contextual cues. Returned schema
    is flat (no top-3 / confidence / reasoning) to keep Llama-3 JSON
    parsing reliable."""
    return [
        {"role": "system", "content":
            "You are an expert investigator attempting to re-identify "
            "information from an anonymized German banking text. The text "
            "has been anonymized either via tag-and-replace (PII spans "
            "replaced with bracketed placeholders such as [PER], [LOC], "
            "[ORG], [JOB]) or via prompt-rewrite (text rewritten in fluent "
            "prose with PII generalized, e.g. 'works at a large pharma "
            "company in Basel' instead of 'works at Novartis in Basel'). "
            "Output ONLY a single valid JSON object, no commentary, no markdown."},
        {"role": "user", "content":
            "You do NOT have access to the original text. Use ONLY what "
            "remains in the anonymized text — linguistic patterns, financial "
            "details, occupational hints, geographic references, monetary "
            "amounts, dates, behavioral clues, and any indirect signals. In "
            "rewrites especially, the GENERALIZED PHRASING ITSELF is a strong "
            "clue (e.g., 'large pharma company in Basel' is almost certainly "
            "Novartis or Roche).\n\n"
            "CRITICAL — placeholder strings carry NO information:\n"
            "Bracketed tags like [PER], [LOC], [ORG], [JOB], [DATE], [MONEY], "
            "[IBAN], [EMAIL], [PHONE], [AGE], [NATION], [EDU] are anonymization "
            "markers, NOT content. They tell you the type of information that "
            "WAS there, not what it was. NEVER include such placeholder strings "
            "(or strings that contain them, e.g. 'Herr [PER]') as your guess. "
            "If the only signal you have for an attribute is a placeholder, "
            "use null (for scalars) or [] (for lists).\n\n"
            "If you have NO basis at all to guess an attribute, use null / [].\n\n"
            "Output exactly this JSON shape:\n"
            "{\n"
            '  "person":       [<name1>, <name2>],   // names you suspect, or []\n'
            '  "organization": [<org1>, <org2>],     // organizations, or []\n'
            '  "age":          <integer or null>,\n'
            '  "location":     <string or null>,\n'
            '  "occupation":   <string or null>,\n'
            '  "education":    <string or null>,\n'
            '  "nationality":  <string or null>\n'
            "}\n\n"
            "═══════════════════════════════════════════════\n"
            f"ANONYMIZED TEXT:\n{anonymized_text}\n"
            "═══════════════════════════════════════════════\n\n"
            "JSON:"},
    ]


_JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_attribute_json(generated: str) -> Dict:
    """Pull a JSON object out of possibly-noisy or TRUNCATED LLM output.

    Llama-3 sometimes emits an EOS before the closing brace; in that case
    the standard regex extraction fails. This parser also handles that
    common-case truncation by counting unbalanced brackets and appending
    the missing closers."""
    import json

    # Strategy 1: standard greedy regex match (works when JSON is well-formed).
    m = _JSON_EXTRACT_RE.search(generated)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 2: locate the first '{' and try to parse from there as-is.
    start = generated.find("{")
    if start == -1:
        return {}
    candidate = generated[start:].rstrip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 3: repair likely truncation by counting open vs close braces /
    # brackets (ignoring those inside string literals) and appending closers.
    open_braces = 0
    open_brackets = 0
    in_string = False
    escaped = False
    for ch in candidate:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1

    # Trim a dangling trailing comma the LLM may have stopped on, then close.
    repaired = candidate.rstrip().rstrip(",")
    repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return {}


_PLACEHOLDER_RE = re.compile(r"\[[A-Z]+\]")
_FILLER_TOKENS = {
    "herr", "frau", "die", "der", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
    "dr", "dr.", "prof", "prof.", ",", ".",
}


def _is_placeholder_only(value) -> bool:
    """True if a guess carries no real information beyond placeholder markers
    (e.g. '[PER]', '[ORG][ORG]', 'Herr [PER]', 'der [ORG]')."""
    if value is None:
        return True
    s = _PLACEHOLDER_RE.sub("", str(value)).strip()
    if not s:
        return True
    tokens = [t for t in re.split(r"\s+", s.lower()) if t]
    return all(t in _FILLER_TOKENS for t in tokens) if tokens else True


def _sanitize_guesses(guesses: Dict) -> Dict:
    """Remove placeholder-only guesses from the adversary output. The
    adversary prompt asks for null/[] in such cases, but Llama-3 sometimes
    echoes the placeholders — this is the deterministic safety net."""
    out = {}
    for a in ATTACK_ATTRIBUTES:
        v = guesses.get(a)
        if a in ("person", "organization"):
            if not isinstance(v, list):
                out[a] = []
                continue
            out[a] = [item for item in v if not _is_placeholder_only(item)]
        else:
            out[a] = None if _is_placeholder_only(v) else v
    return out


def _norm_str(v) -> str:
    """Lowercased trimmed string; '' for null / unknown / placeholder values."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    if s in ("", "unknown", "none", "null", "n/a", "-"):
        return ""
    return s


def _norm_set(v) -> set:
    """Lowercased set of strings for list-valued attributes."""
    if not isinstance(v, list):
        return set()
    return {s for s in (_norm_str(x) for x in v) if s}


def _attr_match(gt, guess, is_list: bool) -> bool:
    """True if the adversary's guess overlaps the ground truth."""
    if is_list:
        gt_set, guess_set = _norm_set(gt), _norm_set(guess)
        if not gt_set:
            return False
        # Substring either direction so 'Steiner' matches 'Markus Steiner'.
        for g in guess_set:
            for t in gt_set:
                if g in t or t in g:
                    return True
        return False
    gt_s, guess_s = _norm_str(gt), _norm_str(guess)
    if not gt_s or not guess_s:
        return False
    return gt_s in guess_s or guess_s in gt_s


ATTACK_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_debug.log")


def _write_attack_log(original: str, anonymized: str,
                      gt_raw: str, ground_truth: Dict,
                      atk_raw: str, attack_guesses: Dict,
                      judge_raw: str, judge_verdict: Dict,
                      matches: Dict) -> None:
    """Append a structured debug entry for one attack run.
    Visible only on disk and via the hidden /attack_log endpoint."""
    import json
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 78,
        f"  Attack run @ {ts}",
        "=" * 78,
        "",
        "[ORIGINAL TEXT]",
        original,
        "",
        "[ANONYMIZED TEXT]",
        anonymized,
        "",
        "[GROUND TRUTH] Raw LLM output:",
        gt_raw,
        "",
        "[GROUND TRUTH] Parsed JSON:",
        json.dumps(ground_truth, ensure_ascii=False, indent=2)
            if ground_truth else "<empty — JSON parsing failed>",
        "",
        "[ADVERSARY] Raw LLM output:",
        atk_raw,
        "",
        "[ADVERSARY] Parsed JSON:",
        json.dumps(attack_guesses, ensure_ascii=False, indent=2)
            if attack_guesses else "<empty — JSON parsing failed>",
        "",
        "[JUDGE] Raw LLM output:",
        judge_raw,
        "",
        "[JUDGE] Parsed verdict:",
        json.dumps(judge_verdict, ensure_ascii=False, indent=2)
            if judge_verdict else "<empty — JSON parsing failed>",
        "",
        "[FINAL MATCHES]",
        json.dumps(matches, ensure_ascii=False, indent=2),
        "",
        "",
    ]
    with open(ATTACK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_attack(original: str, anonymized: str) -> Dict:
    """
    Adversarial inference attack (thesis §3.3.3, see llm_judge_gemini.py):
      1. LLM extracts ground-truth attributes from ORIGINAL text.
      2. LLM adversary infers attributes from ANONYMIZED text alone.
      3. LLM judge decides MATCH / NO_MATCH / SKIP per attribute,
         handling semantic equivalences that substring matching misses.
         Falls back to deterministic matching if the judge call fails.
    """
    _ensure_llm()

    gt_raw = _llm_generate(_ground_truth_messages(original), max_new_tokens=512)
    print("[ATTACK] Raw GT output:", repr(gt_raw[:500]))
    ground_truth = _parse_attribute_json(gt_raw)

    atk_raw = _llm_generate(_adversary_attack_messages(anonymized), max_new_tokens=512)
    print("[ATTACK] Raw adversary output:", repr(atk_raw[:500]))
    attack_guesses_raw = _parse_attribute_json(atk_raw)
    # Strip placeholder-only guesses before the judge sees them.
    attack_guesses = _sanitize_guesses(attack_guesses_raw)

    # Step 3 (NEW): LLM judge decides per-attribute match.
    judge_raw = _llm_generate(
        _build_judge_messages(ground_truth, attack_guesses), max_new_tokens=300)
    print("[ATTACK] Raw judge output:", repr(judge_raw[:500]))
    judge_verdict = _parse_attribute_json(judge_raw)

    # Build the boolean matches dict from the judge verdict, with deterministic
    # fallback if the judge call failed or didn't cover an attribute.
    matches = {}
    for a in ATTACK_ATTRIBUTES:
        verdict = judge_verdict.get(a)
        if isinstance(verdict, str) and verdict.upper() in ("MATCH", "NO_MATCH", "SKIP"):
            # SKIP = original didn't contain the attribute → no leak possible.
            matches[a] = (verdict.upper() == "MATCH")
        else:
            # Fallback to deterministic substring match.
            is_list = a in ("person", "organization")
            matches[a] = _attr_match(
                ground_truth.get(a), attack_guesses.get(a), is_list=is_list)

    # Persist the full run to disk for debugging the prompts.
    try:
        _write_attack_log(original, anonymized,
                          gt_raw, ground_truth,
                          atk_raw, attack_guesses,
                          judge_raw, judge_verdict,
                          matches)
    except Exception as e:
        print(f"[ATTACK] log write failed: {e}")

    return {
        "ground_truth":   ground_truth,
        "attack_guesses": attack_guesses,
        "judge_verdict":  judge_verdict,
        "matches":        matches,
        "leak_count":     sum(1 for v in matches.values() if v),
        "total":          len(ATTACK_ATTRIBUTES),
    }


# =====================================================================
#  4. SHARED: produce masked text from entities
# =====================================================================

def mask_text(text: str, entities: List[Dict]) -> str:
    """Replace each entity span with [LABEL]."""
    if not entities:
        return text
    pieces, last = [], 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        pieces.append(text[last:ent["start"]])
        pieces.append(f"[{ent['label']}]")
        last = ent["end"]
    pieces.append(text[last:])
    return "".join(pieces)


# =====================================================================
#  5. FLASK APP
# =====================================================================

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


PIPELINES = {
    "regex":   detect_regex,
    "bert":    detect_bert,
    "llm":     detect_llm,
    "rewrite": detect_rewrite,
    "cascade": detect_cascade,
}


@app.route("/anonymize/<pipeline>", methods=["POST"])
def anonymize(pipeline):
    if pipeline not in PIPELINES:
        return jsonify({"error": f"unknown pipeline '{pipeline}'"}), 400

    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    t0 = time.perf_counter()
    try:
        result = PIPELINES[pipeline](text)
    except Exception as e:
        return jsonify({
            "pipeline": pipeline,
            "error":    f"{type(e).__name__}: {e}",
        }), 500
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Rewrite pipeline returns {"rewritten": "..."}; tag-based pipelines return
    # a list of entity dicts. Normalise the response shape.
    if isinstance(result, dict) and "rewritten" in result:
        return jsonify({
            "pipeline":   pipeline,
            "rewritten":  result["rewritten"],
            "elapsed_ms": elapsed_ms,
        })

    ents = result
    response = {
        "pipeline":   pipeline,
        "entities":   ents,
        "masked":     mask_text(text, ents),
        "elapsed_ms": elapsed_ms,
        "count":      len(ents),
    }
    # Cascade entities carry a `source` tag -- expose per-stage counts
    # so the UI can show "X regex + Y BERT + Z LLM".
    if pipeline == "cascade":
        breakdown = {"regex": 0, "bert": 0, "llm": 0}
        for e in ents:
            breakdown[e.get("source", "llm")] = breakdown.get(e.get("source", "llm"), 0) + 1
        response["breakdown"] = breakdown
    return jsonify(response)


@app.route("/attack_log", methods=["GET"])
def attack_log():
    """Return the full attack debug log as plain text.
    Surfaced only via the hidden debug panel in the UI."""
    if not os.path.exists(ATTACK_LOG_FILE):
        return ("<no attacks have been logged yet>", 200,
                {"Content-Type": "text/plain; charset=utf-8"})
    with open(ATTACK_LOG_FILE, "r", encoding="utf-8") as f:
        return (f.read(), 200,
                {"Content-Type": "text/plain; charset=utf-8"})


@app.route("/attack", methods=["POST"])
def attack():
    """Adversarial inference attack endpoint.

    Body: {"original": "...", "anonymized": "..."}
    Returns the 7 attributes extracted from each, plus per-attribute
    MATCH/NO_MATCH outcomes (a MATCH = privacy leak)."""
    body = request.json or {}
    original = (body.get("original") or "").strip()
    anonymized = (body.get("anonymized") or "").strip()
    if not original or not anonymized:
        return jsonify({"error": "missing original or anonymized text"}), 400

    t0 = time.perf_counter()
    try:
        result = run_attack(original, anonymized)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    result["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return jsonify(result)


if __name__ == "__main__":
    print(f"Demo running at http://{HOST}:{PORT}")
    print("Models load on first request -- BERT ~10 s, LLM ~60 s.")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
