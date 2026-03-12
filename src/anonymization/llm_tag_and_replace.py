"""
llm_tag_and_replace.py
======================
LLM-based PII detection via tag-and-replace, inspired by GPT-NER
(Wang et al., 2025, NAACL).

The LLM copies the input text and marks entities with @@LABEL ...##
tokens. A deterministic parser then extracts character-offset spans.
Optionally, a self-verification step asks the LLM to confirm each
extracted entity, reducing false positives.

Supported models:
  - meta-llama/Meta-Llama-3-8B-Instruct   (general-purpose baseline)
  - Qwen/Qwen2.5-7B-Instruct             (strongest 7B-class model)
  - VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct  (German-specialized)

How to use:
  1. Uncomment ONE model block in USER SETTINGS below
  2. Press Run in VS Code
  3. Repeat for each model (results auto-named, no overwrites)

Requirements:
    pip install transformers torch accelerate tqdm bitsandbytes
    (evaluation_utils.py must be importable)

Author: André Kuhn – Master Thesis (MScIDS, HSLU)
"""

# =====================================================================
#  USER SETTINGS - Change these before each run, then press Run
# =====================================================================
#
#  Available models:
#    "meta-llama/Meta-Llama-3-8B-Instruct"   (16 GB, float16)  — general-purpose baseline
#    "Qwen/Qwen2.5-7B-Instruct"             (15 GB, float16)  — strongest 7B-class model
#    "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct"  (16 GB, float16)  — German-specialized
#
#  Available strategies:
#    "zero-shot"    (no examples in prompt)
#    "few-shot"     (3 annotated examples in prompt)
#
#  Run configurations — uncomment ONE block at a time:
# =====================================================================

# --- Llama-3 8B (your existing baseline) ---
MODEL       = "meta-llama/Meta-Llama-3-8B-Instruct"
QUANTIZE    = False
STRATEGY    = "few-shot"
VERIFY      = True
MAX_DOCS    = None

# --- Qwen2.5 7B (strongest small model) ---
# MODEL       = "Qwen/Qwen2.5-7B-Instruct"
# QUANTIZE    = False
# STRATEGY    = "few-shot"
# VERIFY      = True
# MAX_DOCS    = None

# --- SauerkrautLM 8B (German-specialized) ---
# MODEL       = "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct"
# QUANTIZE    = False
# STRATEGY    = "few-shot"
# VERIFY      = True
# MAX_DOCS    = None


# Paths (should not need changing)
INPUT_PATH  = r"C:\thesis\data\label_studio\20260302_Export_Label_Studio_Client_Notes.json"
OUTPUT_DIR  = r"C:\thesis\results\llm_baselines"
SPLIT_IDS   = r"C:\thesis\results\bert_finetuned\split_ids.json"
SEED        = 42

import json
import os
import re
import time
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

from evaluation_utils import (
    ALL_LABELS,
    evaluate_tiered,
    format_tiered_report,
    save_results_json,
    generate_error_samples,
    generate_full_document_log,
    generate_category_error_report,
)


# -----------------------------------------
#  1. CONFIGURATION
# -----------------------------------------

# Category descriptions for the prompt
LABEL_DESCRIPTIONS = {
    "PER":    "Person names (full names, surnames, first names - exclude titles like Herr/Frau/Dr.)",
    "LOC":    "Locations (cities, countries, addresses, regions, street names)",
    "ORG":    "Organizations (company names including legal suffixes like AG, GmbH, SA)",
    "DATE":   "Dates and time references (exact dates, relative dates like 'morgen', 'Ende Monat', durations like '3 Monate')",
    "EMAIL":  "Email addresses",
    "PHONE":  "Phone numbers (Swiss +41 and local 0xx formats, international formats)",
    "IBAN":   "Bank account numbers (Swiss IBAN format: CH followed by digits/letters in groups)",
    "MONEY":  "Monetary amounts and standalone currency codes (CHF, EUR, USD with or without amounts, 'Mio.', 'k')",
    "JOB":    "Job titles and professional roles (CEO, CFO, Geschaeftsfuehrer, Leiter Finanzen, Projektleiter, Buchhalter, etc.)",
    "AGE":    "Age references (e.g., '65 Jahre', '48', '68-jaehrig', '68-jaehriger')",
    "NATION": "Nationality adjectives and demonyms (e.g., 'deutscher', 'franzoesische', 'Schweizer', 'europaeischen')",
    "EDU":    "Educational institutions and qualifications (ETH, HSG, MBA, EPFL, ETH-Absolvent, HSG-Absolventin, etc.)",
}


def build_label_description_block() -> str:
    """Format label descriptions for inclusion in prompts."""
    lines = []
    for label in sorted(LABEL_DESCRIPTIONS.keys()):
        lines.append(f"- {label}: {LABEL_DESCRIPTIONS[label]}")
    return "\n".join(lines)


# -----------------------------------------
#  2. PROMPT CONSTRUCTION (@@LABEL...## format)
# -----------------------------------------

def build_system_prompt() -> str:
    """
    System prompt using the @@LABEL ...## marking format (GPT-NER style).
    The LLM copies the input text and wraps entities with special markers.
    """
    return (
        "You are an expert Named Entity Recognition (NER) system specialized in "
        "identifying personally identifiable information (PII) in German-language "
        "financial communications from a Swiss banking context.\n\n"
        "Your task: Given a text, COPY the entire text exactly and mark ALL entities "
        "by wrapping them with @@LABEL and ## tokens, where LABEL is the entity category.\n\n"
        "Entity categories:\n"
        f"{build_label_description_block()}\n\n"
        "RULES:\n"
        "1. Copy the ENTIRE input text character by character.\n"
        "2. Wrap each entity with @@LABEL before it and ## after it.\n"
        "   Example: 'Herr Markus Steiner, CEO' becomes 'Herr @@PER Markus Steiner##, @@JOB CEO##'\n"
        "3. Tag EVERY occurrence, including repeated mentions of the same entity.\n"
        "4. For compound German expressions (e.g., 'HSG-Absolventin'), tag the FULL compound.\n"
        "5. Nationality adjectives (e.g., 'deutscher', 'schweizerisch') belong to NATION, not LOC.\n"
        "6. Standalone currency codes (CHF, EUR, USD) without amounts are still MONEY.\n"
        "7. Person names should NOT include titles like Herr, Frau, Dr.\n"
        "8. If no entities are found, just copy the text unchanged.\n"
        "9. Output ONLY the marked text. No explanations, no preamble, no markdown.\n"
    )


# -- Few-Shot Examples --
FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "Betriebsbesichtigung bei der Biofood Produzenten GmbH in Bern am 12.05.2024. "
            "Der Geschaeftsfuehrer, Herr Markus Aebischer, fuehrte durch die neuen Anlagen. "
            "Das Unternehmen verzeichnet ein starkes Wachstum im europaeischen Markt. "
            "Zur Finanzierung des weiteren Ausbaus wird eine Trade & Export Finance (TEF) "
            "Loesung fuer Lieferungen nach Deutschland und Oesterreich geprueft. Das aktuelle "
            "Volumen betraegt ca. CHF 750'000 pro Monat. Herr Aebischer ist 48 Jahre alt "
            "und hat an der ETH Zuerich studiert."
        ),
        "output": (
            "Betriebsbesichtigung bei der @@ORG Biofood Produzenten GmbH## in @@LOC Bern## am @@DATE 12.05.2024##. "
            "Der @@JOB Geschaeftsfuehrer##, Herr @@PER Markus Aebischer##, fuehrte durch die neuen Anlagen. "
            "Das Unternehmen verzeichnet ein starkes Wachstum im @@NATION europaeischen## Markt. "
            "Zur Finanzierung des weiteren Ausbaus wird eine Trade & Export Finance (TEF) "
            "Loesung fuer Lieferungen nach @@LOC Deutschland## und @@LOC Oesterreich## geprueft. Das aktuelle "
            "Volumen betraegt ca. @@MONEY CHF 750'000## pro Monat. Herr @@PER Aebischer## ist @@AGE 48 Jahre## alt "
            "und hat an der @@EDU ETH Zuerich## studiert."
        ),
    },
    {
        "input": (
            "Telefonat mit Frau Sandra Wyss (079 111 22 33), Buchhalterin bei der "
            "Bau-Expert GmbH am 30. April 2024. Sie informierte uns ueber die Saldierung "
            "des Kontos mit der CH33 0077 7888 9999 0000 1, da die entsprechende "
            "Projektgesellschaft liquidiert wurde. Der Restbetrag von CHF 12'450.50 soll "
            "auf das Hauptkonto ueberwiesen werden. Die rechtsverbindliche Unterschrift des "
            "Geschaeftsfuehrers Peter Schmid liegt uns vor."
        ),
        "output": (
            "Telefonat mit Frau @@PER Sandra Wyss## (@@PHONE 079 111 22 33##), @@JOB Buchhalterin## bei der "
            "@@ORG Bau-Expert GmbH## am @@DATE 30. April 2024##. Sie informierte uns ueber die Saldierung "
            "des Kontos mit der @@IBAN CH33 0077 7888 9999 0000 1##, da die entsprechende "
            "Projektgesellschaft liquidiert wurde. Der Restbetrag von @@MONEY CHF 12'450.50## soll "
            "auf das Hauptkonto ueberwiesen werden. Die rechtsverbindliche Unterschrift des "
            "@@JOB Geschaeftsfuehrers## @@PER Peter Schmid## liegt uns vor."
        ),
    },
    {
        "input": (
            "Am 28. Februar 2024 fand das Eroeffnungsgespraech mit der neu gegruendeten "
            "Pharma Spin-Off AG statt. Der designierte CEO, Dr. Martin Fischer, ein "
            "ETH-Absolvent, benoetigt diverse Firmenkonten in CHF und EUR. Die notwendigen "
            "KYC-Dokumente, inklusive Handelsregisterauszug aus dem Kanton Basel-Stadt, "
            "wurden uebergeben. Kontakt fuer die technische Anbindung des Cash Managements "
            "ist martin.fischer@pharmaspin.ch."
        ),
        "output": (
            "Am @@DATE 28. Februar 2024## fand das Eroeffnungsgespraech mit der neu gegruendeten "
            "@@ORG Pharma Spin-Off AG## statt. Der designierte @@JOB CEO##, Dr. @@PER Martin Fischer##, ein "
            "@@EDU ETH-Absolvent##, benoetigt diverse Firmenkonten in @@MONEY CHF## und @@MONEY EUR##. Die notwendigen "
            "KYC-Dokumente, inklusive Handelsregisterauszug aus dem Kanton @@LOC Basel-Stadt##, "
            "wurden uebergeben. Kontakt fuer die technische Anbindung des Cash Managements "
            "ist @@EMAIL martin.fischer@pharmaspin.ch##."
        ),
    },
]


def build_user_prompt_zero_shot(text: str) -> str:
    """Zero-shot: just the text to mark."""
    return f"Mark all PII entities in the following text:\n\nInput: {text}\nOutput:"


def build_user_prompt_few_shot(text: str) -> str:
    """Few-shot: include examples before the target text."""
    parts = []
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"Example {i}:\nInput: {example['input']}\nOutput: {example['output']}")

    parts.append(f"Now mark all PII entities in this text:\n\nInput: {text}\nOutput:")
    return "\n\n".join(parts)


# -----------------------------------------
#  3. MODEL LOADING
# -----------------------------------------

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


# -----------------------------------------
#  4. INFERENCE
# -----------------------------------------

def format_chat_messages(system_prompt: str, user_prompt: str) -> List[Dict]:
    """Build a chat-format message list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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


# -----------------------------------------
#  5. OUTPUT PARSING (@@LABEL...## format)
# -----------------------------------------

def parse_marked_text(response: str, original_text: str) -> Tuple[List[Dict], str]:
    """
    Parse @@LABEL entity_text## markers from the LLM response.

    Returns:
        entities:     list of {"text": ..., "label": ...}
        parse_method: "marked", "json_fallback", "xml_fallback", or "failed"
    """
    # Build regex for all labels: @@PER ...## or @@LOC ...## etc.
    label_pattern = "|".join(sorted(ALL_LABELS))
    pattern = rf"@@({label_pattern})\s+(.*?)##"

    entities = []
    for match in re.finditer(pattern, response, re.DOTALL):
        label = match.group(1).strip()
        entity_text = match.group(2).strip()
        if entity_text and label in ALL_LABELS:
            entities.append({"text": entity_text, "label": label})

    if entities:
        return entities, "marked"

    # Fallback 1: try JSON parsing (in case the LLM ignores the format instruction)
    json_entities = _parse_json_fallback(response)
    if json_entities:
        return json_entities, "json_fallback"

    # Fallback 2: try XML-style tags
    xml_entities = _parse_xml_fallback(response)
    if xml_entities:
        return xml_entities, "xml_fallback"

    return [], "failed"


def _parse_json_fallback(response: str) -> Optional[List[Dict]]:
    """Try to parse as JSON array (fallback if LLM ignores @@## format)."""
    cleaned = re.sub(r"```json\s*", "", response)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start == -1 or bracket_end == -1 or bracket_end <= bracket_start:
        return None

    json_str = cleaned[bracket_start:bracket_end + 1]
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "text" in item and "label" in item:
                    label = item["label"].upper().strip()
                    if label in ALL_LABELS:
                        valid.append({"text": item["text"], "label": label})
            return valid if valid else None
    except json.JSONDecodeError:
        pass
    return None


def _parse_xml_fallback(response: str) -> Optional[List[Dict]]:
    """Try to parse XML-style tags as fallback."""
    entities = []
    pattern = r"<(" + "|".join(ALL_LABELS) + r")>(.*?)</\1>"
    for match in re.finditer(pattern, response):
        label = match.group(1)
        text = match.group(2).strip()
        if text and label in ALL_LABELS:
            entities.append({"text": text, "label": label})
    return entities if entities else None


# -----------------------------------------
#  6. TEXT-TO-SPAN ALIGNMENT
# -----------------------------------------

def align_entities_to_text(text: str, raw_entities: List[Dict]) -> List[Dict]:
    """
    Convert text-match entities into character-offset entities.
    Finds all occurrences of each entity text in the source document.
    """
    aligned = []
    used_spans = set()

    entity_groups = defaultdict(int)
    for ent in raw_entities:
        entity_groups[(ent["text"], ent["label"])] += 1

    for (entity_text, label), expected_count in entity_groups.items():
        if not entity_text:
            continue

        search_start = 0
        occurrences_found = 0

        while search_start < len(text) and occurrences_found < expected_count:
            idx = text.find(entity_text, search_start)
            if idx == -1:
                idx_lower = text.lower().find(entity_text.lower(), search_start)
                if idx_lower == -1:
                    break
                idx = idx_lower

            span = (idx, idx + len(entity_text))
            if span not in used_spans:
                aligned.append({
                    "start": span[0], "end": span[1],
                    "label": label, "text": text[span[0]:span[1]],
                })
                used_spans.add(span)
                occurrences_found += 1

            search_start = idx + len(entity_text)

    aligned.sort(key=lambda e: e["start"])
    return aligned


# -----------------------------------------
#  7. SELF-VERIFICATION (optional)
# -----------------------------------------

def build_verification_prompt(sentence: str, entity_text: str, label: str) -> str:
    """
    Build a verification prompt that asks the LLM to confirm whether
    an extracted entity is correct (GPT-NER self-verification strategy).
    """
    label_desc = LABEL_DESCRIPTIONS.get(label, label)
    return (
        f"You are verifying named entity recognition results in German financial text.\n\n"
        f"Entity category: {label} - {label_desc}\n\n"
        f"Sentence: {sentence}\n\n"
        f'Is "{entity_text}" a {label} entity in the sentence above?\n'
        f"Answer ONLY with yes or no."
    )


def verify_entities(
    model, tokenizer,
    text: str,
    entities: List[Dict],
) -> List[Dict]:
    """
    Run self-verification on each extracted entity.
    Keeps only entities where the LLM confirms "yes".

    Args:
        text:      the original document text
        entities:  list of aligned entities [{start, end, label, text}]

    Returns:
        verified entities (subset of input)
    """
    if not entities:
        return entities

    verified = []

    for ent in entities:
        prompt = build_verification_prompt(text, ent["text"], ent["label"])
        messages = [{"role": "user", "content": prompt}]

        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=10, temperature=0.0,
        )

        answer = response.strip().lower()
        # Accept if the response starts with "yes" or "ja" (German)
        if answer.startswith("yes") or answer.startswith("ja"):
            verified.append(ent)

    return verified


# -----------------------------------------
#  8. MAIN INFERENCE LOOP
# -----------------------------------------

def run_inference(
    model, tokenizer,
    records: List[Dict],
    strategy: str = "zero-shot",
    verify: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Tuple[List[Dict], Dict, List[Dict]]:
    """
    Run the LLM NER pipeline on all records.

    Returns:
        pred_records:   list of {"id", "entities": [{start, end, label, text}]}
        run_stats:      statistics about parsing success, timing, etc.
        raw_responses:  raw LLM outputs for debugging
    """
    system_prompt = build_system_prompt()
    pred_records = []
    parse_stats = Counter()
    verify_stats = {"total_before": 0, "total_after": 0}
    total_time = 0.0
    raw_responses = []

    for rec in tqdm(records, desc=f"Inference [{strategy}]"):
        text = rec["text"]

        # Build prompt
        if strategy == "few-shot":
            user_prompt = build_user_prompt_few_shot(text)
        else:
            user_prompt = build_user_prompt_zero_shot(text)

        messages = format_chat_messages(system_prompt, user_prompt)

        # Generate
        start_time = time.time()
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        # Parse @@LABEL...## markers
        raw_entities, parse_method = parse_marked_text(response, text)
        parse_stats[parse_method] += 1

        # Align to character offsets
        aligned_entities = align_entities_to_text(text, raw_entities)

        # Optional self-verification
        if verify and aligned_entities:
            before_count = len(aligned_entities)
            aligned_entities = verify_entities(model, tokenizer, text, aligned_entities)
            verify_stats["total_before"] += before_count
            verify_stats["total_after"] += len(aligned_entities)

        elapsed = time.time() - start_time
        total_time += elapsed

        pred_records.append({
            "id":       rec["id"],
            "entities": aligned_entities,
        })

        raw_responses.append({
            "id":            rec["id"],
            "raw_response":  response,
            "parse_method":  parse_method,
            "raw_entities":  raw_entities,
            "aligned_count": len(aligned_entities),
            "inference_time": round(elapsed, 2),
        })

    run_stats = {
        "total_records":    len(records),
        "total_time":       round(total_time, 1),
        "avg_time_per_doc": round(total_time / max(len(records), 1), 2),
        "docs_per_sec":     round(len(records) / max(total_time, 0.01), 1),
        "parse_success":    dict(parse_stats),
        "parse_fail_rate":  round(parse_stats.get("failed", 0) / max(len(records), 1), 4),
        "verification":     verify,
    }

    if verify:
        removed = verify_stats["total_before"] - verify_stats["total_after"]
        run_stats["verify_entities_before"] = verify_stats["total_before"]
        run_stats["verify_entities_after"] = verify_stats["total_after"]
        run_stats["verify_removed"] = removed
        run_stats["verify_removal_rate"] = round(
            removed / max(verify_stats["total_before"], 1), 4
        )

    return pred_records, run_stats, raw_responses


# -----------------------------------------
#  9. EVALUATION & REPORTING
# -----------------------------------------

def evaluate_and_report(
    gold_records: List[Dict],
    pred_records: List[Dict],
    run_stats: Dict,
    model_name: str,
    strategy: str,
    output_dir: str,
    raw_responses: List[Dict],
) -> Dict:
    """Run the full evaluation suite and save all reports."""

    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    verify_tag = "_verified" if run_stats.get("verification", False) else ""
    prefix = f"llm_{model_short}_{strategy.replace('-', '_')}{verify_tag}"

    # -- Build report header --
    report_content = []
    report_content.append(f"LLM Tag-and-Replace Evaluation Report")
    report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Model: {model_name}")
    report_content.append(f"Strategy: {strategy}")
    report_content.append(f"Self-verification: {run_stats.get('verification', False)}")
    report_content.append(f"Test records: {run_stats['total_records']}")
    report_content.append(f"Inference time: {run_stats['total_time']}s "
                          f"({run_stats['avg_time_per_doc']}s/doc, "
                          f"{run_stats['docs_per_sec']} docs/sec)")
    report_content.append(f"Parse stats: {run_stats['parse_success']}")
    report_content.append(f"Parse fail rate: {run_stats['parse_fail_rate']:.1%}")

    if run_stats.get("verification", False):
        report_content.append(f"Verification: {run_stats['verify_entities_before']} -> "
                              f"{run_stats['verify_entities_after']} entities "
                              f"({run_stats['verify_removed']} removed, "
                              f"{run_stats['verify_removal_rate']:.1%} removal rate)")
    report_content.append("")

    all_results = {}

    # Group by temperature
    temp_groups = defaultdict(lambda: {"gold": [], "pred": []})
    temp_groups["Overall"]["gold"] = gold_records
    temp_groups["Overall"]["pred"] = pred_records

    pred_by_id = {r["id"]: r for r in pred_records}
    for g in gold_records:
        temp = g.get("meta_temp", "Unknown")
        temp_groups[temp]["gold"].append(g)
        temp_groups[temp]["pred"].append(pred_by_id[g["id"]])

    preferred_order = ["Overall", "Low", "Medium", "High"]
    for extra_key in temp_groups.keys():
        if extra_key not in preferred_order:
            preferred_order.append(extra_key)

    for temp_label in preferred_order:
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

            v_tag = " +verify" if run_stats.get("verification") else ""
            pipeline_name = (f"LLM {model_name.split('/')[-1]} [{strategy}{v_tag}] - "
                             f"{temp_label} ({matching_mode.upper()} matching)")

            if temp_label == "Overall":
                print(format_tiered_report(tiered_results, pipeline_name))

            report_content.append(format_tiered_report(tiered_results, pipeline_name))
            report_content.append("\n")

    # -- Save all outputs --
    report_path = os.path.join(output_dir, f"{prefix}_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    deep_dive = generate_error_samples(gold_records, pred_records, num_samples=15)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + deep_dive)
    print(f"  Report: {report_path}")

    full_log = generate_full_document_log(gold_records, pred_records)
    log_path = os.path.join(output_dir, f"{prefix}_full_document_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_log)
    print(f"  Document log: {log_path}")

    cat_errors = generate_category_error_report(gold_records, pred_records)
    cat_path = os.path.join(output_dir, f"{prefix}_category_error_analysis.txt")
    with open(cat_path, "w", encoding="utf-8") as f:
        f.write(cat_errors)
    print(f"  Category errors: {cat_path}")

    pred_path = os.path.join(output_dir, f"{prefix}_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2, ensure_ascii=False)
    print(f"  Predictions: {pred_path}")

    results_path = os.path.join(output_dir, f"{prefix}_evaluation_results.json")
    save_results_json(all_results, results_path)
    print(f"  Results JSON: {results_path}")

    raw_path = os.path.join(output_dir, f"{prefix}_raw_responses.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)
    print(f"  Raw responses: {raw_path}")

    return all_results


# -----------------------------------------
#  10. MAIN
# -----------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    # -- Load data --
    print(f"Loading data from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for entry in raw_data:
        entities = []
        for ann in entry.get("label", []):
            label = ann["labels"][0] if ann.get("labels") else None
            if label and label in ALL_LABELS:
                entities.append({
                    "start": ann["start"], "end": ann["end"],
                    "label": label, "text": ann["text"],
                })
        records.append({
            "id":        entry["id"],
            "text":      entry["text"],
            "meta_temp": entry.get("meta_temp", "Unknown"),
            "entities":  sorted(entities, key=lambda e: e["start"]),
        })

    print(f"  Total records: {len(records)}")

    if SPLIT_IDS:
        print(f"  Loading split IDs from: {SPLIT_IDS}")
        with open(SPLIT_IDS, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        test_ids = set(split_info["test_ids"])
        records = [r for r in records if r["id"] in test_ids]
        print(f"  Filtered to test split: {len(records)} records")

    if MAX_DOCS is not None:
        records = records[:MAX_DOCS]
        print(f"  Limited to {len(records)} documents")

    # -- Load model --
    model, tokenizer = load_model(MODEL, quantize_4bit=QUANTIZE)

    # -- Run inference --
    gold_records = records
    pred_records, run_stats, raw_responses = run_inference(
        model, tokenizer, records,
        strategy=STRATEGY,
        verify=VERIFY,
        max_new_tokens=1024,
        temperature=0.0,
    )

    # -- Evaluate --
    evaluate_and_report(
        gold_records, pred_records, run_stats,
        model_name=MODEL, strategy=STRATEGY,
        output_dir=OUTPUT_DIR,
        raw_responses=raw_responses,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
