"""
llm_prompt_anonymize.py
========================
LLM-based DIRECT anonymization via prompt-based rewriting (RQ2a / RQ2b).

Unlike tag-and-replace (which detects PII spans and replaces with [LABEL]
placeholders), this approach instructs the LLM to rewrite the entire text
with PII removed or replaced by natural-sounding alternatives.

This enables a three-way semantic preservation comparison:
  1. Classical tag-and-replace  (spaCy/BERT → [LABEL] placeholders)
  2. LLM tag-and-replace        (LLM → [LABEL] placeholders)
  3. LLM prompt-based rewrite   (LLM → natural language rewrite)  ← THIS

Supported models:
  - meta-llama/Meta-Llama-3-8B-Instruct   (general-purpose baseline)
  - Qwen/Qwen2.5-7B-Instruct             (strongest 7B-class model)
  - VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct  (German-specialized)

The script loops through all configured model/strategy combinations
automatically. Each model is loaded once and reused, then freed from
GPU memory before loading the next model.

How to use:
  1. Edit RUN_MATRIX in USER SETTINGS to enable/disable runs
  2. Press Run in VS Code — everything executes sequentially
  3. Then run semantic_preservation.py and llm_judge_gemini.py on the output

Requirements:
    pip install transformers torch accelerate tqdm bitsandbytes

Author: André Kuhn – Master Thesis (MScIDS, HSLU)
"""

# =====================================================================
#  USER SETTINGS
# =====================================================================
#
#  RUN_MATRIX defines all experiments to run sequentially.
#  Each entry is: (model_id, quantize, strategy)
#
#  The script groups runs by model — loads a model once, runs all its
#  configurations, frees GPU memory, then loads the next model.
#
#  Comment out any rows you want to skip.
# =====================================================================

RUN_MATRIX = [
    # ── Llama-3 8B (general-purpose baseline) ──
    ("meta-llama/Meta-Llama-3-8B-Instruct",                  False, "zero-shot"),
    ("meta-llama/Meta-Llama-3-8B-Instruct",                  False, "few-shot"),

    # ── Qwen2.5 7B (strongest small model) ──
    ("Qwen/Qwen2.5-7B-Instruct",                             False, "zero-shot"),
    ("Qwen/Qwen2.5-7B-Instruct",                             False, "few-shot"),

    # ── SauerkrautLM 8B (German-specialized) ──
    ("VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",     False, "zero-shot"),
    ("VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",     False, "few-shot"),
]

# Paths
INPUT_PATH  = r"C:\thesis\data\label_studio\20260302_Export_Label_Studio_Client_Notes.json"
OUTPUT_DIR  = r"C:\thesis\results\llm_prompt_anonymize"
SPLIT_IDS   = r"C:\thesis\results\bert_finetuned\split_ids.json"
MAX_DOCS    = None      # None for full run, small int for quick test
SEED        = 42


# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import re
import time
import random
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm

from evaluation_utils import ALL_LABELS, load_label_studio_export


# =====================================================================
#  1. PROMPT CONSTRUCTION
# =====================================================================

def build_system_prompt() -> str:
    return (
        "You are an expert text anonymization system for German-language "
        "financial communications from a Swiss banking context.\n\n"
        "Your task: Rewrite the given text so that ALL personally identifiable "
        "information (PII) is removed or replaced with generic alternatives, "
        "while preserving the original meaning, structure, and informational "
        "content as closely as possible.\n\n"
        "PII categories to anonymize:\n"
        "- Person names → replace with generic references (e.g., 'ein Kunde', 'die Geschäftsführerin')\n"
        "- Organization names → replace with generic terms (e.g., 'ein Unternehmen', 'die Firma')\n"
        "- Locations/Addresses → replace with generic references (e.g., 'eine Stadt', 'ein Standort in der Schweiz')\n"
        "- Dates → replace with relative or vague references (e.g., 'kürzlich', 'an einem Datum')\n"
        "- Email addresses → remove or replace with '[E-Mail]'\n"
        "- Phone numbers → remove or replace with '[Telefon]'\n"
        "- IBAN/Account numbers → replace with '[Konto]'\n"
        "- Monetary amounts → replace with generic references (e.g., 'einen Betrag', 'eine Summe')\n"
        "- Job titles → generalize (e.g., 'eine Führungskraft', 'ein Mitarbeiter')\n"
        "- Age references → remove or generalize\n"
        "- Nationality → remove or generalize\n"
        "- Education → generalize (e.g., 'ein Hochschulabschluss')\n\n"
        "RULES:\n"
        "1. Preserve the MEANING and INTENT of the communication.\n"
        "2. Keep the text in German.\n"
        "3. Maintain the same register and tone.\n"
        "4. The rewritten text should be fluent and natural.\n"
        "5. Do NOT add information that was not in the original.\n"
        "6. Do NOT omit non-PII information.\n"
        "7. Output ONLY the rewritten text. No explanations, no markdown.\n"
    )


FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "Betriebsbesichtigung bei der Biofood Produzenten GmbH in Bern am 12.05.2024. "
            "Der Geschäftsführer, Herr Markus Aebischer, führte durch die neuen Anlagen. "
            "Das Unternehmen verzeichnet ein starkes Wachstum im europäischen Markt. "
            "Zur Finanzierung des weiteren Ausbaus wird eine Trade & Export Finance (TEF) "
            "Lösung für Lieferungen nach Deutschland und Österreich geprüft. Das aktuelle "
            "Volumen beträgt ca. CHF 750'000 pro Monat. Herr Aebischer ist 48 Jahre alt "
            "und hat an der ETH Zürich studiert."
        ),
        "output": (
            "Betriebsbesichtigung bei einem Lebensmittelunternehmen an einem Schweizer Standort "
            "an einem Datum im Frühjahr. Ein Mitarbeiter der Geschäftsleitung führte durch die neuen Anlagen. "
            "Das Unternehmen verzeichnet ein starkes Wachstum im Markt. "
            "Zur Finanzierung des weiteren Ausbaus wird eine Trade & Export Finance (TEF) "
            "Lösung für Lieferungen ins Ausland geprüft. Das aktuelle "
            "Volumen beträgt einen erheblichen Betrag pro Monat. Der Verantwortliche hat "
            "einen Hochschulabschluss."
        ),
    },
    {
        "input": (
            "Telefonat mit Frau Sandra Wyss (079 111 22 33), Buchhalterin bei der "
            "Bau-Expert GmbH am 30. April 2024. Sie informierte uns über die Saldierung "
            "des Kontos mit der CH33 0077 7888 9999 0000 1, da die entsprechende "
            "Projektgesellschaft liquidiert wurde. Der Restbetrag von CHF 12'450.50 soll "
            "auf das Hauptkonto überwiesen werden. Die rechtsverbindliche Unterschrift des "
            "Geschäftsführers Peter Schmid liegt uns vor."
        ),
        "output": (
            "Telefonat mit einer Mitarbeiterin eines Bauunternehmens. "
            "Sie informierte uns über die Saldierung "
            "eines Kontos, da die entsprechende "
            "Projektgesellschaft liquidiert wurde. Ein Restbetrag soll "
            "auf das Hauptkonto überwiesen werden. Die rechtsverbindliche Unterschrift "
            "einer Führungskraft liegt uns vor."
        ),
    },
    {
        "input": (
            "Am 28. Februar 2024 fand das Eröffnungsgespräch mit der neu gegründeten "
            "Pharma Spin-Off AG statt. Der designierte CEO, Dr. Martin Fischer, ein "
            "ETH-Absolvent, benötigt diverse Firmenkonten in CHF und EUR. Die notwendigen "
            "KYC-Dokumente, inklusive Handelsregisterauszug aus dem Kanton Basel-Stadt, "
            "wurden übergeben. Kontakt für die technische Anbindung des Cash Managements "
            "ist martin.fischer@pharmaspin.ch."
        ),
        "output": (
            "An einem Datum fand das Eröffnungsgespräch mit einem neu gegründeten "
            "Pharmaunternehmen statt. Der designierte Geschäftsführer, ein "
            "Hochschulabsolvent, benötigt diverse Firmenkonten. Die notwendigen "
            "KYC-Dokumente, inklusive Handelsregisterauszug aus einem Schweizer Kanton, "
            "wurden übergeben. Die Kontaktdaten für die technische Anbindung des Cash Managements "
            "wurden hinterlegt."
        ),
    },
]


def build_user_prompt_zero_shot(text: str) -> str:
    return f"Anonymize the following text by rewriting it:\n\nOriginal: {text}\n\nAnonymized:"


def build_user_prompt_few_shot(text: str) -> str:
    parts = []
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"Example {i}:\nOriginal: {example['input']}\nAnonymized: {example['output']}")
    parts.append(f"Now anonymize the following text by rewriting it:\n\nOriginal: {text}\nAnonymized:")
    return "\n\n".join(parts)


# =====================================================================
#  2. MODEL LOADING
# =====================================================================

def load_model(model_name: str, quantize_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


# =====================================================================
#  3. INFERENCE
# =====================================================================

def generate_response(model, tokenizer, messages, max_new_tokens=1024, temperature=0.0):
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        input_text = f"[INST] {system}\n\n{user} [/INST]"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                      "pad_token_id": tokenizer.pad_token_id}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# =====================================================================
#  4. PII LEAKAGE DETECTION
# =====================================================================

def check_pii_leakage(rewritten_text, gold_entities, case_sensitive=False):
    """
    Check which ground-truth PII strings survive in the rewritten text.
    All PII categories are counted, including MONEY.
    """
    leaked = []
    per_category = defaultdict(lambda: {"total": 0, "leaked": 0})
    check_text = rewritten_text if case_sensitive else rewritten_text.lower()

    for ent in gold_entities:
        label, pii_text = ent["label"], ent["text"]
        per_category[label]["total"] += 1
        search_text = pii_text if case_sensitive else pii_text.lower()

        if len(pii_text) <= 5:
            # Short PII strings (e.g., "Zug", "CEO", "CHF") need word boundary
            # matching to avoid false positives from substring matches
            # (e.g., "Zug" inside "Lesezugriff")
            found = bool(re.search(r'\b' + re.escape(search_text) + r'\b', check_text))
        else:
            found = search_text in check_text

        if found:
            idx = check_text.find(search_text)
            if idx == -1:
                m = re.search(r'\b' + re.escape(search_text) + r'\b', check_text)
                idx = m.start() if m else 0
            ctx_s, ctx_e = max(0, idx - 30), min(len(rewritten_text), idx + len(pii_text) + 30)
            leaked.append({"label": label, "text": pii_text, "context": f"...{rewritten_text[ctx_s:ctx_e]}..."})
            per_category[label]["leaked"] += 1

    per_cat = {l: {**c, "rate": round(c["leaked"] / max(c["total"], 1), 4)} for l, c in sorted(per_category.items())}
    return {"total_pii": len(gold_entities), "leaked_pii": len(leaked),
            "leakage_rate": round(len(leaked) / max(len(gold_entities), 1), 4),
            "leaked_entities": leaked, "per_category": per_cat}


# =====================================================================
#  5. MAIN INFERENCE LOOP
# =====================================================================

def run_prompt_anonymization(model, tokenizer, records, strategy="few-shot",
                              max_new_tokens=1024, temperature=0.0):
    system_prompt = build_system_prompt()
    results, total_time, all_leakage = [], 0.0, []

    for rec in tqdm(records, desc=f"Prompt Anonymization [{strategy}]"):
        user_prompt = build_user_prompt_few_shot(rec["text"]) if strategy == "few-shot" \
            else build_user_prompt_zero_shot(rec["text"])
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        t0 = time.time()
        rewritten = generate_response(model, tokenizer, messages, max_new_tokens, temperature)
        elapsed = time.time() - t0
        total_time += elapsed

        rewritten = re.sub(r"```\w*\n?", "", rewritten)
        rewritten = re.sub(r"^(Anonymized|Anonymisiert|Output|Rewritten):\s*", "", rewritten, flags=re.IGNORECASE).strip()

        leakage = check_pii_leakage(rewritten, rec["entities"])
        all_leakage.append(leakage)

        results.append({"id": rec["id"], "original_text": rec["text"], "rewritten_text": rewritten,
                         "meta_temp": rec.get("meta_temp", "Unknown"), "gold_entities": rec["entities"],
                         "leakage": leakage, "inference_time": round(elapsed, 2),
                         "original_length": len(rec["text"]), "rewritten_length": len(rewritten),
                         "length_ratio": round(len(rewritten) / max(len(rec["text"]), 1), 4)})

    run_stats = {"total_records": len(records), "total_time": round(total_time, 1),
                 "avg_time_per_doc": round(total_time / max(len(records), 1), 2),
                 "docs_per_sec": round(len(records) / max(total_time, 0.01), 1),
                 "avg_leakage_rate": round(sum(l["leakage_rate"] for l in all_leakage) / max(len(all_leakage), 1), 4),
                 "overall_leaked": sum(l["leaked_pii"] for l in all_leakage),
                 "overall_total_pii": sum(l["total_pii"] for l in all_leakage),
                 "avg_length_ratio": round(sum(r["length_ratio"] for r in results) / max(len(results), 1), 4)}
    return results, run_stats


# =====================================================================
#  6. REPORTING
# =====================================================================

def format_report(results, run_stats, model_name, strategy):
    lines = ["=" * 78, "  LLM PROMPT-BASED ANONYMIZATION REPORT",
             f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"  Model: {model_name}",
             f"  Strategy: {strategy}", "=" * 78,
             f"\n  Documents: {run_stats['total_records']}",
             f"  Total time: {run_stats['total_time']}s ({run_stats['avg_time_per_doc']}s/doc)",
             f"  Avg length ratio: {run_stats['avg_length_ratio']:.2f}"]

    ol, ot = run_stats["overall_leaked"], run_stats["overall_total_pii"]
    lines += [f"\n{'#'*78}", "  PII LEAKAGE ANALYSIS", f"{'#'*78}",
              f"\n  Overall: {ol}/{ot} leaked ({ol/max(ot,1):.1%})"]

    cat_totals = defaultdict(lambda: {"total": 0, "leaked": 0})
    for r in results:
        for l, c in r["leakage"]["per_category"].items():
            cat_totals[l]["total"] += c["total"]; cat_totals[l]["leaked"] += c["leaked"]
    lines += [f"\n  {'Category':<12} {'Total':>8} {'Leaked':>8} {'Rate':>10}", f"  {'-'*42}"]
    for l in sorted(cat_totals):
        t, lk = cat_totals[l]["total"], cat_totals[l]["leaked"]
        lines.append(f"  {l:<12} {t:>8} {lk:>8} {lk/max(t,1):>10.1%}")

    lines += [f"\n{'#'*78}", "  LEAKAGE BY COMPLEXITY", f"{'#'*78}"]
    by_c = defaultdict(list)
    for r in results: by_c[r["meta_temp"]].append(r)
    for lv in ["Low", "Medium", "High"]:
        g = by_c.get(lv, [])
        if g:
            lk = sum(r["leakage"]["leaked_pii"] for r in g)
            tt = sum(r["leakage"]["total_pii"] for r in g)
            lines.append(f"\n  {lv}: {len(g)} docs | Leakage: {lk}/{tt} ({lk/max(tt,1):.1%})")

    lines += [f"\n{'#'*78}", "  SAMPLE LEAKED PII (first 20)", f"{'#'*78}"]
    cnt = 0
    for r in results:
        for lk in r["leakage"]["leaked_entities"]:
            if cnt >= 20: break
            lines.append(f"  [{lk['label']:<8}] \"{lk['text']}\"  →  {lk['context']}"); cnt += 1
        if cnt >= 20: break

    lines += [f"\n{'#'*78}", "  QUALITATIVE SAMPLES (10 examples)", f"{'#'*78}"]
    srt = sorted(results, key=lambda r: r["leakage"]["leakage_rate"], reverse=True)
    samples = srt[:5]
    clean = [r for r in results if r["leakage"]["leaked_pii"] == 0]
    if clean: random.shuffle(clean); samples.extend(clean[:5])
    for r in samples:
        lines += [f"\n  {'─'*72}",
                  f"  Doc {r['id']} | {r['meta_temp']} | Leakage: {r['leakage']['leaked_pii']}/{r['leakage']['total_pii']}",
                  f"  ORIGINAL:  {r['original_text'][:300]}{'...' if len(r['original_text'])>300 else ''}",
                  f"  REWRITTEN: {r['rewritten_text'][:300]}{'...' if len(r['rewritten_text'])>300 else ''}"]
    return "\n".join(lines)


# =====================================================================
#  7. MAIN
# =====================================================================

def save_run_outputs(results, run_stats, model_name, strategy, output_dir):
    """Save all outputs for a single run."""
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    prefix = f"prompt_anon_{model_short}_{strategy.replace('-', '_')}"

    report = format_report(results, run_stats, model_name, strategy)
    print(report)

    with open(os.path.join(output_dir, f"{prefix}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(output_dir, f"{prefix}_full_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Full side-by-side comparison
    side_by_side = []
    side_by_side.append("=" * 90)
    side_by_side.append("  FULL SIDE-BY-SIDE COMPARISON: Original → Rewritten (ALL documents)")
    side_by_side.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    side_by_side.append(f"  Model: {model_name} | Strategy: {strategy}")
    side_by_side.append(f"  Documents: {len(results)}")
    side_by_side.append("=" * 90)

    for r in results:
        side_by_side.append(f"\n{'─' * 90}")
        side_by_side.append(f"  Doc ID: {r['id']} | Complexity: {r['meta_temp']} | "
                            f"Leakage: {r['leakage']['leaked_pii']}/{r['leakage']['total_pii']} "
                            f"({r['leakage']['leakage_rate']:.0%}) | "
                            f"Length: {r['original_length']} → {r['rewritten_length']} "
                            f"(ratio: {r['length_ratio']:.2f})")
        side_by_side.append(f"{'─' * 90}")
        side_by_side.append(f"\n  ORIGINAL:\n  {r['original_text']}")
        side_by_side.append(f"\n  REWRITTEN:\n  {r['rewritten_text']}")
        if r['leakage']['leaked_entities']:
            side_by_side.append(f"\n  LEAKED PII:")
            for lk in r['leakage']['leaked_entities']:
                side_by_side.append(f"    [{lk['label']:<8}] \"{lk['text']}\"")

    sbs_path = os.path.join(output_dir, f"{prefix}_side_by_side.txt")
    with open(sbs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(side_by_side))
    print(f"  Side-by-side (all docs): {sbs_path}")

    preds = [{"id": r["id"], "rewritten_text": r["rewritten_text"],
              "leakage": {"total_pii": r["leakage"]["total_pii"], "leaked_pii": r["leakage"]["leaked_pii"],
                          "leakage_rate": r["leakage"]["leakage_rate"]}} for r in results]
    with open(os.path.join(output_dir, f"{prefix}_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, f"{prefix}_run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(run_stats, f, indent=2, ensure_ascii=False)


def main():
    import gc
    from itertools import groupby

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED); torch.manual_seed(SEED)

    # -- Load data (once, shared across all runs) --
    print(f"Loading data from: {INPUT_PATH}")
    gold_records = load_label_studio_export(INPUT_PATH)
    print(f"  Total records: {len(gold_records)}")

    if SPLIT_IDS:
        with open(SPLIT_IDS, "r", encoding="utf-8") as f: split_info = json.load(f)
        gold_records = [r for r in gold_records if r["id"] in set(split_info["test_ids"])]
        print(f"  Filtered to test split: {len(gold_records)} records")

    if MAX_DOCS is not None:
        gold_records = gold_records[:MAX_DOCS]
        print(f"  Limited to {len(gold_records)} documents")

    # -- Group runs by model (load each model once) --
    total_runs = len(RUN_MATRIX)
    completed = 0

    grouped = []
    for model_id, runs in groupby(RUN_MATRIX, key=lambda x: x[0]):
        grouped.append((model_id, list(runs)))

    for model_id, runs in grouped:
        quantize = runs[0][1]

        print(f"\n{'#' * 70}")
        print(f"  LOADING MODEL: {model_id}")
        print(f"  Runs to execute: {len(runs)}")
        print(f"{'#' * 70}")

        model, tokenizer = load_model(model_id, quantize_4bit=quantize)

        for _, _, strategy in runs:
            completed += 1
            print(f"\n{'=' * 60}")
            print(f"  RUN {completed}/{total_runs}: {model_id.split('/')[-1]} [{strategy}]")
            print(f"{'=' * 60}")

            # Check if output already exists (skip if so)
            model_short = model_id.split("/")[-1].lower().replace("-", "_")
            prefix = f"prompt_anon_{model_short}_{strategy.replace('-', '_')}"
            pred_path = os.path.join(OUTPUT_DIR, f"{prefix}_predictions.json")

            if os.path.exists(pred_path):
                print(f"  → SKIPPING (already exists: {pred_path})")
                continue

            results, run_stats = run_prompt_anonymization(
                model, tokenizer, gold_records, strategy=strategy,
                max_new_tokens=1024, temperature=0.0)

            save_run_outputs(results, run_stats, model_id, strategy, OUTPUT_DIR)

        # Free GPU memory before loading next model
        print(f"\n  Freeing GPU memory for {model_id.split('/')[-1]}...")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE! {completed} runs completed. Outputs in: {OUTPUT_DIR}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
