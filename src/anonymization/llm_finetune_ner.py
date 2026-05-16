"""
llm_finetune_ner.py
====================
Fine-tune LLMs for PII NER using QLoRA on the thesis dataset.

Follows the methodology of:
  - Dorémus et al. (JMIR AI 2025): QLoRA fine-tuning of generative LLMs
    for de-identification of clinical records
  - Staab et al. (ICLR 2025, Appendix I): distillation of anonymization
    capabilities into smaller models via LoRA

Supported models:
  - meta-llama/Meta-Llama-3-8B-Instruct   (general-purpose baseline)
  - Qwen/Qwen2.5-7B-Instruct             (strongest 7B-class model)
  - VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct  (German-specialized)

The fine-tuned model uses the same @@LABEL...## marking format as
llm_tag_and_replace.py, so the evaluation pipeline is fully compatible.

Pipeline:
  1. Load annotated data → convert to instruction/response pairs
  2. Fine-tune with QLoRA (4-bit)
  3. Evaluate on test set using evaluation_utils.py
  4. Export predictions in the same format as other pipelines

Output naming is automatic: each model gets a subdirectory under OUTPUT_BASE
(e.g., llm_finetuned_meta_llama_3_8b_instruct/) with predictions named
to match the downstream evaluation scripts.

Hardware: RTX 4090 (24GB VRAM) — sufficient for QLoRA on 7–8B models.

Usage:
  1. Uncomment ONE model block in USER SETTINGS below.
  2. Run the script.
  3. Repeat for each model (results are auto-named, no overwrites).

Requirements:
    pip install transformers torch datasets peft bitsandbytes accelerate trl tqdm seqeval

"""

# =====================================================================
#  USER SETTINGS — Uncomment ONE model block
# =====================================================================
#
#  Available models:
#    "meta-llama/Meta-Llama-3-8B-Instruct"   (16 GB, float16)  — general-purpose baseline
#    "Qwen/Qwen2.5-7B-Instruct"             (15 GB, float16)  — strongest 7B-class model
#    "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct"  (16 GB, float16)  — German-specialized
#
# =====================================================================

# --- Llama-3 8B ---
BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Qwen2.5 7B (strongest small model) ---
# BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

# --- SauerkrautLM 8B (German-specialized) ---
# BASE_MODEL  = "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct"

# Set to True to skip training and print a comparison of all completed models
RUN_SUMMARY = False

# All model IDs (used by the summary function)
ALL_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",
]

import os

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "label_studio", "20260302_Export_Label_Studio_Client_Notes.json")
SPLIT_IDS   = os.path.join(BASE_DIR, "results", "bert_finetuned", "split_ids.json")

# Output directory: auto-created per model under this base path
OUTPUT_BASE = os.path.join(BASE_DIR, "results", "llm_finetuned")
# The actual OUTPUT_DIR is computed at runtime as:
#   OUTPUT_BASE / llm_finetuned_{model_short}/
# This ensures each model's checkpoints and results are separate.

# Training hyperparameters (following Dorémus et al.)
EPOCHS          = 5
BATCH_SIZE      = 2         # per-device batch size (reduced for 4090 24GB)
GRAD_ACCUM      = 12        # effective batch = 2 * 12 = 24 (same as Dorémus)
LR              = 5e-5      # Dorémus: 5e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
MAX_SEQ_LEN     = 1024      # max tokens per example

# QLoRA config (following Dorémus et al.)
LORA_RANK       = 32        # Dorémus: 32
LORA_ALPHA      = 64        # Dorémus: 64
LORA_DROPOUT    = 0.1       # Dorémus: 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Evaluation
EVAL_EVERY_EPOCH = True     # evaluate on dev set after each epoch
MAX_NEW_TOKENS   = 1024     # for inference
SEED             = 42

# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os

import re
import time
import random
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import torch
import numpy as np
from tqdm import tqdm

from evaluation_utils import (
    ALL_LABELS, load_label_studio_export,
    evaluate_tiered, format_tiered_report, save_results_json,
    generate_error_samples, generate_full_document_log,
    generate_category_error_report,
)

# =====================================================================
#  1. DATA FORMATTING: Annotated text → @@LABEL...## format
# =====================================================================

def build_label_description_block() -> str:
    """Same label descriptions as llm_tag_and_replace.py."""
    descriptions = {
        "PER":     "Person names (first, last, or full names)",
        "ORG":     "Organization / company names",
        "LOC":     "Locations, cities, addresses, countries",
        "DATE":    "Dates (absolute or relative)",
        "IBAN":    "IBAN or bank account numbers",
        "EMAIL":   "Email addresses",
        "PHONE":   "Phone numbers",
        "MONEY":   "Monetary amounts with currency",
        "JOB":     "Job titles / professions",
        "AGE":     "Age references",
        "NATION":  "Nationality / citizenship references",
        "EDU":     "Education / degrees / institutions",
    }
    return "\n".join(f"  - {label}: {desc}" for label, desc in sorted(descriptions.items()))

SYSTEM_PROMPT = (
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
    "3. Do NOT change, rephrase, or omit any part of the text.\n"
    "4. If no entities exist, return the text unchanged.\n"
    "5. Output ONLY the marked text. No explanations.\n"
)

def text_with_entity_markers(text: str, entities: List[Dict]) -> str:
    """
    Convert character-offset entity annotations to @@LABEL...## marked text.
    This produces the expected output format for training.
    """
    sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)
    result = text
    for ent in sorted_ents:
        label = ent["label"]
        start = ent["start"]
        end = ent["end"]
        result = result[:start] + f"@@{label} " + result[start:end] + "##" + result[end:]
    return result

def build_training_examples(records: List[Dict]) -> List[Dict]:
    """
    Convert annotated records into instruction/response pairs for SFT.

    Each example becomes:
        system: <system prompt>
        user: "Mark all PII entities in this text:\n\nInput: <original text>\nOutput:"
        assistant: "<text with @@LABEL...## markers>"
    """
    examples = []
    for rec in records:
        original = rec["text"]
        marked = text_with_entity_markers(original, rec["entities"])

        examples.append({
            "id": rec["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Mark all PII entities in this text:\n\nInput: {original}\nOutput:"},
                {"role": "assistant", "content": marked},
            ],
        })

    return examples

# =====================================================================
#  2. TRAINING WITH QLoRA
# =====================================================================

def setup_model_for_training(model_name: str):
    """
    Load the base model with 4-bit quantization and attach LoRA adapters.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"\n  Loading base model: {model_name}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Attach LoRA adapters
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer

def format_chat_for_training(messages: List[Dict], tokenizer) -> str:
    """Apply the chat template to produce the full training string."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback for models without a chat template
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"[INST] <<SYS>>\n{msg['content']}\n<</SYS>>\n\n")
            elif msg["role"] == "user":
                parts.append(f"{msg['content']} [/INST] ")
            elif msg["role"] == "assistant":
                parts.append(f"{msg['content']}")
        return "".join(parts)

def tokenize_examples(examples: List[Dict], tokenizer, max_length: int) -> List[Dict]:
    """
    Tokenize training examples with labels for causal LM training.
    Labels are set to -100 for system+user tokens (we only train on
    the assistant's response).
    """
    tokenized = []

    for ex in tqdm(examples, desc="Tokenizing"):
        messages = ex["messages"]

        # Full conversation string
        full_text = format_chat_for_training(messages, tokenizer)

        # Tokenize full conversation
        full_enc = tokenizer(full_text, truncation=True, max_length=max_length,
                             padding=False, return_tensors=None)
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Find where the assistant response starts by tokenizing everything
        # up to (but not including) the assistant message
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        # Add generation prompt to get the exact prefix
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = format_chat_for_training(prompt_messages, tokenizer)

        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length,
                               padding=False, return_tensors=None)
        prompt_len = len(prompt_enc["input_ids"])

        # Labels: -100 for prompt tokens, actual IDs for response tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Ensure same length
        labels = labels[:len(input_ids)]

        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return tokenized

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_examples):
        self.examples = tokenized_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }

def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(item["input_ids"]) for item in batch)

    padded = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        padded["input_ids"].append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        padded["attention_mask"].append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        padded["labels"].append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {k: torch.stack(v) for k, v in padded.items()}

# =====================================================================
#  3. INFERENCE (same parsing as llm_tag_and_replace.py)
# =====================================================================

def generate_response(model, tokenizer, messages, max_new_tokens=1024):
    """Generate a response from the fine-tuned model."""
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
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

def parse_entity_markers(original_text: str, marked_text: str) -> List[Dict]:
    """
    Parse @@LABEL text## markers from the model output back into
    character-offset entity spans (same logic as llm_tag_and_replace.py).
    """
    pattern = r"@@(\w+)\s(.*?)##"
    entities = []

    # Remove markers to reconstruct clean text, tracking offsets
    clean_text = marked_text
    offset_shift = 0

    for match in re.finditer(pattern, marked_text):
        label = match.group(1)
        entity_text = match.group(2)

        if label not in ALL_LABELS:
            continue

        # Find this entity in the original text
        search_start = 0
        found = False
        while search_start < len(original_text):
            idx = original_text.find(entity_text, search_start)
            if idx == -1:
                break

            # Check if this span is already claimed
            overlap = False
            for existing in entities:
                if idx < existing["end"] and idx + len(entity_text) > existing["start"]:
                    overlap = True
                    break

            if not overlap:
                entities.append({
                    "start": idx,
                    "end": idx + len(entity_text),
                    "label": label,
                    "text": entity_text,
                })
                found = True
                break

            search_start = idx + 1

    return sorted(entities, key=lambda e: e["start"])

def run_inference(model, tokenizer, records: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Run inference on a set of records and return predictions.
    Returns (predictions, total_time).
    """
    predictions = []
    total_time = 0.0

    for rec in tqdm(records, desc="Inference"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Mark all PII entities in this text:\n\nInput: {rec['text']}\nOutput:"},
        ]

        t0 = time.time()
        response = generate_response(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = time.time() - t0
        total_time += elapsed

        entities = parse_entity_markers(rec["text"], response)
        predictions.append({
            "id": rec["id"],
            "entities": entities,
        })

    return predictions, total_time

# =====================================================================
#  4. MAIN
# =====================================================================

def main():
    # ── Compute model-specific output directory ──
    model_short = BASE_MODEL.split("/")[-1].lower().replace("-", "_")
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"llm_finetuned_{model_short}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prefix for all output files (matches downstream evaluation scripts)
    file_prefix = f"llm_finetuned_{model_short}"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ──
    print(f"\nLoading data from: {INPUT_PATH}")
    all_records = load_label_studio_export(INPUT_PATH)
    print(f"  Total records: {len(all_records)}")

    # ── Load split IDs ──
    print(f"  Loading split IDs from: {SPLIT_IDS}")
    with open(SPLIT_IDS, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_ids = set(split_info["train_ids"])
    dev_ids = set(split_info["dev_ids"])
    test_ids = set(split_info["test_ids"])

    records_by_id = {r["id"]: r for r in all_records}
    train_records = [records_by_id[i] for i in split_info["train_ids"] if i in records_by_id]
    dev_records = [records_by_id[i] for i in split_info["dev_ids"] if i in records_by_id]
    test_records = [records_by_id[i] for i in split_info["test_ids"] if i in records_by_id]

    print(f"  Train: {len(train_records)} | Dev: {len(dev_records)} | Test: {len(test_records)}")

    # ── Build training examples ──
    print(f"\n  Building instruction/response pairs...")
    train_examples = build_training_examples(train_records)
    dev_examples = build_training_examples(dev_records)
    print(f"  Train examples: {len(train_examples)} | Dev examples: {len(dev_examples)}")

    # ── Load model with QLoRA ──
    model, tokenizer = setup_model_for_training(BASE_MODEL)

    # ── Tokenize ──
    print(f"\n  Tokenizing training data...")
    train_tokenized = tokenize_examples(train_examples, tokenizer, MAX_SEQ_LEN)
    print(f"  Tokenized: {len(train_tokenized)} examples")

    # Filter out examples that are too long (response got truncated)
    orig_len = len(train_tokenized)
    train_tokenized = [t for t in train_tokenized if len(t["input_ids"]) < MAX_SEQ_LEN]
    if len(train_tokenized) < orig_len:
        print(f"  Filtered: {orig_len - len(train_tokenized)} examples exceeded max_seq_len")

    # ── DataLoader ──
    train_dataset = NERDataset(train_tokenized)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )

    # ── Optimizer & Scheduler ──
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {BASE_MODEL}")
    print(f"  {EPOCHS} epochs, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Total optimization steps: {total_steps}, warmup: {warmup_steps}")
    print(f"  QLoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    # ── Training loop ──
    training_log = []
    best_dev_f1 = 0.0
    best_epoch = -1
    start_time = time.time()

    # Continuous log file — written after every epoch so nothing is lost on crash
    live_log_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_live_log.json")

    def _save_live_log(extra_info=None):
        """Write current training state to disk immediately."""
        elapsed = time.time() - start_time
        log_data = {
            "base_model": BASE_MODEL,
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "epochs_planned": EPOCHS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "lr": LR, "seed": SEED,
            "best_epoch": best_epoch, "best_dev_f1": best_dev_f1,
            "elapsed_seconds": round(elapsed, 1),
            "status": "running",
            "log": training_log,
        }
        if extra_info:
            log_data.update(extra_info)
        with open(live_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Write initial log
    _save_live_log({"status": "training_started"})

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += outputs.loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\n  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # ── Evaluate on dev set ──
        dev_f1 = 0.0
        if EVAL_EVERY_EPOCH and dev_records:
            print(f"  Evaluating on dev set ({len(dev_records)} docs)...")
            model.eval()
            dev_preds, dev_time = run_inference(model, tokenizer, dev_records)
            model.train()

            # Quick F1 calculation
            dev_gold = [{"id": r["id"], "text": r["text"], "meta_temp": r.get("meta_temp", "Unknown"),
                         "entities": r["entities"]} for r in dev_records]
            dev_results = evaluate_tiered(dev_gold, dev_preds, matching="strict")
            dev_f1 = dev_results["All Categories"]["overall"]["f1"]
            print(f"  Dev F1 (strict, all categories): {dev_f1:.4f}")

        epoch_info = {
            "epoch": epoch + 1,
            "avg_loss": round(avg_loss, 4),
            "dev_f1": round(dev_f1, 4),
        }
        training_log.append(epoch_info)

        # Save best checkpoint
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch + 1
            checkpoint_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  → New best! Saved checkpoint (dev_f1={dev_f1:.4f})")
        else:
            print(f"  → No improvement (best: epoch {best_epoch}, f1={best_dev_f1:.4f})")

        # Write live log to disk after every epoch (crash-safe)
        _save_live_log({"status": f"completed_epoch_{epoch+1}"})

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.1f}s")
    print(f"  Best epoch: {best_epoch} (dev_f1={best_dev_f1:.4f})")

    _save_live_log({"status": "training_complete", "training_time": round(training_time, 1)})

    # ── Load best checkpoint for test evaluation ──
    print(f"\n  Loading best checkpoint for test evaluation...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # Reload base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    # Load LoRA weights
    checkpoint_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    # ── Test set evaluation ──
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION ON TEST SET ({len(test_records)} documents)")
    print(f"{'=' * 60}")

    test_preds, test_time = run_inference(model, tokenizer, test_records)
    print(f"  Inference time: {test_time:.1f}s ({test_time / len(test_records):.2f}s/doc)")

    # Prepare gold records
    gold_records = [{"id": r["id"], "text": r["text"], "meta_temp": r.get("meta_temp", "Unknown"),
                     "entities": r["entities"]} for r in test_records]

    # Run tiered evaluation
    report_content = []
    report_content.append(f"LLM Fine-Tuned (QLoRA) NER Evaluation Report")
    report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Base model: {BASE_MODEL}")
    report_content.append(f"QLoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    report_content.append(f"Best epoch: {best_epoch}/{EPOCHS}")
    report_content.append(f"Training time: {training_time:.1f}s")
    report_content.append(f"Test records: {len(test_records)}")
    report_content.append(f"Inference time: {test_time:.1f}s\n")

    all_results = {}
    temp_groups = defaultdict(lambda: {"gold": [], "pred": []})
    temp_groups["Overall"]["gold"] = gold_records
    temp_groups["Overall"]["pred"] = test_preds

    pred_by_id = {r["id"]: r for r in test_preds}
    for g in gold_records:
        t = g.get("meta_temp", "Unknown")
        temp_groups[t]["gold"].append(g)
        temp_groups[t]["pred"].append(pred_by_id[g["id"]])

    for temp_label in ["Overall", "Low", "Medium", "High"]:
        if temp_label not in temp_groups or not temp_groups[temp_label]["gold"]:
            continue
        gg, gp = temp_groups[temp_label]["gold"], temp_groups[temp_label]["pred"]
        all_results[temp_label] = {}
        report_content.append(f"\n{'=' * 70}")
        report_content.append(f"  SUBSET: {temp_label.upper()} ({len(gg)} records)")
        report_content.append(f"{'=' * 70}\n")

        for mm in ("strict", "relaxed"):
            tr = evaluate_tiered(gg, gp, matching=mm)
            all_results[temp_label][mm] = tr
            if temp_label == "Overall":
                print(format_tiered_report(tr, f"LLM Fine-Tuned {BASE_MODEL.split('/')[-1]} ({mm.upper()})"))
            report_content.append(format_tiered_report(tr, f"LLM Fine-Tuned {BASE_MODEL.split('/')[-1]} - {temp_label} ({mm.upper()})"))
            report_content.append("\n")

    # ── Save outputs with model-specific filenames ──
    save_results_json(all_results, os.path.join(OUTPUT_DIR, f"{file_prefix}_evaluation_results.json"))

    with open(os.path.join(OUTPUT_DIR, f"{file_prefix}_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(test_preds, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    # Error analysis
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + generate_error_samples(gold_records, test_preds, 15))

    with open(os.path.join(OUTPUT_DIR, f"{file_prefix}_full_document_log.txt"), "w", encoding="utf-8") as f:
        f.write(generate_full_document_log(gold_records, test_preds))

    with open(os.path.join(OUTPUT_DIR, f"{file_prefix}_category_error_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(generate_category_error_report(gold_records, test_preds))

    with open(os.path.join(OUTPUT_DIR, f"{file_prefix}_training_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "lr": LR, "seed": SEED,
            "best_epoch": best_epoch, "best_dev_f1": best_dev_f1,
            "training_time": training_time, "test_inference_time": test_time,
            "log": training_log,
        }, f, indent=2)

    # Save adapter for later use
    adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n  Adapter saved: {adapter_dir}")

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    _save_live_log({
        "status": "done",
        "training_time": round(training_time, 1),
        "test_inference_time": round(test_time, 1),
    })

def print_all_models_summary():
    """
    Read training logs from all completed models and print a side-by-side
    comparison table. Set RUN_SUMMARY = True and press Run.
    """
    print(f"\n{'=' * 90}")
    print(f"  ALL MODELS SUMMARY — QLoRA Fine-Tuned LLMs for PII NER")
    print(f"{'=' * 90}\n")

    summaries = []

    for full_name in ALL_MODELS:
        model_short = full_name.split("/")[-1].lower().replace("-", "_")
        short_name = full_name.split("/")[-1]
        model_dir = os.path.join(OUTPUT_BASE, f"llm_finetuned_{model_short}")

        # Try final training log first, then live log
        log_path = os.path.join(model_dir, f"llm_finetuned_{model_short}_training_log.json")
        live_log_path = os.path.join(model_dir, f"llm_finetuned_{model_short}_live_log.json")
        results_path = os.path.join(model_dir, f"llm_finetuned_{model_short}_evaluation_results.json")

        if not os.path.exists(log_path) and not os.path.exists(live_log_path):
            print(f"  {short_name}")
            print(f"    → NOT STARTED (no log found)")
            print()
            continue

        # Prefer final log, fall back to live log
        use_path = log_path if os.path.exists(log_path) else live_log_path
        with open(use_path, "r", encoding="utf-8") as f:
            log = json.load(f)

        status = log.get("status", "unknown")
        if "CRASHED" in str(status):
            print(f"  {short_name}")
            print(f"    → {status}")
            print(f"    → Last completed: {len(log.get('log', []))} epoch(s)")
            print()
            continue

        # Load test results if available
        test_results = {}
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                test_results = json.load(f)

        training_time = log.get("training_time", 0)
        test_time = log.get("test_inference_time", 0)
        best_epoch = log.get("best_epoch", "?")
        best_dev_f1 = log.get("best_dev_f1", 0)
        epochs_log = log.get("log", [])

        test_overall_f1 = 0
        test_tier1_f1 = 0
        test_tier2_f1 = 0
        test_tier3_f1 = 0
        if "Overall" in test_results and "strict" in test_results["Overall"]:
            strict = test_results["Overall"]["strict"]
            test_overall_f1 = strict.get("All Categories", {}).get("overall", {}).get("f1", 0)
            test_tier1_f1 = strict.get("Tier 1 – Direct NER", {}).get("overall", {}).get("f1", 0)
            test_tier2_f1 = strict.get("Tier 2 – Structured (Regex)", {}).get("overall", {}).get("f1", 0)
            test_tier3_f1 = strict.get("Tier 3 – Quasi-Identifiers", {}).get("overall", {}).get("f1", 0)

        summaries.append({
            "short": short_name,
            "full": full_name,
            "status": status,
            "best_epoch": best_epoch,
            "best_dev_f1": best_dev_f1,
            "test_overall": test_overall_f1,
            "test_tier1": test_tier1_f1,
            "test_tier2": test_tier2_f1,
            "test_tier3": test_tier3_f1,
            "train_time": training_time,
            "test_time": test_time,
            "epochs_log": epochs_log,
        })

    if not summaries:
        print("  No completed models found. Run training first.")
        return

    print(f"  {'Model':<15} {'Status':>12} {'Best':>6} {'Dev F1':>8} {'Test F1':>9} {'Tier1':>7} {'Tier2':>7} {'Tier3':>7} {'Train':>8}")
    print(f"  {'':15} {'':>12} {'epoch':>6} {'':>8} {'(all)':>9} {'(NER)':>7} {'(Regex)':>7} {'(Quasi)':>7} {'time':>8}")
    print(f"  {'-' * 92}")

    for s in summaries:
        train_str = f"{s['train_time']/60:.0f}m" if s['train_time'] > 0 else "?"
        print(
            f"  {s['short']:<15} "
            f"{s['status']:>12} "
            f"{s['best_epoch']:>6} "
            f"{s['best_dev_f1']:>8.4f} "
            f"{s['test_overall']:>9.4f} "
            f"{s['test_tier1']:>7.4f} "
            f"{s['test_tier2']:>7.4f} "
            f"{s['test_tier3']:>7.4f} "
            f"{train_str:>8}"
        )

    print(f"  {'-' * 92}")

    # Training loss curves
    print(f"\n  Training Loss per Epoch:")
    print(f"  {'Model':<15}", end="")
    max_epochs = max(len(s["epochs_log"]) for s in summaries) if summaries else 0
    for e in range(max_epochs):
        print(f"  {'Ep'+str(e+1):>8}", end="")
    print()
    for s in summaries:
        print(f"  {s['short']:<15}", end="")
        for e in s["epochs_log"]:
            print(f"  {e['avg_loss']:>8.4f}", end="")
        print()

    # Save summary
    summary_path = os.path.join(OUTPUT_BASE, "all_models_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")
    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    if RUN_SUMMARY:
        print_all_models_summary()
    else:
        try:
            main()
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  GPU OUT OF MEMORY: {e}")
            print(f"  Try reducing BATCH_SIZE to 2 or MAX_SEQ_LEN to 768")
            if torch.cuda.is_available():
                print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                print(f"  GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.1f} GB")
            # Write crash info to live log
            model_short = BASE_MODEL.split("/")[-1].lower().replace("-", "_")
            crash_log = os.path.join(OUTPUT_BASE, f"llm_finetuned_{model_short}", f"llm_finetuned_{model_short}_live_log.json")
            if os.path.exists(crash_log):
                with open(crash_log, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                log_data["status"] = f"CRASHED: OOM — {e}"
                with open(crash_log, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            # Write crash info to live log
            model_short = BASE_MODEL.split("/")[-1].lower().replace("-", "_")
            crash_log = os.path.join(OUTPUT_BASE, f"llm_finetuned_{model_short}", f"llm_finetuned_{model_short}_live_log.json")
            if os.path.exists(crash_log):
                with open(crash_log, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                log_data["status"] = f"CRASHED: {type(e).__name__} — {e}"
                with open(crash_log, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)