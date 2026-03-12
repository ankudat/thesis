"""
bert_finetune_ner.py
=====================
Fine-tune a German BERT model for NER on the thesis dataset (12 PII categories).

This script:
  1. Loads the Label Studio JSON export
  2. Converts character-offset annotations to IOB2 token-level labels
  3. Splits data into train / dev / test (stratified by temperature)
  4. Fine-tunes a BERT token classification model
  5. Evaluates on the test set using the shared evaluation_utils.py
  6. Exports predictions in the same format as classical_baseline.py

How to use:
  1. Adjust USER SETTINGS below
  2. Press Run in VS Code

Requirements:
    pip install transformers torch datasets seqeval scikit-learn tqdm
    (evaluation_utils.py must be importable)

Author: André Kuhn – Master Thesis (MScIDS, HSLU)
"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

# Base model for fine-tuning
# Recommended models for German NER:
#   "bert-base-german-cased"    (dbmdz German BERT, good default)
#   "deepset/gbert-base"        (German BERT by deepset)
#   "deepset/gbert-large"       (larger, needs more VRAM)
#   "xlm-roberta-base"          (multilingual, robust fallback)
BASE_MODEL  = "bert-base-german-cased"

# Paths
INPUT_PATH  = r"C:\thesis\data\label_studio\20260302_Export_Label_Studio_Client_Notes.json"
OUTPUT_DIR  = r"C:\thesis\results\bert_finetuned"

# Training hyperparameters
EPOCHS          = 5
BATCH_SIZE      = 16
LR              = 3e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
SEED            = 42

# Data split ratios (must sum to 1.0)
TRAIN_RATIO     = 0.50
DEV_RATIO       = 0.25
# TEST_RATIO is computed automatically as 1 - TRAIN_RATIO - DEV_RATIO

# Save the full model (not just checkpoint)?
SAVE_MODEL      = False


# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
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


# ─────────────────────────────────────────────
#  1. CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────

# All 12 entity labels from the thesis annotation schema
ENTITY_LABELS = sorted(ALL_LABELS)  # deterministic order

# Build IOB2 label list:  O, B-AGE, I-AGE, B-DATE, I-DATE, ...
IOB2_LABELS = ["O"]
for label in ENTITY_LABELS:
    IOB2_LABELS.append(f"B-{label}")
    IOB2_LABELS.append(f"I-{label}")

LABEL_TO_ID = {label: idx for idx, label in enumerate(IOB2_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# The special token label ID used for sub-word tokens and special tokens
# (these are ignored during loss computation and evaluation)
IGNORE_LABEL_ID = -100

# Recommended base models for German NER fine-tuning:
#   - "bert-base-german-cased"          (dbmdz German BERT, good default)
#   - "deepset/gbert-base"              (German BERT by deepset)
#   - "deepset/gbert-large"             (larger, needs more VRAM)
#   - "xlm-roberta-base"                (multilingual, robust fallback)


# ─────────────────────────────────────────────
#  2. DATA LOADING & IOB2 CONVERSION
# ─────────────────────────────────────────────

def load_label_studio_raw(filepath: str) -> List[Dict]:
    """
    Load the Label Studio JSON export and return a list of records,
    each with: id, text, meta_temp, and character-level entities.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        entities = []
        for ann in entry.get("label", []):
            label = ann["labels"][0] if ann.get("labels") else None
            if label and label in ALL_LABELS:
                entities.append({
                    "start": ann["start"],
                    "end":   ann["end"],
                    "label": label,
                    "text":  ann["text"],
                })
        records.append({
            "id":        entry["id"],
            "text":      entry["text"],
            "meta_temp": entry.get("meta_temp", "Unknown"),
            "entities":  sorted(entities, key=lambda e: e["start"]),
        })
    return records


def align_labels_with_tokens(
    text: str,
    entities: List[Dict],
    tokenizer,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Tokenize `text` and assign an IOB2 label to each token based on
    character-offset entity annotations.

    How it works:
      1. Tokenize the text, keeping track of each token's character offsets
      2. For each token, check if its character span falls inside an entity
      3. Assign B-LABEL to the first token of an entity, I-LABEL to
         continuation tokens, and O to tokens outside all entities
      4. Special tokens ([CLS], [SEP]) and sub-word continuation tokens
         (where offset is None) get IGNORE_LABEL_ID so they're excluded
         from loss and evaluation

    Returns:
        input_ids:   token IDs for the model
        attention_mask: 1 for real tokens, 0 for padding
        label_ids:   IOB2 label IDs (or IGNORE_LABEL_ID for special/sub-word tokens)
    """
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False,       # we'll pad later per batch
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    offsets = tokenized["offset_mapping"]

    # Build a character-to-entity lookup for efficient alignment
    # For each character position, store (entity_label, is_start_of_entity)
    char_labels = [None] * len(text)
    for ent in entities:
        for char_idx in range(ent["start"], ent["end"]):
            if char_idx < len(char_labels):
                is_start = (char_idx == ent["start"])
                char_labels[char_idx] = (ent["label"], is_start)

    label_ids = []
    previous_entity_label = None

    for token_idx, (start, end) in enumerate(offsets):
        # Special tokens ([CLS], [SEP], [PAD]) have offset (0, 0)
        if start == 0 and end == 0:
            label_ids.append(IGNORE_LABEL_ID)
            previous_entity_label = None
            continue

        # Look at the first character of this token to determine its label
        char_info = char_labels[start] if start < len(char_labels) else None

        if char_info is None:
            # Token is outside any entity
            label_ids.append(LABEL_TO_ID["O"])
            previous_entity_label = None
        else:
            entity_label, is_entity_start = char_info
            if is_entity_start or entity_label != previous_entity_label:
                # First token of this entity → B-tag
                label_ids.append(LABEL_TO_ID[f"B-{entity_label}"])
            else:
                # Continuation token within the same entity → I-tag
                label_ids.append(LABEL_TO_ID[f"I-{entity_label}"])
            previous_entity_label = entity_label

    return input_ids, attention_mask, label_ids


def convert_dataset(
    records: List[Dict],
    tokenizer,
) -> List[Dict]:
    """
    Convert all records from character-offset format to tokenized IOB2 format.
    Returns a list of dicts with: id, input_ids, attention_mask, labels, text, entities, meta_temp.
    """
    converted = []
    alignment_errors = 0

    for rec in records:
        try:
            input_ids, attention_mask, label_ids = align_labels_with_tokens(
                rec["text"], rec["entities"], tokenizer
            )
            converted.append({
                "id":             rec["id"],
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         label_ids,
                "text":           rec["text"],
                "entities":       rec["entities"],
                "meta_temp":      rec["meta_temp"],
            })
        except Exception as e:
            alignment_errors += 1
            print(f"  Warning: alignment failed for doc {rec['id']}: {e}")

    if alignment_errors > 0:
        print(f"  Total alignment errors: {alignment_errors}/{len(records)}")

    return converted


# ─────────────────────────────────────────────
#  3. STRATIFIED TRAIN / DEV / TEST SPLIT
# ─────────────────────────────────────────────

def stratified_split(
    records: List[Dict],
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split records into train/dev/test, stratified by meta_temp so that
    each split has a proportional mix of Low/Medium/High temperature texts.

    Returns (train_records, dev_records, test_records).
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    # Group by temperature
    by_temp = defaultdict(list)
    for rec in records:
        by_temp[rec["meta_temp"]].append(rec)

    train_all, dev_all, test_all = [], [], []

    for temp_label, temp_records in sorted(by_temp.items()):
        rng.shuffle(temp_records)
        n = len(temp_records)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        # Remaining goes to test (handles rounding)

        train_all.extend(temp_records[:n_train])
        dev_all.extend(temp_records[n_train:n_train + n_dev])
        test_all.extend(temp_records[n_train + n_dev:])

    # Shuffle each split so training isn't ordered by temperature
    rng.shuffle(train_all)
    rng.shuffle(dev_all)
    rng.shuffle(test_all)

    return train_all, dev_all, test_all


# ─────────────────────────────────────────────
#  4. COLLATION & DATA LOADING
# ─────────────────────────────────────────────

class NERDataset(torch.utils.data.Dataset):
    """Simple wrapper around tokenized records for PyTorch DataLoader."""

    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "input_ids":      torch.tensor(rec["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(rec["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(rec["labels"], dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad all sequences in a batch to the same length.
    input_ids and attention_mask are padded with 0.
    labels are padded with IGNORE_LABEL_ID (-100) so padding is excluded from loss.
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    padded_input_ids = []
    padded_attention = []
    padded_labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_attention.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), IGNORE_LABEL_ID, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention),
        "labels":         torch.stack(padded_labels),
    }


# ─────────────────────────────────────────────
#  5. TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.
    Returns the average loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    for batch in progress:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def evaluate_on_split(
    model,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
) -> Tuple[float, List[List[str]], List[List[str]]]:
    """
    Run inference on a data split.

    Returns:
        avg_loss:    average loss across batches
        all_true:    list of label sequences (one per document) — true labels
        all_pred:    list of label sequences (one per document) — predicted labels

    Only tokens with label != IGNORE_LABEL_ID are included, so the output
    contains only "real" tokens (no [CLS], [SEP], [PAD], or sub-word continuations).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

            # Get predicted label IDs (argmax over logits)
            predictions = torch.argmax(outputs.logits, dim=-1)  # (batch, seq_len)
            labels = batch["labels"]                             # (batch, seq_len)

            # Convert to lists and filter out ignored positions
            for pred_seq, label_seq in zip(predictions, labels):
                doc_true = []
                doc_pred = []
                for pred_id, label_id in zip(pred_seq.tolist(), label_seq.tolist()):
                    if label_id == IGNORE_LABEL_ID:
                        continue
                    doc_true.append(ID_TO_LABEL[label_id])
                    doc_pred.append(ID_TO_LABEL[pred_id])
                all_true.append(doc_true)
                all_pred.append(doc_pred)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, all_true, all_pred


# ─────────────────────────────────────────────
#  6. TOKEN PREDICTIONS → CHARACTER SPANS
# ─────────────────────────────────────────────

def token_predictions_to_char_spans(
    text: str,
    tokenizer,
    pred_label_ids: List[int],
) -> List[Dict]:
    """
    Convert token-level IOB2 predictions back to character-offset entity spans,
    matching the format used by evaluation_utils.py: {start, end, label, text}.

    This is the inverse of align_labels_with_tokens().
    """
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False,
    )
    offsets = tokenized["offset_mapping"]

    entities = []
    current_entity = None  # {"label": ..., "start": ..., "end": ...}

    # We need to iterate over only the "real" tokens (skip special tokens)
    real_token_idx = 0

    for token_idx, (char_start, char_end) in enumerate(offsets):
        # Special tokens have offset (0, 0)
        if char_start == 0 and char_end == 0:
            continue

        if real_token_idx >= len(pred_label_ids):
            break

        iob_label = ID_TO_LABEL.get(pred_label_ids[real_token_idx], "O")
        real_token_idx += 1

        if iob_label.startswith("B-"):
            # Close any open entity
            if current_entity is not None:
                current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)

            # Start a new entity
            entity_label = iob_label[2:]  # strip "B-"
            current_entity = {
                "start": char_start,
                "end":   char_end,
                "label": entity_label,
            }

        elif iob_label.startswith("I-") and current_entity is not None:
            # Continuation of the current entity — extend the span
            entity_label = iob_label[2:]
            if entity_label == current_entity["label"]:
                current_entity["end"] = char_end
            else:
                # I-tag doesn't match current entity — close and start new
                current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)
                current_entity = {
                    "start": char_start,
                    "end":   char_end,
                    "label": entity_label,
                }

        else:
            # O-tag or I-tag without a preceding B-tag
            if current_entity is not None:
                current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)
                current_entity = None

    # Close the last entity if still open
    if current_entity is not None:
        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
        entities.append(current_entity)

    return entities


def generate_char_span_predictions(
    model,
    records: List[Dict],
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Run the fine-tuned model on a list of records and produce character-offset
    predictions compatible with evaluation_utils.py.

    Returns a list of: {"id": ..., "entities": [{start, end, label, text}]}
    """
    model.eval()
    pred_records = []

    for i in tqdm(range(0, len(records), batch_size), desc="Generating predictions"):
        batch_records = records[i:i + batch_size]
        texts = [rec["text"] for rec in batch_records]

        # Tokenize the batch
        tokenized = tokenizer(
            texts,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

        # Keep offsets on CPU, move the rest to device
        offset_mappings = tokenized.pop("offset_mapping")
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu()  # (batch, seq_len)

        # Convert each document's predictions back to character spans
        for j, rec in enumerate(batch_records):
            pred_ids_full = predictions[j].tolist()
            offsets_j = offset_mappings[j].tolist()

            # Extract only "real" token predictions (skip special tokens)
            real_pred_ids = []
            for token_idx, (cs, ce) in enumerate(offsets_j):
                if cs == 0 and ce == 0:
                    continue
                real_pred_ids.append(pred_ids_full[token_idx])

            entities = token_predictions_to_char_spans(
                rec["text"], tokenizer, real_pred_ids
            )
            pred_records.append({
                "id":       rec["id"],
                "entities": entities,
            })

    return pred_records


# ─────────────────────────────────────────────
#  7. seqeval METRICS (token-level, for monitoring)
# ─────────────────────────────────────────────

def compute_seqeval_metrics(
    all_true: List[List[str]],
    all_pred: List[List[str]],
) -> Dict:
    """
    Compute token-level NER metrics using the seqeval library.
    Used for monitoring during training — the final thesis evaluation
    uses character-offset matching via evaluation_utils.py instead.
    """
    try:
        from seqeval.metrics import classification_report, f1_score
        f1 = f1_score(all_true, all_pred, average="weighted")
        report = classification_report(all_true, all_pred, digits=4)
        return {"f1": f1, "report": report}
    except ImportError:
        # Fallback: simple token-level accuracy
        correct = 0
        total = 0
        for true_seq, pred_seq in zip(all_true, all_pred):
            for t, p in zip(true_seq, pred_seq):
                total += 1
                if t == p:
                    correct += 1
        accuracy = correct / max(total, 1)
        return {"f1": accuracy, "report": f"Token accuracy: {accuracy:.4f}"}


# ─────────────────────────────────────────────
#  8. MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def main():
    test_ratio = 1.0 - TRAIN_RATIO - DEV_RATIO
    assert test_ratio > 0, "TRAIN_RATIO + DEV_RATIO must be < 1.0"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Reproducibility ──
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokenizer and model ──
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(IOB2_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,  # base model has different num_labels
    )
    model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  IOB2 label count: {len(IOB2_LABELS)}")

    # ── Load & convert data ──
    print(f"\nLoading data from: {INPUT_PATH}")
    raw_records = load_label_studio_raw(INPUT_PATH)
    print(f"  Raw records: {len(raw_records)}")

    print(f"  Converting to IOB2 token format...")
    all_records = convert_dataset(raw_records, tokenizer)
    print(f"  Successfully converted: {len(all_records)}")

    # ── Split ──
    train_data, dev_data, test_data = stratified_split(
        all_records,
        train_ratio=TRAIN_RATIO,
        dev_ratio=DEV_RATIO,
        test_ratio=test_ratio,
        seed=SEED,
    )

    # Print split statistics
    for split_name, split_data in [("Train", train_data), ("Dev", dev_data), ("Test", test_data)]:
        temp_counts = Counter(r["meta_temp"] for r in split_data)
        entity_counts = sum(
            1 for r in split_data for lbl in r["labels"] if lbl not in (IGNORE_LABEL_ID, LABEL_TO_ID["O"])
        )
        print(f"  {split_name}: {len(split_data)} docs, {entity_counts} entity tokens, "
              f"temps: {dict(sorted(temp_counts.items()))}")

    # ── Save split IDs for reproducibility ──
    split_info = {
        "train_ids": [r["id"] for r in train_data],
        "dev_ids":   [r["id"] for r in dev_data],
        "test_ids":  [r["id"] for r in test_data],
        "seed":      SEED,
        "split":     f"{TRAIN_RATIO}/{DEV_RATIO}/{test_ratio}",
    }
    split_path = os.path.join(OUTPUT_DIR, "split_ids.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Split IDs saved to: {split_path}")

    # ── DataLoaders ──
    train_loader = DataLoader(
        NERDataset(train_data), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        NERDataset(dev_data), batch_size=BATCH_SIZE * 2,
        shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        NERDataset(test_data), batch_size=BATCH_SIZE * 2,
        shuffle=False, collate_fn=collate_fn,
    )

    # ── Optimizer & Scheduler ──
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    print(f"{'=' * 60}\n")

    # ── Training loop with early stopping ──
    best_dev_f1 = 0.0
    best_epoch = -1
    patience = 3
    patience_counter = 0
    training_log = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        # Evaluate on dev
        dev_loss, dev_true, dev_pred = evaluate_on_split(model, dev_loader, device, desc=f"Epoch {epoch + 1} [Dev]")
        dev_metrics = compute_seqeval_metrics(dev_true, dev_pred)
        dev_f1 = dev_metrics["f1"]

        epoch_log = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "dev_loss":   round(dev_loss, 4),
            "dev_f1":     round(dev_f1, 4),
        }
        training_log.append(epoch_log)

        print(f"\n  Epoch {epoch + 1}/{EPOCHS}: "
              f"train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}, dev_f1={dev_f1:.4f}")

        # Check for improvement
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model checkpoint
            checkpoint_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  → New best! Saved checkpoint (dev_f1={dev_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.1f}s")
    print(f"  Best epoch: {best_epoch} (dev_f1={best_dev_f1:.4f})")

    # ── Load best checkpoint for final evaluation ──
    print(f"\n  Loading best checkpoint for final evaluation...")
    checkpoint_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    model.to(device)

    # ── Final evaluation on TEST set ──
    print(f"\n{'=' * 60}")
    print(f"  FINAL EVALUATION ON TEST SET ({len(test_data)} documents)")
    print(f"{'=' * 60}")

    # Token-level metrics (seqeval)
    test_loss, test_true, test_pred = evaluate_on_split(model, test_loader, device, desc="Test")
    test_metrics = compute_seqeval_metrics(test_true, test_pred)
    print(f"\n  Token-level (seqeval) F1: {test_metrics['f1']:.4f}")
    print(test_metrics["report"])

    # Character-offset entity-level metrics (evaluation_utils.py — the thesis standard)
    print(f"\n  Generating character-offset predictions for entity-level evaluation...")
    pred_records = generate_char_span_predictions(model, test_data, tokenizer, device)

    # Convert test_data to gold format expected by evaluation_utils
    gold_records = [{
        "id":        rec["id"],
        "text":      rec["text"],
        "meta_temp": rec["meta_temp"],
        "entities":  rec["entities"],
    } for rec in test_data]

    # Run tiered evaluation (strict + relaxed)
    report_content = []
    report_content.append(f"BERT Fine-Tuned NER Evaluation Report")
    report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Base model: {BASE_MODEL}")
    report_content.append(f"Best epoch: {best_epoch}/{EPOCHS}")
    report_content.append(f"Training time: {training_time:.1f}s")
    report_content.append(f"Test records: {len(test_data)}")
    report_content.append(f"Training log: {training_log}\n")

    all_results = {}

    # Group test data by temperature (same logic as classical_baseline.py)
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

            if temp_label == "Overall":
                print(format_tiered_report(tiered_results, f"BERT Fine-Tuned ({matching_mode.upper()} matching)"))

            report_content.append(
                format_tiered_report(tiered_results, f"BERT Fine-Tuned - {temp_label} ({matching_mode.upper()} matching)")
            )
            report_content.append("\n")

    # ── Save everything ──
    # Evaluation results JSON
    results_path = os.path.join(OUTPUT_DIR, "bert_finetuned_evaluation_results.json")
    save_results_json(all_results, results_path)
    print(f"\n  Results JSON: {results_path}")

    # Predictions JSON
    pred_path = os.path.join(OUTPUT_DIR, "bert_finetuned_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2, ensure_ascii=False)
    print(f"  Predictions: {pred_path}")

    # Evaluation report
    report_path = os.path.join(OUTPUT_DIR, "bert_finetuned_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    print(f"  Report: {report_path}")

    # ── Error analysis reports (same format as classical_baseline.py) ──

    # Sample error deep-dive (appended to the main report as well)
    print(f"  Generating sample error analysis...")
    deep_dive_text = generate_error_samples(gold_records, pred_records, num_samples=15)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + deep_dive_text)

    # Full document-by-document comparison log
    print(f"  Generating full document-by-document log...")
    full_log_text = generate_full_document_log(gold_records, pred_records)
    full_log_path = os.path.join(OUTPUT_DIR, "bert_finetuned_full_document_log.txt")
    with open(full_log_path, "w", encoding="utf-8") as f:
        f.write(full_log_text)
    print(f"  Full document log: {full_log_path}")

    # Category-level frequency error report
    print(f"  Generating category frequency error report...")
    category_error_text = generate_category_error_report(gold_records, pred_records)
    category_error_path = os.path.join(OUTPUT_DIR, "bert_finetuned_category_error_analysis.txt")
    with open(category_error_path, "w", encoding="utf-8") as f:
        f.write(category_error_text)
    print(f"  Category error analysis: {category_error_path}")

    # Training log
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "seed": SEED, "train_ratio": TRAIN_RATIO, "dev_ratio": DEV_RATIO,
            "best_epoch": best_epoch,
            "best_dev_f1": best_dev_f1,
            "training_time_seconds": training_time,
            "epochs_log": training_log,
        }, f, indent=2)
    print(f"  Training log: {log_path}")

    # Optionally save the full model
    if SAVE_MODEL:
        model_dir = os.path.join(OUTPUT_DIR, "model")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"  Model saved to: {model_dir}")

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
