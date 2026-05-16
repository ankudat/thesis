"""
semantic_preservation.py
=========================
Evaluate semantic preservation of anonymized texts (RQ2a / RQ2b).

Implements the methodology from Section 4.3.2 of the thesis:

  1. TAG-AND-REPLACE ANONYMIZATION
     For each pipeline's detected entities, replace PII spans in the original
     text with category placeholders (e.g., "[PERSON]", "[IBAN]").
     This produces an anonymized version per pipeline.

  2. MASKED-TEXT COMPARISON (BERTScore)
     To isolate whether anonymization changed *non-PII* content:
       - Mask all ground-truth PII spans in the ORIGINAL text  → masked_original
       - Mask all placeholder tokens in the ANONYMIZED text     → masked_anonymized
     Then compute BERTScore(masked_anonymized, masked_original).
     Perfect semantic preservation → near-identical masked texts → BERTScore ≈ 1.0.

  3. FULL-TEXT BERTScore (supplementary)
     Also compute BERTScore(anonymized, original) on the full texts as a
     secondary measure capturing overall readability/coherence shifts.

  4. LEVENSHTEIN DISTANCE ON MASKED TEXTS (supplementary)
     Character-level edit distance on masked texts, normalized by length.
     Catches small insertions/deletions that embedding similarity might miss.

  5. ROUGE-1 AND BLEU-4 ON FULL TEXTS (supplementary)
     Following Staab et al. (ICLR 2025) and Dorémus et al. (JMIR AI 2025):
       - ROUGE-1: Unigram overlap F1 between anonymized and original text.
         Measures how many individual words survived anonymization.
       - BLEU-4: 4-gram precision with brevity penalty between anonymized
         and original text. Measures how many short word sequences remain
         intact after anonymization.
     These surface-level metrics complement BERTScore (semantic similarity)
     by quantifying lexical fidelity. Tag-and-replace methods will naturally
     score higher (most n-grams survive) than prompt-based rewrites (which
     rephrase the text), even when semantic preservation is comparable.

  6. PII LEAKAGE RATE (privacy metric)
     For each document, checks how many ground-truth PII strings still
     appear (case-insensitive) in the anonymized/rewritten text.
     Computed as: leaked_count / total_pii_count.
     This is the privacy metric for prompt-based rewrites (where span-based
     recall cannot be computed). For tag-and-replace pipelines, leakage
     rate is related to but NOT identical to (1 - recall): boundary
     mismatches count as false negatives in recall but do not cause
     leakage if the PII string is still removed. Therefore leakage rate
     is always <= (1 - recall). Enables direct privacy comparison across
     all anonymization paradigms.

  7. PER-PIPELINE AND PER-COMPLEXITY BREAKDOWN
     Results are reported overall AND per complexity level (Low/Medium/High).

Inputs required:
  - Gold-standard records (Label Studio export with ground-truth entities)
  - Prediction records from each pipeline (JSON with detected entities)
  - OR: the evaluation report files (this script can regenerate predictions
    from the original pipeline scripts; see usage notes)

Usage:
    python semantic_preservation.py

    Adjust the USER SETTINGS section below to point to the data and
    prediction files.

Requirements:
    pip install bert-score torch transformers tqdm

    Note: bert-score will download a model (~400 MB) on first run.
    Recommended: use a multilingual or German model for BERTScore
    (this script defaults to "bert-base-multilingual-cased" which
    handles German well).

"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

# Path to the Label Studio export (ground truth)
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_PATH = os.path.join(BASE_DIR, "data", "label_studio", "20260302_Export_Label_Studio_Client_Notes.json")

# Path to split IDs (to evaluate only test set, matching other evaluations)
SPLIT_IDS = os.path.join(BASE_DIR, "results", "bert_finetuned", "split_ids.json")

# Prediction files from each pipeline (JSON: list of {id, entities})
# Set to None to skip a pipeline.
PREDICTION_FILES = {
    # Classical baselines
    "spaCy + Regex":                   os.path.join(BASE_DIR, "results", "classical_baselines", "spacy", "spacy_predictions.json"),
    "BERT (pretrained) + Regex":       os.path.join(BASE_DIR, "results", "classical_baselines", "bert", "bert_predictions.json"),
    "BERT Fine-Tuned":                 os.path.join(BASE_DIR, "results", "bert_finetuned", "bert_finetuned_predictions.json"),
    # Presidio
    "Presidio":                        os.path.join(BASE_DIR, "results", "presidio_baseline", "presidio_predictions.json"),
    # LLM tag-and-replace: Llama-3
    "LLM Llama-3 [zero-shot]":         os.path.join(BASE_DIR, "results", "llm_baselines", "llm_meta_llama_3_8b_instruct_zero_shot_predictions.json"),
    "LLM Llama-3 [few-shot]":          os.path.join(BASE_DIR, "results", "llm_baselines", "llm_meta_llama_3_8b_instruct_few_shot_predictions.json"),
    "LLM Llama-3 [few-shot +verify]":  os.path.join(BASE_DIR, "results", "llm_baselines", "llm_meta_llama_3_8b_instruct_few_shot_verified_predictions.json"),
    # LLM tag-and-replace: Qwen2.5
    "LLM Qwen2.5 [zero-shot]":         os.path.join(BASE_DIR, "results", "llm_baselines", "llm_qwen2.5_7b_instruct_zero_shot_predictions.json"),
    "LLM Qwen2.5 [few-shot]":          os.path.join(BASE_DIR, "results", "llm_baselines", "llm_qwen2.5_7b_instruct_few_shot_predictions.json"),
    "LLM Qwen2.5 [few-shot +verify]":  os.path.join(BASE_DIR, "results", "llm_baselines", "llm_qwen2.5_7b_instruct_few_shot_verified_predictions.json"),
    # LLM tag-and-replace: SauerkrautLM
    "LLM SauerkrautLM [zero-shot]":    os.path.join(BASE_DIR, "results", "llm_baselines", "llm_llama_3.1_sauerkrautlm_8b_instruct_zero_shot_predictions.json"),
    "LLM SauerkrautLM [few-shot]":     os.path.join(BASE_DIR, "results", "llm_baselines", "llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_predictions.json"),
    "LLM SauerkrautLM [few-shot +verify]": os.path.join(BASE_DIR, "results", "llm_baselines", "llm_llama_3.1_sauerkrautlm_8b_instruct_few_shot_verified_predictions.json"),
    # LLM fine-tuned (QLoRA) — only Llama-3 has been fine-tuned so far
    "LLM Llama-3 [fine-tuned]":        os.path.join(BASE_DIR, "results", "llm_finetuned", "llm_finetuned_meta_llama_3_8b_instruct", "llm_finetuned_meta_llama_3_8b_instruct_predictions.json"),
}

# Prompt-based anonymization files (rewritten text, not entity predictions)
# These use a different format: {"id", "rewritten_text"} instead of {"id", "entities"}
PROMPT_ANON_FILES = {
    "LLM Llama-3 [prompt zero-shot]":      os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_meta_llama_3_8b_instruct_zero_shot_predictions.json"),
    "LLM Llama-3 [prompt few-shot]":       os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_meta_llama_3_8b_instruct_few_shot_predictions.json"),
    "LLM Qwen2.5 [prompt zero-shot]":      os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_qwen2.5_7b_instruct_zero_shot_predictions.json"),
    "LLM Qwen2.5 [prompt few-shot]":       os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_qwen2.5_7b_instruct_few_shot_predictions.json"),
    "LLM SauerkrautLM [prompt zero-shot]": os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_llama_3.1_sauerkrautlm_8b_instruct_zero_shot_predictions.json"),
    "LLM SauerkrautLM [prompt few-shot]":  os.path.join(BASE_DIR, "results", "llm_prompt_anonymize", "prompt_anon_llama_3.1_sauerkrautlm_8b_instruct_few_shot_predictions.json"),
    # External API baseline
    "Datenwertsch\u00f6pfung API":         os.path.join(BASE_DIR, "results", "datenwertschoepfung_baseline", "datenwertschoepfung_predictions.json"),
}

# Output directory for semantic preservation results
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "semantic_preservation")

# BERTScore model (multilingual handles German well)
BERTSCORE_MODEL = "bert-base-multilingual-cased"

# Number of example comparisons to include in the qualitative report
NUM_QUALITATIVE_SAMPLES = 30

# Limit documents (set to None for full evaluation, small int for testing)
LIMIT = None

SEED = 42


# =====================================================================
#  IMPORTS
# =====================================================================

import json
import os
import re
import time
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from evaluation_utils import ALL_LABELS, load_label_studio_export

# =====================================================================
#  1. ANONYMIZATION (Tag-and-Replace)
# =====================================================================

# Placeholder format: [LABEL] or [LABEL_N] for repeated entities
PLACEHOLDER_TEMPLATE = "[{label}]"


def anonymize_text_with_predictions(
    text: str,
    pred_entities: List[Dict],
) -> Tuple[str, List[Dict]]:
    """
    Replace each predicted PII span with a placeholder token.

    For example:
        "Herr Markus Steiner aus Zürich"
        with pred_entities for PER(Markus Steiner) and LOC(Zürich)
    becomes:
        "Herr [PER] aus [LOC]"

    Entities are processed back-to-front (by start offset) so that
    earlier character positions remain valid after replacement.

    Returns:
        anonymized_text: the text with PII replaced by placeholders
        replacements:    list of {start, end, label, original, placeholder}
    """
    # Sort entities by start offset descending (so we replace from the end)
    sorted_ents = sorted(pred_entities, key=lambda e: e["start"], reverse=True)

    anonymized = text
    replacements = []

    for ent in sorted_ents:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        original_text = text[start:end]

        placeholder = PLACEHOLDER_TEMPLATE.format(label=label)

        anonymized = anonymized[:start] + placeholder + anonymized[end:]

        replacements.append({
            "start": start,
            "end": end,
            "label": label,
            "original": original_text,
            "placeholder": placeholder,
        })

    replacements.reverse()  # back to document order
    return anonymized, replacements


def anonymize_text_with_gold(
    text: str,
    gold_entities: List[Dict],
) -> str:
    """
    Replace ground-truth PII spans with placeholders.
    Used to create the 'ideal' anonymization for comparison.
    """
    anon, _ = anonymize_text_with_predictions(text, gold_entities)
    return anon


def resolve_overlapping_entities(entities: List[Dict]) -> List[Dict]:
    """
    Remove overlapping entities by keeping the longer span.

    LLM-based taggers sometimes produce overlapping entity spans
    (e.g., tagging both "Dr. Müller" and "Müller" separately).
    This causes garbled output in anonymize_text_with_predictions
    because both spans try to replace overlapping character ranges.

    Strategy: sort by start offset, then greedily keep spans that
    don't overlap with the last accepted span. When two spans overlap,
    keep the longer one (more informative for anonymization).
    """
    if not entities:
        return []

    # Sort by start, then by length descending (prefer longer spans first)
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))

    resolved = [sorted_ents[0]]
    for ent in sorted_ents[1:]:
        prev = resolved[-1]
        if ent["start"] >= prev["end"]:
            # No overlap — keep it
            resolved.append(ent)
        else:
            # Overlap — keep the longer span
            prev_len = prev["end"] - prev["start"]
            curr_len = ent["end"] - ent["start"]
            if curr_len > prev_len:
                resolved[-1] = ent  # replace with longer

    return resolved


# =====================================================================
#  2. MASKED-TEXT GENERATION
# =====================================================================

# Universal mask token used to blank out PII regions
MASK_TOKEN = "▮"


def mask_pii_spans(text: str, entities: List[Dict]) -> str:
    """
    Replace PII spans in the text with a fixed mask token.

    This is used to create comparable texts where PII regions are
    neutralized, so that only non-PII content differences are measured.

    Args:
        text:     the text (original or anonymized)
        entities: list of {start, end, label} spans to mask

    Returns:
        masked text with PII regions replaced by MASK_TOKEN
    """
    sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)
    masked = text
    for ent in sorted_ents:
        masked = masked[:ent["start"]] + MASK_TOKEN + masked[ent["end"]:]
    return masked


def mask_placeholders(anonymized_text: str) -> str:
    """
    Replace all [LABEL] placeholders in an anonymized text with MASK_TOKEN.

    This ensures the masked version of the anonymized text is comparable
    to the masked version of the original text.
    """
    # Match [LABEL] or [LABEL_N] patterns for all known labels
    label_pattern = "|".join(re.escape(f"[{label}]") for label in ALL_LABELS)
    # Also handle numbered variants like [PER_1], [PER_2]
    numbered_pattern = "|".join(
        re.escape(f"[{label}") + r"_?\d*\]" for label in ALL_LABELS
    )
    combined = f"({label_pattern}|{numbered_pattern})"
    return re.sub(combined, MASK_TOKEN, anonymized_text)


def create_masked_pair(
    original_text: str,
    gold_entities: List[Dict],
    anonymized_text: str,
) -> Tuple[str, str]:
    """
    Create the masked text pair for BERTScore comparison.

    Following the thesis methodology (Section 4.3.2):
      1. Mask ground-truth PII in the original     → masked_original
      2. Mask placeholders in the anonymized text   → masked_anonymized

    Returns:
        (masked_original, masked_anonymized)
    """
    masked_original = mask_pii_spans(original_text, gold_entities)
    masked_anonymized = mask_placeholders(anonymized_text)
    return masked_original, masked_anonymized


# =====================================================================
#  3. BERTSCORE COMPUTATION
# =====================================================================

def compute_bertscore_batch(
    candidates: List[str],
    references: List[str],
    model_type: str = "bert-base-multilingual-cased",
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Compute BERTScore for a batch of (candidate, reference) pairs.

    Args:
        candidates:  list of candidate texts (anonymized/masked)
        references:  list of reference texts (original/masked)
        model_type:  HuggingFace model for BERTScore embeddings
        batch_size:  batch size for inference (reduced to 16 for stability)
        device:      "cuda" or "cpu"

    Returns:
        Dict with "precision", "recall", "f1" — each a list of floats
    """
    import bert_score as bs

    # bert_score.score returns (P, R, F1) as tensors
    P, R, F1 = bs.score(
        candidates,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=True,
        lang="de",
    )

    result = {
        "precision": P.tolist(),
        "recall":    R.tolist(),
        "f1":        F1.tolist(),
    }

    # Force-unload the cached BERTScore model to prevent GPU memory buildup.
    # bert_score caches the model in a module-level dict; without clearing it,
    # repeated calls accumulate GPU memory until CUDA hangs.
    import gc, torch
    if hasattr(bs, 'scorer') and bs.scorer is not None:
        del bs.scorer
        bs.scorer = None
    # Also clear the internal model cache used by some bert_score versions
    if hasattr(bs.score, '__wrapped__'):
        pass  # some versions don't cache this way
    for attr in ['_model', '_tokenizer', 'model', 'tokenizer']:
        if hasattr(bs, attr):
            try:
                delattr(bs, attr)
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =====================================================================
#  4. LEVENSHTEIN DISTANCE (supplementary metric)
# =====================================================================

def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Compute the normalized Levenshtein distance between two strings.

    Returns a value in [0, 1] where:
      0.0 = identical strings
      1.0 = completely different strings

    Uses dynamic programming (O(n*m) time and O(min(n,m)) space).
    """
    if s1 == s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 1.0

    # Ensure s1 is the shorter string for space efficiency
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev_row = list(range(len1 + 1))
    for j in range(1, len2 + 1):
        curr_row = [j] + [0] * len1
        for i in range(1, len1 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                curr_row[i - 1] + 1,       # insertion
                prev_row[i] + 1,            # deletion
                prev_row[i - 1] + cost,     # substitution
            )
        prev_row = curr_row

    distance = prev_row[len1]
    return distance / max(len1, len2)


# =====================================================================
#  4b. ROUGE-1 AND BLEU-4 (following Staab et al., ICLR 2025)
# =====================================================================

def compute_rouge1_f1(candidate: str, reference: str) -> float:
    """
    Compute ROUGE-1 F1 score between candidate and reference texts.

    ROUGE-1 measures unigram overlap — how many individual words in the
    anonymized text also appear in the original. Following Staab et al.
    (ICLR 2025), this is computed on the full (original, anonymized) pair.

    Returns F1 score in [0, 1].
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    # Count unigram overlaps
    cand_counts = {}
    for t in cand_tokens:
        cand_counts[t] = cand_counts.get(t, 0) + 1

    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    # Intersection count (min of each token's count)
    overlap = 0
    for token, count in cand_counts.items():
        if token in ref_counts:
            overlap += min(count, ref_counts[token])

    precision = overlap / len(cand_tokens) if cand_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_bleu4(candidate: str, reference: str) -> float:
    """
    Compute BLEU-4 score between candidate and reference texts.

    BLEU-4 measures 4-gram overlap with a brevity penalty — how many
    short word sequences from the anonymized text appear in the original.
    Following Staab et al. (ICLR 2025) and Dorémus et al. (JMIR AI 2025),
    this is computed on the full (original, anonymized) pair.

    A high BLEU score indicates the anonymization only changed PII spans
    and left everything else untouched. A low score signals the model
    rewrote more than necessary.

    Returns score in [0, 1].
    """
    import math

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    # Compute n-gram precision for n=1..4
    precisions = []
    for n in range(1, 5):
        if len(cand_tokens) < n:
            precisions.append(0.0)
            continue

        # Extract n-grams
        cand_ngrams = {}
        for i in range(len(cand_tokens) - n + 1):
            ng = tuple(cand_tokens[i:i + n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        # Clipped count
        clipped = 0
        total = 0
        for ng, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
            total += count

        precisions.append(clipped / total if total > 0 else 0.0)

    # If any precision is 0, BLEU is 0 (avoid log(0))
    if any(p == 0.0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(math.log(p) for p in precisions) / 4

    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))

    return bp * math.exp(log_avg)


# =====================================================================
#  5. EVALUATION PIPELINE
# =====================================================================

def evaluate_semantic_preservation_rewrite(
    gold_records: List[Dict],
    rewrite_records: List[Dict],
    pipeline_name: str,
    bertscore_model: str = "bert-base-multilingual-cased",
    device: str = "cuda",
) -> Dict:
    """
    Semantic preservation evaluation for PROMPT-BASED rewrites.

    Unlike tag-and-replace (where the anonymized text is constructed from
    entity predictions), the rewritten text is received directly.

    Computes:
      - BERTScore(rewritten, original) on full texts
      - BERTScore(masked_rewrite, masked_original) where ground-truth PII
        is masked in the original, and any surviving PII + generic replacements
        are masked in the rewrite (approximation: we mask the same PII strings
        if they appear, and any [LABEL]-style placeholders)
      - Levenshtein distance
    """
    import torch

    rewrite_by_id = {r["id"]: r.get("rewritten_text", "") for r in rewrite_records}

    doc_results = []
    full_originals = []
    full_rewrites = []
    masked_originals = []
    masked_rewrites = []

    for gold in gold_records:
        doc_id = gold["id"]
        text = gold["text"]
        gold_ents = gold["entities"]
        complexity = gold.get("meta_temp", "Unknown")

        rewritten = rewrite_by_id.get(doc_id, "")
        if not rewritten:
            continue

        # Mask ground-truth PII in original
        masked_orig = mask_pii_spans(text, gold_ents)

        # For the rewrite, mask any surviving PII strings + placeholder patterns
        masked_rew = rewritten
        # First, mask [LABEL]-style placeholders if the LLM used them
        masked_rew = mask_placeholders(masked_rew)
        # Then mask any surviving ground-truth PII strings
        for ent in sorted(gold_ents, key=lambda e: len(e["text"]), reverse=True):
            # Replace all occurrences (case-insensitive) of the PII text
            pattern = re.compile(re.escape(ent["text"]), re.IGNORECASE)
            masked_rew = pattern.sub(MASK_TOKEN, masked_rew)

        full_originals.append(text)
        full_rewrites.append(rewritten)
        masked_originals.append(masked_orig)
        masked_rewrites.append(masked_rew)

        lev_masked = normalized_levenshtein(masked_orig, masked_rew)

        # Per-document ROUGE-1 and BLEU-4 (following Staab et al., ICLR 2025)
        rouge1 = compute_rouge1_f1(rewritten, text)
        bleu4 = compute_bleu4(rewritten, text)

        # PII leakage rate: proportion of ground-truth PII strings that
        # still appear in the rewritten text (case-insensitive).
        # Uses word boundary matching (\b) to avoid false positives from
        # German compound words (e.g., "Fischer" in "Fischerei").
        leaked = 0
        rewritten_lower = rewritten.lower()
        for ent in gold_ents:
            search = re.escape(ent["text"].lower())
            if re.search(r'\b' + search + r'\b', rewritten_lower):
                leaked += 1
        total_pii = len(gold_ents)
        pii_leakage_rate = leaked / total_pii if total_pii > 0 else 0.0

        doc_results.append({
            "id": doc_id,
            "complexity": complexity,
            "original_length": len(text),
            "anonymized_length": len(rewritten),
            "num_gold_entities": len(gold_ents),
            "pii_leaked": leaked,
            "pii_total": total_pii,
            "pii_leakage_rate": round(pii_leakage_rate, 6),
            "levenshtein_masked": round(lev_masked, 6),
            "rouge1_full": round(rouge1, 6),
            "bleu4_full": round(bleu4, 6),
            "anonymized_text": rewritten,
            "masked_original": masked_orig,
            "masked_anonymized": masked_rew,
        })

    if not doc_results:
        return {"pipeline": pipeline_name, "per_document": [], "aggregated": {}}

    # BERTScore on masked pairs
    print(f"\n  Computing BERTScore (masked texts) for {pipeline_name}...")
    if not torch.cuda.is_available():
        device = "cpu"

    masked_bs = compute_bertscore_batch(
        masked_rewrites, masked_originals,
        model_type=bertscore_model, device=device,
    )

    # BERTScore on full texts
    print(f"  Computing BERTScore (full texts) for {pipeline_name}...")
    full_bs = compute_bertscore_batch(
        full_rewrites, full_originals,
        model_type=bertscore_model, device=device,
    )

    for i, doc in enumerate(doc_results):
        doc["bertscore_masked_p"] = round(masked_bs["precision"][i], 6)
        doc["bertscore_masked_r"] = round(masked_bs["recall"][i], 6)
        doc["bertscore_masked_f1"] = round(masked_bs["f1"][i], 6)
        doc["bertscore_full_p"] = round(full_bs["precision"][i], 6)
        doc["bertscore_full_r"] = round(full_bs["recall"][i], 6)
        doc["bertscore_full_f1"] = round(full_bs["f1"][i], 6)

    aggregated = aggregate_results(doc_results, pipeline_name)

    return {
        "pipeline": pipeline_name,
        "per_document": doc_results,
        "aggregated": aggregated,
    }


def evaluate_semantic_preservation(
    gold_records: List[Dict],
    pred_records: List[Dict],
    pipeline_name: str,
    bertscore_model: str = "bert-base-multilingual-cased",
    device: str = "cuda",
) -> Dict:
    """
    Full semantic preservation evaluation for one pipeline.

    Steps:
      1. Anonymize each text using the pipeline's predicted entities
      2. Create masked text pairs (masked_original, masked_anonymized)
      3. Compute BERTScore on masked pairs
      4. Compute BERTScore on full (original, anonymized) pairs
      5. Compute normalized Levenshtein on masked pairs
      6. Aggregate by complexity level

    Returns:
        Dict with all scores, per-document and aggregated
    """
    import torch

    # Build prediction lookup
    pred_by_id = {r["id"]: r.get("entities", []) for r in pred_records}

    # Prepare text pairs
    doc_results = []
    masked_originals = []
    masked_anonymizeds = []
    full_originals = []
    full_anonymizeds = []

    for gold in gold_records:
        doc_id = gold["id"]
        text = gold["text"]
        gold_ents = gold["entities"]
        pred_ents = pred_by_id.get(doc_id, [])
        complexity = gold.get("meta_temp", "Unknown")

        # Resolve overlapping entity spans (LLMs sometimes produce these)
        pred_ents = resolve_overlapping_entities(pred_ents)

        # Step 1: Anonymize with predicted entities
        anonymized_text, replacements = anonymize_text_with_predictions(text, pred_ents)

        # Step 2: Create masked pair
        masked_orig, masked_anon = create_masked_pair(text, gold_ents, anonymized_text)

        # Step 3: Also create gold-standard anonymization for reference
        gold_anonymized = anonymize_text_with_gold(text, gold_ents)

        # Store for batch BERTScore computation
        masked_originals.append(masked_orig)
        masked_anonymizeds.append(masked_anon)
        full_originals.append(text)
        full_anonymizeds.append(anonymized_text)

        # Per-document Levenshtein
        lev_masked = normalized_levenshtein(masked_orig, masked_anon)

        # Per-document ROUGE-1 and BLEU-4 (following Staab et al., ICLR 2025)
        rouge1 = compute_rouge1_f1(anonymized_text, text)
        bleu4 = compute_bleu4(anonymized_text, text)

        # PII leakage rate: proportion of ground-truth PII strings that
        # still appear in the anonymized text (case-insensitive).
        # Uses word boundary matching to avoid false positives from
        # German compound words. For tag-and-replace, leakage is related
        # to but not identical to (1 - recall): boundary mismatches count
        # as FN in recall but do not cause leakage if the PII is removed.
        leaked = 0
        anon_lower = anonymized_text.lower()
        for ent in gold_ents:
            search = re.escape(ent["text"].lower())
            if re.search(r'\b' + search + r'\b', anon_lower):
                leaked += 1
        total_pii = len(gold_ents)
        pii_leakage_rate = leaked / total_pii if total_pii > 0 else 0.0

        doc_results.append({
            "id": doc_id,
            "complexity": complexity,
            "original_length": len(text),
            "anonymized_length": len(anonymized_text),
            "gold_anonymized_length": len(gold_anonymized),
            "num_pred_entities": len(pred_ents),
            "num_gold_entities": len(gold_ents),
            "pii_leaked": leaked,
            "pii_total": total_pii,
            "pii_leakage_rate": round(pii_leakage_rate, 6),
            "levenshtein_masked": round(lev_masked, 6),
            "rouge1_full": round(rouge1, 6),
            "bleu4_full": round(bleu4, 6),
            "anonymized_text": anonymized_text,
            "masked_original": masked_orig,
            "masked_anonymized": masked_anon,
        })

    # Step 3: Compute BERTScore on masked pairs
    print(f"\n  Computing BERTScore (masked texts) for {pipeline_name}...")
    if not torch.cuda.is_available():
        device = "cpu"
        print("  Warning: CUDA not available, using CPU (this will be slow)")

    masked_bs = compute_bertscore_batch(
        masked_anonymizeds, masked_originals,
        model_type=bertscore_model, device=device,
    )

    # Step 4: Compute BERTScore on full texts
    print(f"  Computing BERTScore (full texts) for {pipeline_name}...")
    full_bs = compute_bertscore_batch(
        full_anonymizeds, full_originals,
        model_type=bertscore_model, device=device,
    )

    # Attach scores to per-document results
    for i, doc in enumerate(doc_results):
        doc["bertscore_masked_p"] = round(masked_bs["precision"][i], 6)
        doc["bertscore_masked_r"] = round(masked_bs["recall"][i], 6)
        doc["bertscore_masked_f1"] = round(masked_bs["f1"][i], 6)
        doc["bertscore_full_p"] = round(full_bs["precision"][i], 6)
        doc["bertscore_full_r"] = round(full_bs["recall"][i], 6)
        doc["bertscore_full_f1"] = round(full_bs["f1"][i], 6)

    # Step 5: Aggregate results
    aggregated = aggregate_results(doc_results, pipeline_name)

    return {
        "pipeline": pipeline_name,
        "per_document": doc_results,
        "aggregated": aggregated,
    }


def aggregate_results(doc_results: List[Dict], pipeline_name: str) -> Dict:
    """
    Aggregate per-document scores into overall and per-complexity means.
    """
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    def _std(values):
        if len(values) < 2:
            return 0.0
        m = _mean(values)
        return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    def _aggregate_group(docs):
        if not docs:
            return {}
        return {
            "count": len(docs),
            "bertscore_masked_f1_mean": round(_mean([d["bertscore_masked_f1"] for d in docs]), 4),
            "bertscore_masked_f1_std":  round(_std([d["bertscore_masked_f1"] for d in docs]), 4),
            "bertscore_masked_p_mean":  round(_mean([d["bertscore_masked_p"] for d in docs]), 4),
            "bertscore_masked_r_mean":  round(_mean([d["bertscore_masked_r"] for d in docs]), 4),
            "bertscore_full_f1_mean":   round(_mean([d["bertscore_full_f1"] for d in docs]), 4),
            "bertscore_full_f1_std":    round(_std([d["bertscore_full_f1"] for d in docs]), 4),
            "bertscore_full_p_mean":    round(_mean([d["bertscore_full_p"] for d in docs]), 4),
            "bertscore_full_r_mean":    round(_mean([d["bertscore_full_r"] for d in docs]), 4),
            "levenshtein_masked_mean":  round(_mean([d["levenshtein_masked"] for d in docs]), 4),
            "levenshtein_masked_std":   round(_std([d["levenshtein_masked"] for d in docs]), 4),
            "rouge1_full_mean":         round(_mean([d["rouge1_full"] for d in docs]), 4),
            "rouge1_full_std":          round(_std([d["rouge1_full"] for d in docs]), 4),
            "bleu4_full_mean":          round(_mean([d["bleu4_full"] for d in docs]), 4),
            "bleu4_full_std":           round(_std([d["bleu4_full"] for d in docs]), 4),
            "pii_leakage_rate_mean":    round(_mean([d["pii_leakage_rate"] for d in docs if "pii_leakage_rate" in d]), 4),
            "pii_leakage_rate_std":     round(_std([d["pii_leakage_rate"] for d in docs if "pii_leakage_rate" in d]), 4),
            "pii_leaked_total":         sum(d.get("pii_leaked", 0) for d in docs),
            "pii_total":                sum(d.get("pii_total", 0) for d in docs),
        }

    # Overall
    overall = _aggregate_group(doc_results)

    # Per complexity
    by_complexity = defaultdict(list)
    for doc in doc_results:
        by_complexity[doc["complexity"]].append(doc)

    per_complexity = {}
    for level in ["Low", "Medium", "High"]:
        if level in by_complexity:
            per_complexity[level] = _aggregate_group(by_complexity[level])

    return {
        "pipeline": pipeline_name,
        "overall": overall,
        "by_complexity": per_complexity,
    }


# =====================================================================
#  6. REPORT FORMATTING
# =====================================================================

def format_semantic_report(all_results: List[Dict]) -> str:
    """
    Format a comprehensive semantic preservation report comparing
    all pipelines.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("  SEMANTIC PRESERVATION EVALUATION REPORT")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  BERTScore model: {BERTSCORE_MODEL}")
    lines.append("=" * 78)

    # ── Summary Table (Overall) ──
    lines.append("\n" + "#" * 78)
    lines.append("  SUMMARY: Overall Semantic Preservation Scores")
    lines.append("#" * 78)
    lines.append("")
    lines.append(f"  {'Pipeline':<38} {'BERTScore':>10} {'BERTScore':>10} {'Levenshtein':>12} {'ROUGE-1':>8} {'BLEU-4':>8} {'PII Leak':>9}")
    lines.append(f"  {'':38} {'(masked)':>10} {'(full)':>10} {'(masked)':>12} {'(full)':>8} {'(full)':>8} {'rate':>9}")
    lines.append(f"  {'-' * 99}")

    for result in all_results:
        agg = result["aggregated"]["overall"]
        name = result["pipeline"]
        leak_str = f"{agg['pii_leakage_rate_mean']:>9.4f}" if "pii_leakage_rate_mean" in agg else f"{'n/a':>9}"
        lines.append(
            f"  {name:<38} "
            f"{agg['bertscore_masked_f1_mean']:>10.4f} "
            f"{agg['bertscore_full_f1_mean']:>10.4f} "
            f"{agg['levenshtein_masked_mean']:>12.4f} "
            f"{agg['rouge1_full_mean']:>8.4f} "
            f"{agg['bleu4_full_mean']:>8.4f} "
            f"{leak_str}"
        )

    lines.append(f"  {'-' * 99}")
    lines.append("  BERTScore: higher = better semantic preservation (max 1.0)")
    lines.append("  Levenshtein: lower = better (0.0 = identical masked texts)")
    lines.append("  PII Leak rate: lower = better privacy (0.0 = no PII survived anonymization)")

    # ── Per-Complexity Breakdown ──
    for level in ["Low", "Medium", "High"]:
        lines.append(f"\n  >>> Complexity: {level.upper()} <<<")
        lines.append(f"  {'Pipeline':<38} {'BERTScore':>10} {'BERTScore':>10} {'Levenshtein':>12} {'ROUGE-1':>8} {'BLEU-4':>8} {'n':>5}")
        lines.append(f"  {'':38} {'(masked)':>10} {'(full)':>10} {'(masked)':>12} {'(full)':>8} {'(full)':>8}")
        lines.append(f"  {'-' * 95}")

        for result in all_results:
            comp = result["aggregated"]["by_complexity"].get(level, {})
            if not comp:
                continue
            name = result["pipeline"]
            lines.append(
                f"  {name:<38} "
                f"{comp['bertscore_masked_f1_mean']:>10.4f} "
                f"{comp['bertscore_full_f1_mean']:>10.4f} "
                f"{comp['levenshtein_masked_mean']:>12.4f} "
                f"{comp['rouge1_full_mean']:>8.4f} "
                f"{comp['bleu4_full_mean']:>8.4f} "
                f"{comp['count']:>5}"
            )

    # ── Detailed Per-Pipeline Statistics ──
    lines.append(f"\n\n{'#' * 78}")
    lines.append("  DETAILED PER-PIPELINE STATISTICS")
    lines.append(f"{'#' * 78}")

    for result in all_results:
        agg = result["aggregated"]
        name = result["pipeline"]
        overall = agg["overall"]

        lines.append(f"\n  {'=' * 70}")
        lines.append(f"  Pipeline: {name}")
        lines.append(f"  {'=' * 70}")
        lines.append(f"  Documents evaluated: {overall['count']}")
        lines.append(f"")
        lines.append(f"  BERTScore (masked texts — primary metric):")
        lines.append(f"    F1:        {overall['bertscore_masked_f1_mean']:.4f} ± {overall['bertscore_masked_f1_std']:.4f}")
        lines.append(f"    Precision: {overall['bertscore_masked_p_mean']:.4f}")
        lines.append(f"    Recall:    {overall['bertscore_masked_r_mean']:.4f}")
        lines.append(f"")
        lines.append(f"  BERTScore (full texts — supplementary):")
        lines.append(f"    F1:        {overall['bertscore_full_f1_mean']:.4f} ± {overall['bertscore_full_f1_std']:.4f}")
        lines.append(f"    Precision: {overall['bertscore_full_p_mean']:.4f}")
        lines.append(f"    Recall:    {overall['bertscore_full_r_mean']:.4f}")
        lines.append(f"")
        lines.append(f"  Levenshtein (masked texts):")
        lines.append(f"    Mean:      {overall['levenshtein_masked_mean']:.4f} ± {overall['levenshtein_masked_std']:.4f}")
        lines.append(f"")
        lines.append(f"  ROUGE-1 (full texts — following Staab et al., ICLR 2025):")
        lines.append(f"    Mean:      {overall['rouge1_full_mean']:.4f} ± {overall['rouge1_full_std']:.4f}")
        lines.append(f"")
        lines.append(f"  BLEU-4 (full texts — following Staab et al., ICLR 2025 / Dorémus et al., JMIR AI 2025):")
        lines.append(f"    Mean:      {overall['bleu4_full_mean']:.4f} ± {overall['bleu4_full_std']:.4f}")
        lines.append(f"")
        if "pii_leakage_rate_mean" in overall:
            lines.append(f"  PII Leakage Rate (privacy metric — lower = better):")
            lines.append(f"    Mean:      {overall['pii_leakage_rate_mean']:.4f} ± {overall['pii_leakage_rate_std']:.4f}")
            lines.append(f"    Leaked:    {overall['pii_leaked_total']} / {overall['pii_total']} PII instances")

        for level in ["Low", "Medium", "High"]:
            comp = agg["by_complexity"].get(level, {})
            if comp:
                lines.append(f"")
                lines.append(f"  Complexity: {level} (n={comp['count']})")
                lines.append(f"    BERTScore masked F1: {comp['bertscore_masked_f1_mean']:.4f} ± {comp['bertscore_masked_f1_std']:.4f}")
                lines.append(f"    BERTScore full F1:   {comp['bertscore_full_f1_mean']:.4f} ± {comp['bertscore_full_f1_std']:.4f}")
                lines.append(f"    Levenshtein masked:  {comp['levenshtein_masked_mean']:.4f} ± {comp['levenshtein_masked_std']:.4f}")
                lines.append(f"    ROUGE-1 full:        {comp['rouge1_full_mean']:.4f} ± {comp['rouge1_full_std']:.4f}")
                lines.append(f"    BLEU-4 full:         {comp['bleu4_full_mean']:.4f} ± {comp['bleu4_full_std']:.4f}")
                if "pii_leakage_rate_mean" in comp:
                    lines.append(f"    PII leakage rate:    {comp['pii_leakage_rate_mean']:.4f} ± {comp['pii_leakage_rate_std']:.4f}")

    return "\n".join(lines)


def format_qualitative_samples(
    gold_records: List[Dict],
    all_results: List[Dict],
    num_samples: int = 30,
) -> str:
    """
    Generate a qualitative comparison showing original text, each pipeline's
    anonymization, and their BERTScore. Samples are stratified by complexity
    and include best/worst cases.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("  QUALITATIVE SAMPLES: Side-by-Side Anonymization Comparison")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)

    gold_by_id = {r["id"]: r for r in gold_records}

    # For each pipeline, find the worst-performing documents (lowest masked BERTScore)
    # and sample a mix of best/worst/random

    # Collect IDs across all pipelines by complexity
    all_doc_ids_by_complexity = defaultdict(set)
    for gold in gold_records:
        all_doc_ids_by_complexity[gold.get("meta_temp", "Unknown")].add(gold["id"])

    # Get worst-case documents from the first pipeline with results
    sample_ids = set()
    if all_results:
        first_result = all_results[0]
        per_doc = first_result["per_document"]

        # Sort by masked BERTScore ascending (worst first)
        sorted_docs = sorted(per_doc, key=lambda d: d["bertscore_masked_f1"])

        # Take some worst cases per complexity
        seen_complexity = defaultdict(int)
        for doc in sorted_docs:
            if seen_complexity[doc["complexity"]] < num_samples // 6:
                sample_ids.add(doc["id"])
                seen_complexity[doc["complexity"]] += 1

        # Add some random samples to reach the target
        remaining_ids = [d["id"] for d in per_doc if d["id"] not in sample_ids]
        random.shuffle(remaining_ids)
        for doc_id in remaining_ids:
            if len(sample_ids) >= num_samples:
                break
            sample_ids.add(doc_id)

    # Build lookup for each pipeline's per-document results
    pipeline_docs = {}
    for result in all_results:
        pipeline_docs[result["pipeline"]] = {
            d["id"]: d for d in result["per_document"]
        }

    # Generate samples
    for doc_id in sorted(sample_ids):
        gold = gold_by_id.get(doc_id)
        if not gold:
            continue

        lines.append(f"\n{'─' * 78}")
        lines.append(f"  Document ID: {doc_id} | Complexity: {gold.get('meta_temp', '?')}")
        lines.append(f"{'─' * 78}")
        lines.append(f"  ORIGINAL ({len(gold['text'])} chars, {len(gold['entities'])} PII entities):")
        lines.append(f"    {gold['text'][:500]}{'...' if len(gold['text']) > 500 else ''}")
        lines.append("")

        # Gold-standard anonymization
        gold_anon = anonymize_text_with_gold(gold["text"], gold["entities"])
        lines.append(f"  GOLD STANDARD ANONYMIZATION:")
        lines.append(f"    {gold_anon[:500]}{'...' if len(gold_anon) > 500 else ''}")
        lines.append("")

        for result in all_results:
            name = result["pipeline"]
            doc_data = pipeline_docs[name].get(doc_id)
            if not doc_data:
                continue

            lines.append(f"  {name}:")
            anon_text = doc_data.get('anonymized_text', '(text not available — loaded from cached results)')
            lines.append(f"    Anonymized: {anon_text[:500]}{'...' if len(anon_text) > 500 else ''}")
            lines.append(f"    BERTScore (masked): F1={doc_data['bertscore_masked_f1']:.4f} | "
                         f"BERTScore (full): F1={doc_data['bertscore_full_f1']:.4f} | "
                         f"Levenshtein: {doc_data['levenshtein_masked']:.4f} | "
                         f"ROUGE-1: {doc_data['rouge1_full']:.4f} | "
                         f"BLEU-4: {doc_data['bleu4_full']:.4f}")
            lines.append("")

    return "\n".join(lines)


# =====================================================================
#  7. MAIN
# =====================================================================

def main():
    import torch

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load gold standard ──
    print(f"\nLoading gold standard from: {INPUT_PATH}")
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

    # ── Load prediction files ──
    all_results = []

    for pipeline_name, pred_path in PREDICTION_FILES.items():
        if pred_path is None or not os.path.exists(pred_path):
            print(f"\n  Skipping {pipeline_name} (file not found: {pred_path})")
            continue

        # Skip if results already exist (resume after interruption)
        safe_name = pipeline_name.lower().replace(" ", "_").replace("+", "").replace("[", "").replace("]", "")
        per_doc_path = os.path.join(OUTPUT_DIR, f"{safe_name}_semantic_per_document.json")
        if os.path.exists(per_doc_path):
            print(f"\n  → SKIPPING {pipeline_name} (already exists: {per_doc_path})")
            # Load existing results so they appear in the final report
            with open(per_doc_path, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)
            existing_agg = aggregate_results(existing_docs, pipeline_name)
            all_results.append({
                "pipeline": pipeline_name,
                "per_document": existing_docs,
                "aggregated": existing_agg,
            })
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {pipeline_name}")
        print(f"{'=' * 60}")

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_records = json.load(f)
        print(f"  Loaded {len(pred_records)} prediction records")

        # Run semantic preservation evaluation
        result = evaluate_semantic_preservation(
            gold_records=gold_records,
            pred_records=pred_records,
            pipeline_name=pipeline_name,
            bertscore_model=BERTSCORE_MODEL,
            device=device,
        )

        all_results.append(result)

        # Save per-pipeline detailed results
        # Save without the full text fields to keep file sizes manageable
        slim_docs = []
        for d in result["per_document"]:
            slim = {k: v for k, v in d.items()
                    if k not in ("anonymized_text", "masked_original", "masked_anonymized")}
            slim_docs.append(slim)

        with open(per_doc_path, "w", encoding="utf-8") as f:
            json.dump(slim_docs, f, indent=2, ensure_ascii=False)
        print(f"  Per-document results: {per_doc_path}")

        # Free GPU memory before next pipeline (prevents CUDA hang)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  GPU memory cleared.")

    if not all_results and not PROMPT_ANON_FILES:
        print("\nNo pipelines could be evaluated. Check your prediction file paths.")
        return

    # ── Evaluate prompt-based rewrites ──
    for pipeline_name, pred_path in PROMPT_ANON_FILES.items():
        if pred_path is None or not os.path.exists(pred_path):
            print(f"\n  Skipping {pipeline_name} (file not found: {pred_path})")
            continue

        # Skip if results already exist (resume after interruption)
        safe_name = pipeline_name.lower().replace(" ", "_").replace("+", "").replace("[", "").replace("]", "")
        per_doc_path = os.path.join(OUTPUT_DIR, f"{safe_name}_semantic_per_document.json")
        if os.path.exists(per_doc_path):
            print(f"\n  → SKIPPING {pipeline_name} (already exists: {per_doc_path})")
            with open(per_doc_path, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)
            existing_agg = aggregate_results(existing_docs, pipeline_name)
            all_results.append({
                "pipeline": pipeline_name,
                "per_document": existing_docs,
                "aggregated": existing_agg,
            })
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating (prompt-based rewrite): {pipeline_name}")
        print(f"{'=' * 60}")

        with open(pred_path, "r", encoding="utf-8") as f:
            rewrite_records = json.load(f)
        print(f"  Loaded {len(rewrite_records)} rewrite records")

        result = evaluate_semantic_preservation_rewrite(
            gold_records=gold_records,
            rewrite_records=rewrite_records,
            pipeline_name=pipeline_name,
            bertscore_model=BERTSCORE_MODEL,
            device=device,
        )

        all_results.append(result)

        # Save per-pipeline detailed results
        slim_docs = []
        for d in result["per_document"]:
            slim = {k: v for k, v in d.items()
                    if k not in ("anonymized_text", "masked_original", "masked_anonymized")}
            slim_docs.append(slim)

        with open(per_doc_path, "w", encoding="utf-8") as f:
            json.dump(slim_docs, f, indent=2, ensure_ascii=False)
        print(f"  Per-document results: {per_doc_path}")

        # Free GPU memory before next pipeline (prevents CUDA hang)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  GPU memory cleared.")

    if not all_results:
        print("\nNo pipelines could be evaluated. Check your prediction file paths.")
        return

    # ── Generate and save reports ──
    print(f"\n{'=' * 60}")
    print(f"  Generating reports...")
    print(f"{'=' * 60}")

    # Main report
    report = format_semantic_report(all_results)
    report_path = os.path.join(OUTPUT_DIR, "semantic_preservation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n  Report saved: {report_path}")

    # Qualitative samples
    qual_report = format_qualitative_samples(
        gold_records, all_results, num_samples=NUM_QUALITATIVE_SAMPLES
    )
    qual_path = os.path.join(OUTPUT_DIR, "semantic_qualitative_samples.txt")
    with open(qual_path, "w", encoding="utf-8") as f:
        f.write(qual_report)
    print(f"  Qualitative samples: {qual_path}")

    # Aggregated results JSON (for tables/charts in the thesis)
    agg_results = {r["pipeline"]: r["aggregated"] for r in all_results}
    agg_path = os.path.join(OUTPUT_DIR, "semantic_aggregated_results.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2, ensure_ascii=False)
    print(f"  Aggregated JSON: {agg_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()