"""
attack_deep_dive_dump.py
=========================
Dump the adversarial-inference-attack deep-dive content into organized txt
files for offline browsing.

Produces, under `results/llm_judge_gemini/adversarial_attack/deep_dive/`:

    summary.txt
        Headline cross-pipeline summary table (matches the report) plus
        per-attribute hit counts.

    matches_grid.txt
        Compact (doc × pipeline × attribute) MATCH/NO_MATCH/SKIP grid.
        One section per attribute, one row per document, one column per
        pipeline. Useful to spot which docs are universally leaky.

    by_pipeline/<safe_name>.txt
        One file per pipeline. Each file lists every document in that
        pipeline showing original text, ground truth, attacker top-3
        guesses with confidence and reasoning, and the match-judge verdict
        per attribute.

    by_document/doc_<id>.txt
        Optional (--per-document). One file per document showing every
        pipeline's attack outcome side by side. ~611 files when complete.

Usage:
    python attack_deep_dive_dump.py
    python attack_deep_dive_dump.py --per-document
    python attack_deep_dive_dump.py --output-dir custom/path
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Paths and constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ATTACK_DIR  = os.path.join(BASE_DIR, "results", "llm_judge_gemini", "adversarial_attack")
GOLD_PATH   = os.path.join(BASE_DIR, "data", "label_studio", "20260302_Export_Label_Studio_Client_Notes.json")
DEFAULT_OUT = os.path.join(ATTACK_DIR, "deep_dive")

ATTACK_ATTRIBUTES = ["person", "age", "location", "occupation", "education", "nationality", "organization"]


# ─────────────────────────────────────────────────────────────────────────────
#  Pull pipeline-prediction paths and the anonymizer helper from the main
#  judge/attack script so this dump tool stays in sync if those paths change.
#  We add `src/metrics` and `src/anonymization` to sys.path because
#  llm_judge_gemini.py imports `evaluation_utils` from the latter.
# ─────────────────────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(_SRC_DIR), "anonymization"))
try:
    from llm_judge_gemini import (
        TAG_REPLACE_PREDICTIONS,
        PROMPT_REWRITE_PREDICTIONS,
        build_anonymized_text,
    )
except Exception as e:  # pragma: no cover - graceful fallback
    print(f"  Warning: could not import from llm_judge_gemini ({e}). "
          "Anonymized-text reconstruction will be disabled.")
    TAG_REPLACE_PREDICTIONS = {}
    PROMPT_REWRITE_PREDICTIONS = {}
    build_anonymized_text = None  # type: ignore


def _safe_name(pretty_name: str) -> str:
    """Mirror of the safe-name derivation used in run_attack_evaluation."""
    return (
        pretty_name.lower().replace(" ", "_").replace("+", "")
        .replace("[", "").replace("]", "")
    )


def build_anonymized_text_lookups():
    """
    Build {safe_name: (pipeline_type, lookup_dict)} where lookup_dict is:
      - tag-and-replace: {doc_id: list_of_entities}
      - prompt-rewrite : {doc_id: rewritten_text}
    Pipelines whose prediction file doesn't exist are skipped.

    Also returns {safe_name: pretty_name} so that downstream metric loading
    can locate evaluation_results.json relative to the prediction file.
    """
    lookups = {}
    safe_to_pretty: Dict[str, str] = {}
    # Tag-and-replace
    for pretty, path in TAG_REPLACE_PREDICTIONS.items():
        sn = _safe_name(pretty)
        safe_to_pretty[sn] = pretty
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        ents_by_id = {r["id"]: r.get("entities", []) for r in preds}
        lookups[sn] = ("tag", ents_by_id)
    # Prompt-rewrite
    for pretty, path in PROMPT_REWRITE_PREDICTIONS.items():
        sn = _safe_name(pretty)
        safe_to_pretty[sn] = pretty
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        rew_by_id = {r["id"]: r.get("rewritten_text", "") for r in preds}
        lookups[sn] = ("rewrite", rew_by_id)
    return lookups, safe_to_pretty


def get_anonymized_text(
    safe_name: str,
    doc_id: int,
    original_text: str,
    lookups: Dict,
) -> str:
    """Reconstruct the anonymized text the attacker saw, or '' if unavailable."""
    if safe_name not in lookups or build_anonymized_text is None:
        return ""
    kind, mapping = lookups[safe_name]
    if kind == "tag":
        ents = mapping.get(doc_id, [])
        return build_anonymized_text(original_text, ents) if ents else original_text
    elif kind == "rewrite":
        return mapping.get(doc_id, "") or ""
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Detection / privacy / utility metrics from existing result files.
#  We pull two complementary metric streams per pipeline:
#    - per-document leakage and BERTScore from results/semantic_preservation/
#    - per-pipeline overall P/R/F1 (All Categories, strict) from
#      `<predictions-path>` with `_predictions.json` replaced by
#      `_evaluation_results.json`. Some pipelines (prompt-rewrite, the external
#      API) don't have entity-level spans and therefore no eval-results file —
#      we surface "n/a" gracefully in those cases.
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_DIR = os.path.join(BASE_DIR, "results", "semantic_preservation")


def _load_per_doc_semantic(safe_name: str) -> Dict[int, Dict]:
    """{doc_id: full per-doc semantic record} or {} if file missing."""
    path = os.path.join(SEMANTIC_DIR, f"{safe_name}_semantic_per_document.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["id"]: d for d in docs if "id" in d}


def _derive_eval_results_path(predictions_path: str) -> str:
    """`...predictions.json` → `...evaluation_results.json` next to it."""
    if predictions_path.endswith("_predictions.json"):
        return predictions_path[: -len("_predictions.json")] + "_evaluation_results.json"
    return ""


def _safe_get(d, *keys):
    """Walk nested dict, return None on any missing key."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def _extract_overall_prf(eval_results: Dict) -> Dict[str, Dict]:
    """
    From a top-level evaluation_results dict (with keys Overall/Low/Medium/High),
    pull strict 'All Categories' overall P/R/F1 for each level.
    Returns {level: {"P": x, "R": y, "F1": z, "tp": ..., "fp": ..., "fn": ...}}.
    """
    out = {}
    if not eval_results:
        return out
    for level in ("Overall", "Low", "Medium", "High"):
        # Try the "All Categories" tier under strict matching
        ov = _safe_get(eval_results, level, "strict", "All Categories", "overall")
        if ov is None:
            # Fallback: some files might not have the tiered structure
            ov = _safe_get(eval_results, level, "overall")
        if isinstance(ov, dict):
            out[level] = {
                "P":   ov.get("precision"),
                "R":   ov.get("recall"),
                "F1":  ov.get("f1"),
                "tp":  ov.get("tp"),
                "fp":  ov.get("fp"),
                "fn":  ov.get("fn"),
            }
    return out


def load_pipeline_metrics(safe_to_pretty: Dict[str, str]) -> Dict[str, Dict]:
    """
    Returns {safe_name: {
        "per_doc": {doc_id: {pii_leakage_rate, pii_leaked, pii_total, bertscore_full_f1, ...}},
        "overall_prf": {level: {P, R, F1, tp, fp, fn}},   # may be empty
        "overall_leak": float | None,                      # mean over docs
    }}
    """
    out: Dict[str, Dict] = {}

    # Build pretty→predictions path index from both pipeline dicts.
    pretty_to_preds = {}
    for pretty, p in TAG_REPLACE_PREDICTIONS.items():
        pretty_to_preds[pretty] = p
    for pretty, p in PROMPT_REWRITE_PREDICTIONS.items():
        pretty_to_preds[pretty] = p

    for safe_name in sorted(safe_to_pretty.keys()):
        pretty = safe_to_pretty[safe_name]
        record = {"per_doc": {}, "overall_prf": {}, "overall_leak": None}

        # Per-doc semantic data
        per_doc = _load_per_doc_semantic(safe_name)
        record["per_doc"] = per_doc
        if per_doc:
            leak_vals = [d.get("pii_leakage_rate") for d in per_doc.values()
                         if d.get("pii_leakage_rate") is not None]
            if leak_vals:
                record["overall_leak"] = sum(leak_vals) / len(leak_vals)

        # Per-pipeline P/R/F1 (best-effort)
        preds_path = pretty_to_preds.get(pretty, "")
        eval_path = _derive_eval_results_path(preds_path) if preds_path else ""
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    er = json.load(f)
                record["overall_prf"] = _extract_overall_prf(er)
            except Exception:
                pass

        out[safe_name] = record
    return out


def _format_pct(v) -> str:
    return "n/a   " if v is None else f"{v:.1%}"


def _format_prf_overall(prf: Dict) -> str:
    """Format the Overall row of a P/R/F1 dict for header display."""
    if not prf or "Overall" not in prf:
        return "P=n/a R=n/a F1=n/a (no entity-level evaluation for this pipeline)"
    o = prf["Overall"]
    p, r, f1 = o.get("P"), o.get("R"), o.get("F1")
    if p is None:
        return "P=n/a R=n/a F1=n/a"
    return f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (tp={o.get('tp','?')}, fp={o.get('fp','?')}, fn={o.get('fn','?')})"


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_attack_files() -> Dict[str, List[Dict]]:
    """Map pipeline-safe-name → list of per-doc records."""
    if not os.path.isdir(ATTACK_DIR):
        sys.exit(f"Attack results dir does not exist: {ATTACK_DIR}")
    out: Dict[str, List[Dict]] = {}
    for fname in sorted(os.listdir(ATTACK_DIR)):
        if not fname.endswith("_attack_scores.json"):
            continue
        path = os.path.join(ATTACK_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        safe_name = fname.replace("_attack_scores.json", "")
        out[safe_name] = docs
    if not out:
        sys.exit(f"No *_attack_scores.json files found in {ATTACK_DIR}")
    return out


def load_gold_originals() -> Dict[int, Dict]:
    if not os.path.exists(GOLD_PATH):
        sys.exit(f"Gold export not found: {GOLD_PATH}")
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for entry in raw:
        out[entry["id"]] = {
            "id": entry["id"],
            "text": entry.get("text") or entry.get("data", {}).get("text", ""),
            "complexity": entry.get("meta_temp", "Unknown"),
        }
    return out


def load_ground_truth_cache() -> Dict[int, Dict]:
    path = os.path.join(ATTACK_DIR, "ground_truth_attributes.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return {int(k): v for k, v in cache.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_BADGE = {
    "MATCH":    "[+ MATCH]   ",
    "NO_MATCH": "[- NO_MATCH]",
    "SKIP":     "[. SKIP]    ",
}


def _format_value(v) -> str:
    if v is None:
        return "<null>"
    if isinstance(v, list):
        return "[]" if not v else "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def _wrap(text: str, width: int = 75, indent: str = "                            ") -> List[str]:
    """Word-wrap a string and prefix each line with `indent`."""
    lines = []
    cur = ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            if cur:
                lines.append(indent + cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(indent + cur)
    return lines


def render_one_doc_block(
    doc_id: int,
    gold: Dict,
    gt: Dict,
    attack_record: Dict,
    pipeline_label: str,
    anonymized_text: str = "",
    per_doc_metrics: Optional[Dict] = None,
) -> str:
    """Render one (document, pipeline) block as a string."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"  Document ID: {doc_id}    Complexity: {gold.get('complexity', '?')}    Pipeline: {pipeline_label}")
    lines.append("=" * 100)
    lines.append("")

    # Per-doc detection / utility metrics from semantic_preservation
    if per_doc_metrics:
        leak_rate = per_doc_metrics.get("pii_leakage_rate")
        leaked = per_doc_metrics.get("pii_leaked")
        total = per_doc_metrics.get("pii_total")
        bert_full_f1 = per_doc_metrics.get("bertscore_full_f1")
        bert_masked_f1 = per_doc_metrics.get("bertscore_masked_f1")
        rouge1 = per_doc_metrics.get("rouge1_full")
        leak_str = f"{leak_rate:.1%}" if leak_rate is not None else "n/a"
        lines.append("PER-DOCUMENT METRICS (from results/semantic_preservation/)")
        lines.append("-" * 100)
        lines.append(
            f"  PII leakage rate: {leak_str}  "
            f"({leaked if leaked is not None else '?'} of "
            f"{total if total is not None else '?'} PII tokens still visible)"
        )
        if any(v is not None for v in (bert_masked_f1, bert_full_f1, rouge1)):
            extra = []
            if bert_masked_f1 is not None:
                extra.append(f"BERTScore-masked F1={bert_masked_f1:.3f}")
            if bert_full_f1 is not None:
                extra.append(f"BERTScore-full F1={bert_full_f1:.3f}")
            if rouge1 is not None:
                extra.append(f"ROUGE-1={rouge1:.3f}")
            lines.append("  " + "  ".join(extra))
        lines.append("")

    lines.append("ORIGINAL TEXT")
    lines.append("-" * 100)
    lines.append(gold.get("text", "<missing>"))
    lines.append("")
    lines.append("ANONYMIZED TEXT (this is exactly what the attacker received)")
    lines.append("-" * 100)
    if anonymized_text:
        lines.append(anonymized_text)
    else:
        lines.append("<anonymized text not available — pipeline predictions file not found>")
    lines.append("")
    lines.append("GROUND TRUTH (extracted from original text)")
    lines.append("-" * 100)
    if gt:
        for attr in ATTACK_ATTRIBUTES:
            lines.append(f"  {attr:<13} {_format_value(gt.get(attr))}")
    else:
        lines.append("  <ground truth not available>")
    lines.append("")
    lines.append("ATTACKER (sees ONLY the anonymized text shown above)")
    lines.append("-" * 100)
    guesses = attack_record.get("attacker_guesses", {})
    match = attack_record.get("match", {})
    for attr in ATTACK_ATTRIBUTES:
        outcome = match.get(attr, "?")
        badge = OUTCOME_BADGE.get(outcome, f"[? {outcome}] ")
        g = guesses.get(attr, {})
        top3 = g.get("guesses", [])
        conf = g.get("confidence", "?")
        reason = (g.get("reasoning") or "").strip()
        lines.append(f"  {badge} {attr:<13} guesses={top3}  conf={conf}")
        if reason:
            wrapped = _wrap(reason, width=73, indent="                              ")
            for i, ln in enumerate(wrapped):
                # First continuation line gets an arrow marker
                if i == 0:
                    lines.append("                              -> " + ln.lstrip())
                else:
                    lines.append("                                 " + ln.lstrip())
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-pipeline files
# ─────────────────────────────────────────────────────────────────────────────

def write_per_pipeline(out_dir: str, attack_files, gold, gt_cache, anon_lookups, pipeline_metrics):
    sub = os.path.join(out_dir, "by_pipeline")
    os.makedirs(sub, exist_ok=True)
    for safe_name, docs in sorted(attack_files.items()):
        metrics = pipeline_metrics.get(safe_name, {})
        per_doc_metrics_map = metrics.get("per_doc", {})
        prf = metrics.get("overall_prf", {})
        overall_leak = metrics.get("overall_leak")

        path = os.path.join(sub, f"{safe_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#" * 100 + "\n")
            f.write(f"  ADVERSARIAL ATTACK DEEP-DIVE — PIPELINE: {safe_name}\n")
            f.write(f"  Documents in this file: {len(docs)}\n")
            f.write(f"  Source: {ATTACK_DIR}/{safe_name}_attack_scores.json\n")
            f.write("#" * 100 + "\n\n")

            f.write("PIPELINE-LEVEL METRICS\n")
            f.write("-" * 100 + "\n")
            # Detection (P/R/F1)
            f.write("  Detection (All Categories, strict matching):\n")
            f.write(f"    {_format_prf_overall(prf)}\n")
            for level in ("Low", "Medium", "High"):
                if level in prf:
                    o = prf[level]
                    p, r, f1 = o.get("P"), o.get("R"), o.get("F1")
                    if p is not None:
                        f.write(f"    {level:<8} P={p:.3f}  R={r:.3f}  F1={f1:.3f}\n")
            # Privacy (PII leakage)
            f.write("\n  Privacy:\n")
            if overall_leak is not None:
                f.write(f"    PII leakage rate (mean over {len(per_doc_metrics_map)} docs): {overall_leak:.1%}\n")
            else:
                f.write("    PII leakage rate: n/a (no per-doc semantic data)\n")
            f.write("\n")

            for d in docs:
                did = d.get("id")
                gd = gold.get(did, {"id": did, "text": "<missing>", "complexity": d.get("complexity", "?")})
                gt = gt_cache.get(did, {})
                anon = get_anonymized_text(safe_name, did, gd.get("text", ""), anon_lookups)
                pdoc = per_doc_metrics_map.get(did)
                f.write(render_one_doc_block(
                    did, gd, gt, d, safe_name,
                    anonymized_text=anon,
                    per_doc_metrics=pdoc,
                ))
                f.write("\n")
        print(f"  wrote {path}  ({len(docs)} docs)")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-document files (optional)
# ─────────────────────────────────────────────────────────────────────────────

def write_per_document(out_dir: str, attack_files, gold, gt_cache, anon_lookups, pipeline_metrics):
    sub = os.path.join(out_dir, "by_document")
    os.makedirs(sub, exist_ok=True)
    # Build doc → {pipeline: record}
    by_doc = defaultdict(dict)
    for safe_name, docs in attack_files.items():
        for d in docs:
            did = d.get("id")
            by_doc[did][safe_name] = d
    n = 0
    for did, per_pipe in sorted(by_doc.items()):
        gd = gold.get(did, {"id": did, "text": "<missing>", "complexity": "?"})
        gt = gt_cache.get(did, {})
        path = os.path.join(sub, f"doc_{did}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#" * 100 + "\n")
            f.write(f"  ADVERSARIAL ATTACK DEEP-DIVE — DOCUMENT {did}\n")
            f.write(f"  Complexity: {gd.get('complexity', '?')}\n")
            f.write(f"  Pipelines covered: {len(per_pipe)}\n")
            f.write("#" * 100 + "\n\n")
            f.write("ORIGINAL TEXT\n")
            f.write("-" * 100 + "\n")
            f.write(gd.get("text", "<missing>") + "\n\n")
            f.write("GROUND TRUTH (extracted from original text)\n")
            f.write("-" * 100 + "\n")
            if gt:
                for attr in ATTACK_ATTRIBUTES:
                    f.write(f"  {attr:<13} {_format_value(gt.get(attr))}\n")
            else:
                f.write("  <ground truth not available>\n")
            f.write("\n")
            for safe_name in sorted(per_pipe.keys()):
                d = per_pipe[safe_name]
                metrics = pipeline_metrics.get(safe_name, {})
                pdoc = (metrics.get("per_doc") or {}).get(did, {})
                prf = metrics.get("overall_prf", {})

                f.write("=" * 100 + "\n")
                f.write(f"  PIPELINE: {safe_name}\n")
                f.write("=" * 100 + "\n")

                # Per-pipeline + per-doc metrics
                f.write("PIPELINE METRICS  (overall, All Categories strict)\n")
                f.write("." * 100 + "\n")
                f.write(f"  {_format_prf_overall(prf)}\n")
                if pdoc:
                    leak_rate = pdoc.get("pii_leakage_rate")
                    leaked = pdoc.get("pii_leaked")
                    total = pdoc.get("pii_total")
                    leak_str = f"{leak_rate:.1%}" if leak_rate is not None else "n/a"
                    f.write(f"  This document — PII leakage: {leak_str}  "
                            f"({leaked if leaked is not None else '?'}/"
                            f"{total if total is not None else '?'} tokens still visible)\n")
                f.write("\n")

                anon = get_anonymized_text(safe_name, did, gd.get("text", ""), anon_lookups)
                f.write("ANONYMIZED TEXT (what the attacker received)\n")
                f.write("." * 100 + "\n")
                f.write((anon or "<anonymized text not available>") + "\n\n")
                f.write("ATTACK OUTCOME\n")
                f.write("." * 100 + "\n")
                guesses = d.get("attacker_guesses", {})
                match = d.get("match", {})
                for attr in ATTACK_ATTRIBUTES:
                    outcome = match.get(attr, "?")
                    badge = OUTCOME_BADGE.get(outcome, f"[? {outcome}] ")
                    g = guesses.get(attr, {})
                    top3 = g.get("guesses", [])
                    conf = g.get("confidence", "?")
                    reason = (g.get("reasoning") or "").strip()
                    f.write(f"  {badge} {attr:<13} guesses={top3}  conf={conf}\n")
                    if reason:
                        for i, ln in enumerate(_wrap(reason, width=73, indent="                              ")):
                            prefix = "                              -> " if i == 0 else "                                 "
                            f.write(prefix + ln.lstrip() + "\n")
                f.write("\n")
        n += 1
    print(f"  wrote {n} per-document files in {sub}")


# ─────────────────────────────────────────────────────────────────────────────
#  Summary file
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(out_dir: str, attack_files, pipeline_metrics):
    """Per-pipeline summary table + per-attribute hit counts + detection P/R/F1 + leakage."""
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#" * 100 + "\n")
        f.write("  ADVERSARIAL INFERENCE ATTACK — DEEP-DIVE SUMMARY\n")
        f.write("#" * 100 + "\n\n")
        f.write("Detection metrics (P/R/F1) and PII leakage are reproduced from\n")
        f.write("  results/<pipeline>/.../*_evaluation_results.json   (P/R/F1, All Categories, strict)\n")
        f.write("  results/semantic_preservation/*_semantic_per_document.json   (leakage; mean over docs)\n\n")
        f.write("Attack metrics:  success_rate = MATCH / (MATCH + NO_MATCH);  SKIP excluded.\n")
        f.write("  AnyLk = fraction of docs with at least one MATCH on at least one attribute.\n")
        f.write("  AvgLk = mean of the per-attribute success_rates.\n\n")

        # ── Table 1 — Detection + privacy + attack headline ─────────────
        f.write("=" * 100 + "\n")
        f.write("  Table 1 — Detection (P/R/F1) + PII leakage + attack headline\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Pipeline':<42}  {'P':>6} {'R':>6} {'F1':>6}   {'Leak':>6}   {'AnyLk':>6} {'AvgLk':>6}  {'n':>5}\n")
        f.write("-" * 100 + "\n")
        for pname, docs in sorted(attack_files.items()):
            metrics = pipeline_metrics.get(pname, {})
            prf = metrics.get("overall_prf", {})
            ov = prf.get("Overall", {})
            p, r, f1 = ov.get("P"), ov.get("R"), ov.get("F1")
            overall_leak = metrics.get("overall_leak")

            # Recompute attack rates
            per_attr_match     = defaultdict(int)
            per_attr_evaluable = defaultdict(int)
            any_match = 0
            for d in docs:
                m = d.get("match", {})
                doc_any = False
                for a in ATTACK_ATTRIBUTES:
                    v = m.get(a)
                    if v == "MATCH":
                        per_attr_match[a] += 1
                        per_attr_evaluable[a] += 1
                        doc_any = True
                    elif v == "NO_MATCH":
                        per_attr_evaluable[a] += 1
                if doc_any:
                    any_match += 1
            rates = [per_attr_match[a] / per_attr_evaluable[a]
                     for a in ATTACK_ATTRIBUTES if per_attr_evaluable[a] > 0]
            any_lk = any_match / len(docs) if docs else 0
            avg_lk = sum(rates) / len(rates) if rates else 0

            f.write(f"{pname:<42}  ")
            f.write(f"{p:>6.3f} " if p is not None else f"{'n/a':>6} ")
            f.write(f"{r:>6.3f} " if r is not None else f"{'n/a':>6} ")
            f.write(f"{f1:>6.3f}  " if f1 is not None else f"{'n/a':>6} ")
            f.write(" " + (f"{overall_leak:>6.1%}" if overall_leak is not None else f"{'n/a':>6}") + "   ")
            f.write(f"{any_lk:>6.1%} {avg_lk:>6.1%}  {len(docs):>5}\n")

        # ── Table 2 — Per-attribute attack success rate ─────────────────
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("  Table 2 — Per-attribute attack success rate\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Pipeline':<42} ")
        for a in ATTACK_ATTRIBUTES:
            f.write(f"{a[:5]:>7} ")
        f.write(f" {'AnyLk':>6} {'AvgLk':>6} {'n':>5}\n")
        f.write("-" * 110 + "\n")

        for pname, docs in sorted(attack_files.items()):
            per_attr_match     = defaultdict(int)
            per_attr_evaluable = defaultdict(int)
            any_match = 0
            for d in docs:
                m = d.get("match", {})
                doc_any = False
                for a in ATTACK_ATTRIBUTES:
                    v = m.get(a)
                    if v == "MATCH":
                        per_attr_match[a] += 1
                        per_attr_evaluable[a] += 1
                        doc_any = True
                    elif v == "NO_MATCH":
                        per_attr_evaluable[a] += 1
                if doc_any:
                    any_match += 1
            rates = []
            f.write(f"{pname:<42} ")
            for a in ATTACK_ATTRIBUTES:
                ev = per_attr_evaluable[a]
                if ev > 0:
                    sr = per_attr_match[a] / ev
                    rates.append(sr)
                    f.write(f"{sr:>7.1%} ")
                else:
                    f.write(f"{'n/a':>7} ")
            any_lk = any_match / len(docs) if docs else 0
            avg_lk = sum(rates) / len(rates) if rates else 0
            f.write(f" {any_lk:>6.1%} {avg_lk:>6.1%} {len(docs):>5}\n")

        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write("PER-ATTRIBUTE HIT COUNTS (across all pipelines)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Attribute':<14} {'MATCH':>10} {'NO_MATCH':>10} {'SKIP':>10} {'evaluable':>12} {'rate':>8}\n")
        for a in ATTACK_ATTRIBUTES:
            tot_m = tot_nm = tot_sk = 0
            for docs in attack_files.values():
                for d in docs:
                    v = d.get("match", {}).get(a)
                    if v == "MATCH":
                        tot_m += 1
                    elif v == "NO_MATCH":
                        tot_nm += 1
                    elif v == "SKIP":
                        tot_sk += 1
            ev = tot_m + tot_nm
            rate = (tot_m / ev) if ev > 0 else 0
            f.write(f"{a:<14} {tot_m:>10,} {tot_nm:>10,} {tot_sk:>10,} {ev:>12,} {rate:>8.1%}\n")
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Match grid file
# ─────────────────────────────────────────────────────────────────────────────

def write_matches_grid(out_dir: str, attack_files):
    """Compact per-attribute tables: rows = doc IDs, columns = pipelines, cell = M/N/S/?."""
    path = os.path.join(out_dir, "matches_grid.txt")
    pipelines = sorted(attack_files.keys())

    # Build doc_id → {pipeline: {attr: outcome}}
    table = defaultdict(lambda: defaultdict(dict))
    for pname, docs in attack_files.items():
        for d in docs:
            did = d.get("id")
            for a in ATTACK_ATTRIBUTES:
                table[did][pname][a] = d.get("match", {}).get(a, "?")
    doc_ids = sorted(table.keys())

    short = {"MATCH": "M", "NO_MATCH": ".", "SKIP": "_", "?": "?"}

    with open(path, "w", encoding="utf-8") as f:
        f.write("#" * 100 + "\n")
        f.write("  MATCH GRID — one section per attribute\n")
        f.write("#" * 100 + "\n\n")
        f.write("Legend:  M = MATCH  (attacker correctly inferred this attribute)\n")
        f.write("         . = NO_MATCH\n")
        f.write("         _ = SKIP   (GT was null in the original text)\n")
        f.write("         ? = unknown / parse failure\n\n")

        # 2-letter pipeline codes
        codes = {p: f"P{i+1:02d}" for i, p in enumerate(pipelines)}
        f.write("Pipeline codes:\n")
        for p in pipelines:
            f.write(f"  {codes[p]}  =  {p}\n")
        f.write("\n")

        for attr in ATTACK_ATTRIBUTES:
            f.write("=" * 100 + "\n")
            f.write(f"  ATTRIBUTE: {attr}\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'doc_id':>8} ")
            for p in pipelines:
                f.write(f"{codes[p]:>4} ")
            f.write("\n")
            for did in doc_ids:
                f.write(f"{did:>8} ")
                for p in pipelines:
                    v = table[did].get(p, {}).get(attr, "?")
                    f.write(f"{short.get(v, '?'):>4} ")
                f.write("\n")
            f.write("\n")
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dump adversarial-attack deep-dive into txt files.")
    ap.add_argument("--output-dir", default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--per-document", action="store_true", help="Also write one file per document (~611 files).")
    ap.add_argument("--no-pipeline", action="store_true", help="Skip per-pipeline files.")
    ap.add_argument("--no-summary", action="store_true", help="Skip summary.txt.")
    ap.add_argument("--no-grid", action="store_true", help="Skip matches_grid.txt.")
    args = ap.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading attack files from {ATTACK_DIR}")
    attack_files = load_attack_files()
    print(f"  {len(attack_files)} pipelines, {sum(len(d) for d in attack_files.values())} total records")

    print("Loading gold originals and GT cache")
    gold = load_gold_originals()
    gt_cache = load_ground_truth_cache()

    print("Building anonymized-text lookups (loading pipeline predictions)")
    anon_lookups, safe_to_pretty = build_anonymized_text_lookups()
    print(f"  {len(anon_lookups)} pipelines with reconstructable anonymized text")

    print("Loading pipeline detection metrics (P/R/F1) and per-doc PII leakage")
    pipeline_metrics = load_pipeline_metrics(safe_to_pretty)
    n_with_prf = sum(1 for m in pipeline_metrics.values() if m.get("overall_prf"))
    n_with_leak = sum(1 for m in pipeline_metrics.values() if m.get("overall_leak") is not None)
    print(f"  P/R/F1 found for {n_with_prf}/{len(pipeline_metrics)} pipelines")
    print(f"  Per-doc leakage found for {n_with_leak}/{len(pipeline_metrics)} pipelines")

    print(f"\nWriting deep-dive files into {out_dir}")
    if not args.no_summary:
        write_summary(out_dir, attack_files, pipeline_metrics)
    if not args.no_grid:
        write_matches_grid(out_dir, attack_files)
    if not args.no_pipeline:
        write_per_pipeline(out_dir, attack_files, gold, gt_cache, anon_lookups, pipeline_metrics)
    if args.per_document:
        write_per_document(out_dir, attack_files, gold, gt_cache, anon_lookups, pipeline_metrics)

    print(f"\nDone. Files written to: {out_dir}")


if __name__ == "__main__":
    main()
