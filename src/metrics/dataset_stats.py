"""
dataset_stats.py
================
Recomputes the descriptive statistics reported in Section 3.1.6 (Dataset
Statistics) and Table 3.2 of the thesis, verifies them against the values
written in the thesis, and saves a plain-text report.

All figures are derived from the two canonical input files:
  - Gold (Label Studio export): the validated set of 2,542 notes, each with
    a `meta_temp` complexity label and a `label` list of gold entities.
  - Split-IDs JSON: the train / dev / test document-id assignment.

Not reproducible here: the per-complexity validation rejection rates quoted
in Section 3.1.6 (Low 12.9%, Medium 14.5%, High 17.7%). Those derive from
the pre-validation set of 3,000 generated notes (Section 3.1.5), which is
not part of the final dataset and is not loaded by this script.

Output:
  - <OUTPUT_DIR>/dataset_stats_report.txt

Run in VSCode:
    python src/metrics/dataset_stats.py
"""

# =====================================================================
#  USER SETTINGS
# =====================================================================

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GOLD_PATH  = os.path.join(BASE_DIR, "data", "label_studio",
                          "20260302_Export_Label_Studio_Client_Notes.json")
SPLIT_IDS  = os.path.join(BASE_DIR, "results", "bert_finetuned", "split_ids.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dataset_stats")


# =====================================================================
#  IMPLEMENTATION
# =====================================================================

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from io import StringIO

# Tier assignment for the twelve PII categories (mirrors Table 3.1).
TIER = {
    "PER": 1, "LOC": 1, "ORG": 1,
    "IBAN": 2, "EMAIL": 2, "PHONE": 2, "DATE": 2, "MONEY": 2,
    "JOB": 3, "AGE": 3, "NATION": 3, "EDU": 3,
}
# Display order: by tier, then as in Table 3.2.
CATEGORY_ORDER = ["PER", "LOC", "ORG",
                  "IBAN", "EMAIL", "PHONE", "DATE", "MONEY",
                  "JOB", "AGE", "NATION", "EDU"]
SPLITS = ("train", "dev", "test")
COMPLEXITY_ORDER = ("Low", "Medium", "High")

# Expected values as written in Section 3.1.6 and Table 3.2, for verification.
EXPECTED_DOCS = {"total": 2542, "train": 1270, "dev": 634, "test": 638}
EXPECTED_TEST_COMPLEXITY = {"Low": 218, "Medium": 215, "High": 205}
EXPECTED_ENTITIES_TOTAL = 20472
EXPECTED_SPLIT_ENTITY_TOTALS = {"train": 10224, "dev": 5111, "test": 5137}
EXPECTED_TIER_TOTALS = {1: 9800, 2: 6539, 3: 4133}
EXPECTED_TIER_PCT = {1: 47.9, 2: 31.9, 3: 20.2}
EXPECTED_ENT_BY_COMPLEXITY = {"Low": 7790, "Medium": 6643, "High": 6039}
EXPECTED_MEAN_ENT = 8.1
EXPECTED_MEDIAN_ENT = 8
EXPECTED_ENT_RANGE = (1, 16)
EXPECTED_MEAN_WORDS = 53.7
EXPECTED_MEDIAN_WORDS = 53
EXPECTED_WORDS_RANGE = (6, 90)
# Table 3.2: category -> (train, dev, test, total)
EXPECTED_TABLE = {
    "PER":    (2185, 1086, 1091, 4362),
    "LOC":    (1274,  653,  632, 2559),
    "ORG":    (1428,  711,  740, 2879),
    "IBAN":   ( 201,  104,  107,  412),
    "EMAIL":  ( 322,  147,  165,  634),
    "PHONE":  ( 243,  126,  113,  482),
    "DATE":   ( 959,  514,  508, 1981),
    "MONEY":  (1542,  729,  759, 3030),
    "JOB":    (1621,  803,  802, 3226),
    "AGE":    ( 153,   83,   81,  317),
    "NATION": ( 180,  103,   91,  374),
    "EDU":    ( 116,   52,   48,  216),
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute():
    """Read the canonical files and return all aggregates as a dict."""
    gold = load_json(GOLD_PATH)
    split_data = load_json(SPLIT_IDS)

    split_of = {}
    for s in SPLITS:
        for doc_id in split_data[f"{s}_ids"]:
            split_of[doc_id] = s

    docs_by_split = Counter()
    docs_by_split_complexity = Counter()       # (split, complexity) -> n
    docs_by_complexity = Counter()
    ent_by_cat_split = defaultdict(Counter)    # category -> split -> n
    ent_by_complexity = Counter()
    words_per_doc = []
    ents_per_doc = []

    for d in gold:
        doc_id = d.get("id")
        split = split_of.get(doc_id)
        if split is None:
            continue  # note not part of any split
        complexity = d.get("meta_temp", "?")
        docs_by_split[split] += 1
        docs_by_complexity[complexity] += 1
        docs_by_split_complexity[(split, complexity)] += 1

        labels = d.get("label") or []
        ents_per_doc.append(len(labels))
        words_per_doc.append(len(d.get("text", "").split()))

        for e in labels:
            cats = e.get("labels") or []
            if not cats:
                continue
            ent_by_cat_split[cats[0]][split] += 1
            ent_by_complexity[complexity] += 1

    # Per-category (train, dev, test, total) table.
    table = {}
    for cat in CATEGORY_ORDER:
        tr = ent_by_cat_split[cat]["train"]
        dv = ent_by_cat_split[cat]["dev"]
        te = ent_by_cat_split[cat]["test"]
        table[cat] = (tr, dv, te, tr + dv + te)

    tier_totals = {1: 0, 2: 0, 3: 0}
    split_entity_totals = {"train": 0, "dev": 0, "test": 0}
    for cat, (tr, dv, te, tot) in table.items():
        tier_totals[TIER[cat]] += tot
        split_entity_totals["train"] += tr
        split_entity_totals["dev"] += dv
        split_entity_totals["test"] += te
    entities_total = sum(tier_totals.values())

    return {
        "n_docs_total": sum(docs_by_split.values()),
        "docs_by_split": docs_by_split,
        "docs_by_split_complexity": docs_by_split_complexity,
        "docs_by_complexity": docs_by_complexity,
        "table": table,
        "tier_totals": tier_totals,
        "split_entity_totals": split_entity_totals,
        "entities_total": entities_total,
        "ent_by_complexity": ent_by_complexity,
        "words_per_doc": words_per_doc,
        "ents_per_doc": ents_per_doc,
    }


def _median(values):
    m = statistics.median(values)
    return int(m) if m == int(m) else m


def build_report(c):
    """Build the full plain-text report from the computed aggregates."""
    buf = StringIO()
    db = c["docs_by_split"]
    dbc = c["docs_by_split_complexity"]
    table = c["table"]
    tier = c["tier_totals"]
    set_ = c["split_entity_totals"]
    ent_total = c["entities_total"]
    ent_comp = c["ent_by_complexity"]
    epd = c["ents_per_doc"]
    wpd = c["words_per_doc"]

    mean_ent = round(statistics.mean(epd), 1)
    med_ent = _median(epd)
    mean_words = round(statistics.mean(wpd), 1)
    med_words = _median(wpd)
    tier_pct = {t: round(tier[t] / ent_total * 100, 1) for t in (1, 2, 3)}

    buf.write("=" * 78 + "\n")
    buf.write("  DATASET STATISTICS  (Section 3.1.6, Table 3.2)\n")
    buf.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buf.write(f"  Gold:      {os.path.relpath(GOLD_PATH, BASE_DIR)}\n")
    buf.write(f"  Split-IDs: {os.path.relpath(SPLIT_IDS, BASE_DIR)}\n")
    buf.write("=" * 78 + "\n\n")

    # --- Documents ---
    buf.write("DOCUMENTS\n")
    buf.write("-" * 78 + "\n")
    n_total = c["n_docs_total"]
    buf.write(f"  Total notes (in splits) : {n_total:,}\n")
    for s in SPLITS:
        pct = db[s] / n_total * 100 if n_total else 0
        buf.write(f"  {s.capitalize():<24}: {db[s]:>6,}  ({pct:4.1f}%)\n")
    buf.write("\n")
    buf.write("  Documents by complexity level\n")
    buf.write(f"  {'':<8}{'Low':>8}{'Medium':>8}{'High':>8}{'Total':>10}\n")
    for s in SPLITS:
        row = [dbc[(s, k)] for k in COMPLEXITY_ORDER]
        buf.write(f"  {s.capitalize():<8}"
                  + "".join(f"{v:>8,}" for v in row)
                  + f"{sum(row):>10,}\n")
    all_row = [sum(dbc[(s, k)] for s in SPLITS) for k in COMPLEXITY_ORDER]
    buf.write(f"  {'All':<8}"
              + "".join(f"{v:>8,}" for v in all_row)
              + f"{sum(all_row):>10,}\n\n")

    # --- Entities: Table 3.2 ---
    buf.write("ENTITIES\n")
    buf.write("-" * 78 + "\n")
    buf.write(f"  Total gold annotations  : {ent_total:,}\n")
    buf.write(f"  Train / Dev / Test      : "
              f"{set_['train']:,} / {set_['dev']:,} / {set_['test']:,}\n\n")
    buf.write("  Table 3.2 -- gold entity counts per category and split\n")
    buf.write(f"  {'Tier':<6}{'Category':<10}{'Train':>9}{'Dev':>9}"
              f"{'Test':>9}{'Total':>10}\n")
    buf.write("  " + "-" * 53 + "\n")
    for tier_no in (1, 2, 3):
        cats = [cat for cat in CATEGORY_ORDER if TIER[cat] == tier_no]
        for cat in cats:
            tr, dv, te, tot = table[cat]
            buf.write(f"  {'T' + str(tier_no):<6}{cat:<10}"
                      f"{tr:>9,}{dv:>9,}{te:>9,}{tot:>10,}\n")
        sub_tr = sum(table[cat][0] for cat in cats)
        sub_dv = sum(table[cat][1] for cat in cats)
        sub_te = sum(table[cat][2] for cat in cats)
        sub_to = sum(table[cat][3] for cat in cats)
        buf.write(f"  {'':<6}{'Subtotal':<10}"
                  f"{sub_tr:>9,}{sub_dv:>9,}{sub_te:>9,}{sub_to:>10,}\n")
        buf.write("  " + "-" * 53 + "\n")
    buf.write(f"  {'All':<6}{'12 cats':<10}"
              f"{set_['train']:>9,}{set_['dev']:>9,}"
              f"{set_['test']:>9,}{ent_total:>10,}\n\n")

    buf.write("  Tier shares of total annotations\n")
    for t in (1, 2, 3):
        buf.write(f"    Tier {t} : {tier[t]:>7,}  ({tier_pct[t]:4.1f}%)\n")
    buf.write("\n")
    buf.write("  Entities by complexity level (all splits)\n")
    for k in COMPLEXITY_ORDER:
        buf.write(f"    {k:<8}: {ent_comp[k]:>7,}\n")
    buf.write("\n")

    # --- Per-note statistics ---
    buf.write("PER-NOTE STATISTICS (all splits)\n")
    buf.write("-" * 78 + "\n")
    buf.write(f"  Entities per note : mean {mean_ent}, median {med_ent}, "
              f"range {min(epd)}-{max(epd)}\n")
    buf.write(f"  Words per note    : mean {mean_words}, median {med_words}, "
              f"range {min(wpd)}-{max(wpd)}\n")
    buf.write("  (\"words\" = whitespace-delimited tokens of the clean "
              "`text` field)\n\n")

    # --- Verification ---
    buf.write("VERIFICATION AGAINST THE THESIS (Section 3.1.6 and Table 3.2)\n")
    buf.write("-" * 78 + "\n")
    checks = []  # (label, ok, detail)

    def chk(label, ok, detail):
        checks.append((label, ok, detail))

    chk("Total notes", n_total == EXPECTED_DOCS["total"], f"{n_total:,}")
    chk("Train / Dev / Test",
        all(db[s] == EXPECTED_DOCS[s] for s in SPLITS),
        f"{db['train']:,} / {db['dev']:,} / {db['test']:,}")
    test_comp = {k: dbc[("test", k)] for k in COMPLEXITY_ORDER}
    chk("Test complexity split",
        test_comp == EXPECTED_TEST_COMPLEXITY,
        f"{test_comp['Low']} / {test_comp['Medium']} / {test_comp['High']}")
    chk("Total gold annotations",
        ent_total == EXPECTED_ENTITIES_TOTAL, f"{ent_total:,}")
    chk("Split entity totals",
        {s: set_[s] for s in SPLITS} == EXPECTED_SPLIT_ENTITY_TOTALS,
        f"{set_['train']:,} / {set_['dev']:,} / {set_['test']:,}")
    chk("Tier totals",
        tier == EXPECTED_TIER_TOTALS,
        f"{tier[1]:,} / {tier[2]:,} / {tier[3]:,}")
    chk("Tier percentages",
        tier_pct == EXPECTED_TIER_PCT,
        f"{tier_pct[1]} / {tier_pct[2]} / {tier_pct[3]}")
    chk("Entities by complexity",
        {k: ent_comp[k] for k in COMPLEXITY_ORDER} == EXPECTED_ENT_BY_COMPLEXITY,
        f"{ent_comp['Low']:,} / {ent_comp['Medium']:,} / {ent_comp['High']:,}")
    chk("Mean entities per note", mean_ent == EXPECTED_MEAN_ENT, str(mean_ent))
    chk("Median entities per note",
        med_ent == EXPECTED_MEDIAN_ENT, str(med_ent))
    chk("Entities-per-note range",
        (min(epd), max(epd)) == EXPECTED_ENT_RANGE,
        f"{min(epd)}-{max(epd)}")
    chk("Mean words per note",
        mean_words == EXPECTED_MEAN_WORDS, str(mean_words))
    chk("Median words per note",
        med_words == EXPECTED_MEDIAN_WORDS, str(med_words))
    chk("Words-per-note range",
        (min(wpd), max(wpd)) == EXPECTED_WORDS_RANGE,
        f"{min(wpd)}-{max(wpd)}")
    cells_total = len(EXPECTED_TABLE) * 4
    cells_ok = sum(
        1
        for cat, exp in EXPECTED_TABLE.items()
        for i in range(4)
        if table[cat][i] == exp[i]
    )
    chk("Table 3.2 (12 categories x 4 columns)",
        cells_ok == cells_total,
        f"{cells_ok}/{cells_total} cells match")

    for label, ok, detail in checks:
        tag = "[OK]      " if ok else "[MISMATCH]"
        dots = "." * max(2, 40 - len(label))
        buf.write(f"  {tag} {label} {dots} {detail}\n")
    n_ok = sum(1 for _, ok, _ in checks if ok)
    buf.write("\n")
    if n_ok == len(checks):
        buf.write(f"  Result: all {len(checks)} checks passed.\n")
    else:
        buf.write(f"  Result: {len(checks) - n_ok} of {len(checks)} "
                  f"checks FAILED. See [MISMATCH] lines above.\n")
    buf.write("\n")

    # --- Not computed here ---
    buf.write("NOT COMPUTED HERE\n")
    buf.write("-" * 78 + "\n")
    buf.write("  The per-complexity validation rejection rates in Section\n")
    buf.write("  3.1.6 (Low 12.9%, Medium 14.5%, High 17.7%) derive from the\n")
    buf.write("  pre-validation set of 3,000 generated notes (Section 3.1.5),\n")
    buf.write("  which this script does not load. They cannot be reproduced\n")
    buf.write("  from the final 2,542-note dataset alone.\n")
    buf.write("=" * 78 + "\n")

    return buf.getvalue(), (n_ok == len(checks))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in (GOLD_PATH, SPLIT_IDS):
        if not os.path.exists(path):
            print(f"Input file not found: {path}", file=sys.stderr)
            return 1

    computed = compute()
    report, all_passed = build_report(computed)

    print(report)

    report_path = os.path.join(OUTPUT_DIR, "dataset_stats_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")

    return 0 if all_passed else 2


if __name__ == "__main__":
    sys.exit(main())
