# How well do LLMs anonymize text data?

### Master's Thesis, Hochschule Luzern (HSLU)

---

## Overview

This repository contains the code, synthetic dataset, and experimental results for a Master's thesis investigating whether locally deployed Large Language Models (LLMs) can effectively anonymize German-language financial text. The target domain is Swiss banking customer notes, and the central constraint is that no customer data may leave the local machine. Every evaluated configuration is therefore either on-premises or, in the single external-API baseline, explicitly flagged as off-premises.

The thesis benchmarks **twenty in-house anonymization pipeline configurations plus one external API baseline**, across five categories:

- **Classical baselines:** spaCy + Regex, BERT (pretrained) + Regex, Microsoft Presidio.
- **Fine-tuned encoder:** BERT (token classification).
- **LLM tag-and-replace:** Llama-3, Qwen2.5, and SauerkrautLM, each in zero-shot, few-shot, and few-shot with self-verification.
- **LLM fine-tuned (QLoRA):** Llama-3 8B.
- **LLM prompt-based rewriting:** Llama-3, Qwen2.5, and SauerkrautLM, each in zero-shot and few-shot.

Plus one external API baseline.

All in-house experiments run on a single **NVIDIA RTX 4090 (24 GB VRAM)**.

---

## Headline results

Per-pipeline summary on the 638-document test set. Overall F1 uses strict span-and-label matching. PII leak rate is the proportion of ground-truth PII strings still visible in the output (lower is better). Readability and Meaning are LLM-judge ratings on a 1-10 scale (higher is better).

| Pipeline                                | Overall F1 | PII leak rate | Masked BERTScore | Readability | Meaning |
|-----------------------------------------|-----------:|--------------:|-----------------:|------------:|--------:|
| Llama-3 [fine-tuned, QLoRA]             |  **0.969** |          1.6% |        **0.995** |        8.78 | **9.40** |
| BERT Fine-Tuned                         |      0.962 |      **1.3%** |        **0.995** |        8.68 |    9.31 |
| SauerkrautLM [few-shot + verify]        |      0.890 |          8.3% |            0.983 |        8.10 |    8.89 |
| Llama-3 [few-shot + verify]             |      0.866 |          6.6% |            0.982 |        7.86 |    8.67 |
| Qwen2.5 [few-shot + verify]             |      0.832 |         21.3% |            0.969 |        8.10 |    9.11 |
| Microsoft Presidio                      |      0.813 |          7.1% |            0.970 |        6.85 |    7.41 |
| BERT (pretrained) + Regex               |      0.760 |         16.9% |            0.967 |        7.95 |    8.35 |
| spaCy + Regex                           |      0.735 |         21.0% |            0.957 |        7.16 |    7.63 |
| Llama-3 [prompt rewrite]                |        n/a |          5.2% |            0.864 |        8.72 |    5.91 |
| SauerkrautLM [prompt rewrite]           |        n/a |          7.2% |            0.866 |    **9.02** |    6.18 |
| Qwen2.5 [prompt rewrite]                |        n/a |         19.2% |            0.882 |        8.59 |    6.36 |
| Anonymizer API                          |        n/a |         37.6% |            0.837 |        6.89 |    8.24 |

Per-category, per-tier, and per-pipeline numbers, along with hallucination rate and inference-attack results, are in `results/` and Chapter 5 of the thesis.

---

## Repository layout

```
thesis/
|-- data/                          Synthetic dataset (2,542 German banking notes)
|   |-- raw/                       Pre-validation Gemini output
|   |-- processed/                 Validated dataset and Label Studio import format
|   |-- label_studio/              Annotated export with PII spans (canonical dataset)
|   |-- splits/                    Train / dev / test splits used by every pipeline
|
|-- src/                           Experimental code
|   |-- data_generation/
|   |   |-- generate_data.py                  Gemini-based synthetic-note generator
|   |   |-- validate_data.py                  Two-stage rule + LLM-judge validation
|   |   |-- label_studio_transformation.py    Reformat for Label Studio import
|   |   |-- compute_rejection_rates.py        Validation rejection-rate breakdown
|   |   |-- export_data_splits.py             Stratified train/dev/test splits
|   |
|   |-- anonymization/             Twenty in-house pipelines + the external API baseline
|   |   |-- classical_baseline.py             spaCy and BERT pretrained, both with regex
|   |   |-- presidio_baseline.py              Microsoft Presidio with custom recognizers
|   |   |-- bert_finetune_ner.py              Token-classification fine-tune
|   |   |-- llm_tag_and_replace.py            Zero-shot, few-shot, few-shot + verify
|   |   |-- llm_prompt_anonymize.py           Prompt-based rewriting
|   |   |-- llm_finetune_ner.py               QLoRA fine-tune of Llama-3 8B
|   |   |-- evaluation_utils.py               Strict / relaxed matching, P/R/F1
|   |
|   |-- metrics/                   Evaluation
|       |-- semantic_preservation.py     Masked BERTScore, full BERTScore, ROUGE-1, BLEU-4
|       |-- llm_judge_gemini.py          Gemini judge: readability, meaning, hallucination,
|       |                                anonymization quality, re-identification risk,
|       |                                and the seven-attribute inference attack
|       |-- leakage_breakdown.py         PII leakage rate per category
|       |-- classify_ner_misses.py       Boundary / label / omission split of recall gap
|       |-- attack_deep_dive_dump.py     Adversarial-attack deep-dive analysis
|       |-- dataset_stats.py             Per-category and per-split entity counts
|       |-- evaluation_utils.py          Shared with anonymization/
|
|-- results/                       Per-pipeline outputs (JSON + human-readable reports)
|   |-- classical_baselines/
|   |-- presidio_baseline/
|   |-- bert_finetuned/
|   |-- llm_baselines/
|   |-- llm_finetuned/              (model checkpoints excluded; reproducible from scripts)
|   |-- llm_prompt_anonymize/
|   |-- semantic_preservation/
|   |-- llm_judge_gemini/
|   |-- leakage_breakdown/
|   |-- miss_classification/
|   |-- dataset_stats/
|
|-- demo/                          Web-based demonstrator (Flask)
|   |-- app.py
|   |-- templates/
|   |-- requirements.txt
|   |-- README.md                  Standalone install + run guide
|
|-- requirements.txt               Pinned Python dependencies
|-- setup.bat                      One-click Windows setup
|-- .env.example                   Template for API-key configuration
|-- .gitignore
|-- README.md                      (this file)
```

---

## Installation

### Prerequisites

| Requirement                                                | Why |
|------------------------------------------------------------|-----|
| Python 3.10 or newer                                       | Required by `transformers` and `peft`. |
| ~50 GB free disk                                           | Models, datasets, and results. |
| NVIDIA GPU with at least 16 GB VRAM (RTX 4090 used here)   | Required for fine-tuning and prompt-based LLM experiments. |
| Gemini API key                                             | Optional; only needed to regenerate the synthetic dataset or to re-run the LLM-judge step. |
| Hugging Face account with Llama-3 access                   | Required to download Llama-3 8B (gated). |

CPU-only setups can run the regex and pretrained-BERT pipelines but cannot reproduce the LLM experiments.

### Quick setup (Windows)

```bash
git clone https://github.com/<your-account>/thesis.git
cd thesis
setup.bat
```

`setup.bat` checks the Python version, creates a virtual environment in `venv/`, and installs everything in `requirements.txt`.

### Manual setup (any OS)

```bash
git clone https://github.com/<your-account>/thesis.git
cd thesis

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with whichever keys you need:

```ini
# Required only for synthetic-data regeneration and the LLM-judge stage
GEMINI_API_KEY=your_gemini_key_here
```

The external Anonymizer-API baseline requires credentials that were provided privately by the thesis supervisor; the service is not publicly available, so that one row of the results cannot be reproduced from this repository alone.

Authenticate with Hugging Face once (needed for Llama-3, which is gated):

```bash
huggingface-cli login
# request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Reproducing the thesis results

Every script under `src/` has a clearly marked configuration block at the top. Run from the repository root with the venv active. The pre-built dataset is included under `data/`, so step 1 can be skipped.

### 1. (Optional) regenerate the synthetic dataset

```bash
python src/data_generation/generate_data.py            # ~1 hour, uses Gemini API
python src/data_generation/validate_data.py            # rule + LLM-judge audit
python src/data_generation/export_data_splits.py
```

### 2. Run the classical and fine-tuned baselines

```bash
python src/anonymization/classical_baseline.py        # spaCy + Regex, BERT pretrained + Regex
python src/anonymization/presidio_baseline.py         # Microsoft Presidio
python src/anonymization/bert_finetune_ner.py         # ~30 s on RTX 4090
python src/anonymization/llm_finetune_ner.py          # ~15 h QLoRA fine-tune
```

### 3. Run the prompted LLM pipelines

```bash
python src/anonymization/llm_tag_and_replace.py       # 9 configurations across 3 models
python src/anonymization/llm_prompt_anonymize.py      # 6 prompt-based rewriting configs
```

### 4. Compute multi-dimensional metrics

```bash
python src/metrics/semantic_preservation.py           # BERTScore (masked, full), ROUGE-1, BLEU-4
python src/metrics/llm_judge_gemini.py                # Gemini readability, meaning, hallucination,
                                                      # AnonQ, ReID, inference attack
python src/metrics/leakage_breakdown.py               # per-category PII leakage
```

---

## Web-based demonstrator

A standalone Flask app under `demo/` exposes five pipelines side by side: regex, BERT (pretrained), Llama-3 tag-and-replace, Llama-3 rewrite, and a cascading hybrid (regex then BERT then LLM). Paste a German banking note in the browser, click Run All, and compare the outputs.

```bash
cd demo
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

See [demo/README.md](demo/README.md) for the full install guide, hardware requirements, and configuration options. The cascade is presented as a candidate production architecture: structured PII goes to fast deterministic detectors, and the LLM is reserved for the Tier-3 quasi-identifiers that classical methods miss.

---

## License

- **Code** (`src/`, `demo/`): [MIT](LICENSE).
- **Synthetic dataset** (`data/`): [CC BY 4.0](data/LICENSE).
