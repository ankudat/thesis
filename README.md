# How well do LLMs anonymize text data?

### Master's Thesis, Hochschule Luzern (HSLU)

**Programme:** MSc in Applied Information and Data Science
**Submission:** 2026

---

## Overview

This repository contains the code, synthetic dataset, and experimental results for a Master's thesis investigating whether locally deployed Large Language Models (LLMs) can effectively anonymize German-language financial text. The target domain is Swiss banking customer notes, and the central constraint is that no customer data may leave the local machine. Every evaluated configuration is therefore either on-premises or, in the single external-API baseline, explicitly flagged as off-premises.

The thesis benchmarks **twenty in-house anonymization pipeline configurations plus one external API baseline**, across three method families:

- **Classical:** spaCy + Regex, BERT (pretrained) + Regex, Microsoft Presidio.
- **Learned token classifiers:** a fine-tuned BERT encoder, and a Llama-3 8B fine-tuned with QLoRA.
- **Prompted LLMs:** Llama-3, Qwen 2.5, and SauerkrautLM, evaluated in zero-shot, few-shot, and few-shot with self-verification, in two anonymization paradigms (tag-and-replace and prompt-based rewriting).

All in-house experiments run on a single **NVIDIA RTX 4090 (24 GB VRAM)**. The practical claim is that on-premises deployment is realistic for regulated industries that cannot use cloud LLM APIs.

---

## Headline results

PII detection (overall F1 on 638 test documents, strict span-and-label matching):

| Pipeline                                | Overall F1 | Tier-3 F1 (quasi-IDs) | Masked BERTScore | PII leak rate |
|-----------------------------------------|-----------:|----------------------:|-----------------:|--------------:|
| Llama-3 [fine-tuned, QLoRA]             |  **0.969** |             **0.918** |        **0.995** |      **1.6%** |
| BERT Fine-Tuned                         |      0.962 |                 0.911 |            0.995 |          1.3% |
| SauerkrautLM [few-shot + verify]        |      0.890 |                 0.848 |            0.983 |          8.3% |
| Llama-3 [few-shot + verify]             |      0.866 |                 0.842 |            0.982 |          6.6% |
| Qwen2.5 [few-shot + verify]             |      0.832 |                 0.758 |            0.969 |         21.3% |
| Microsoft Presidio                      |      0.813 |                 0.740 |            0.970 |          7.1% |
| BERT (pretrained) + Regex               |      0.760 |             **0.000** |            0.967 |         16.9% |
| spaCy + Regex                           |      0.735 |             **0.000** |            0.957 |         21.0% |
| Llama-3 [prompt rewrite]                |        n/a |                   n/a |          0.864 * |          5.2% |
| SauerkrautLM [prompt rewrite]           |        n/a |                   n/a |          0.866 * |          7.2% |
| Qwen2.5 [prompt rewrite]                |        n/a |                   n/a |          0.882 * |         19.2% |
| Anonymizer API (external, classical)    |        n/a |                   n/a |            0.837 |         37.6% |

\* The prompt-rewriting pipelines change non-PII wording as well as PII, so their masked BERTScore is not directly comparable to the tag-and-replace pipelines above; their primary utility metric in the thesis is the LLM-judge meaning score (5.91-6.36 for the three pipelines, well below the 9.31-9.40 of the fine-tuned tag-and-replace pipelines).

Per-category, per-tier, and per-pipeline numbers, along with semantic preservation, hallucination, readability, meaning preservation, and inference-attack results, are in `results/` and Chapter 5 of the thesis.

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
| Gemini API key (free tier OK)                              | Optional; only needed to regenerate the synthetic dataset or to re-run the LLM-judge step. |
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

## Notes on AI usage and data handling

- The synthetic dataset is generated by **Google Gemini 2.5 Pro**, to avoid using real customer data.
- Generated notes go through a deterministic rule pass and a second LLM-based audit before being added to the corpus. The author inspected a sample manually.
- The anonymization models themselves run **fully on-premises**; no input data leaves the local machine during inference.
- The **Anonymizer API** baseline (a third-party commercial service) is the only experiment that sends data over the network, and is included strictly for comparison. Access to this service was provided privately for the thesis and is not publicly available, so the corresponding row in the results table cannot be reproduced by external readers.

---

## License

- **Code** (`src/`, `demo/`): MIT License. See [`LICENSE`](LICENSE).
- **Synthetic dataset** (`data/`): Creative Commons Attribution 4.0 International (CC BY 4.0). See [`data/LICENSE`](data/LICENSE). Free to share and adapt with attribution.

Third-party model weights (Llama-3, Qwen 2.5, SauerkrautLM, BERT-de-NER, spaCy `de_core_news_lg`) remain subject to their respective licenses and are not redistributed by this repository.
