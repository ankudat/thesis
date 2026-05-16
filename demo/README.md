# German PII Anonymizer — Web Demo

Five-pipeline web demo derived from the master's thesis
*"How well do LLMs anonymize text data?"* (HSLU, 2026).

Paste a German banking note in the browser, click **Run All**, and compare
five anonymization approaches side by side:

1. **Regex** — structured PII (IBAN, EMAIL, PHONE, DATE, MONEY)
2. **BERT (pretrained)** — `fhswf/bert_de_ner` for PER/LOC/ORG, with regex layered on top
3. **Llama-3 8B — Tag & Replace** — entity-level masking with `[PER]`, `[IBAN]`, etc.
4. **Llama-3 8B — Rewrite** — fluent paraphrase that removes PII
5. **Cascade** — regex → BERT → Llama-3 in sequence; earlier stages win on overlap

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/<your-account>/thesis.git
cd thesis/demo

# 2. Create a virtual environment (Python 3.10+ required)
python -m venv venv

# 3. Activate it
#    Windows (PowerShell):
.\venv\Scripts\Activate.ps1
#    Windows (CMD):
venv\Scripts\activate.bat
#    Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Required for the LLM pipelines only) authenticate with Hugging Face
#    Llama-3 is a gated model — see "Hugging Face access" below.
huggingface-cli login

# 6. Run the demo
python app.py
```

Open <http://127.0.0.1:5000> in a browser.

The first request to each model triggers a download:
* BERT: ~440 MB, ~10 s after download.
* Llama-3 8B: ~16 GB, several minutes the first time. Cached afterwards in `~/.cache/huggingface/hub/`.

---

## Hardware requirements

| Pipeline | GPU? | VRAM | Notes |
|---|---|---|---|
| Regex | no | — | Pure CPU, sub-millisecond |
| BERT | no (recommended) | — | CPU works; ~50 ms/doc |
| Llama-3 Tag / Rewrite / Cascade | **yes** | 16 GB (fp16) or 6 GB (4-bit) | NVIDIA GPU with CUDA. CPU inference is technically possible but takes minutes per request. |

**No GPU?** You can still run the regex and BERT pipelines. The three LLM-based panels will show an out-of-memory error and the others will work normally.

**Tight on VRAM?** Open `app.py` and set:

```python
QUANTIZE_LLM_4BIT = True
```

This uses 4-bit quantization (requires the `bitsandbytes` package, which is in `requirements.txt` for non-macOS systems).

---

## Hugging Face access

The Llama-3 model is **gated**. To download it:

1. Create a free account at <https://huggingface.co/>.
2. Visit <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct> and click *"Request access"*. Approval is usually instant.
3. Generate a personal access token at <https://huggingface.co/settings/tokens> (read-only is enough).
4. Run `huggingface-cli login` and paste the token.

If you skip this step, the regex and BERT pipelines still work. The three LLM-based panels will show a `Repository not found` or `401 Unauthorized` error.

---

## Configuration

Edit the **USER SETTINGS** block at the top of `app.py`:

```python
LLM_MODEL_NAME      = "meta-llama/Meta-Llama-3-8B-Instruct"
BERT_MODEL_NAME     = "fhswf/bert_de_ner"
QUANTIZE_LLM_4BIT   = False     # set True for ~6 GB VRAM mode
HOST                = "127.0.0.1"
PORT                = 5000
```

Replace `LLM_MODEL_NAME` with any chat-tuned LLM available on Hugging Face — the prompt is generic. For German-specialised performance try `VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct`.

---

## Project layout

```
demo/
├── app.py                  Flask server + 5 detection pipelines
├── templates/
│   └── index.html          Single-page UI (HTML + CSS + JS inline)
├── requirements.txt
└── README.md
```

All logic for the five pipelines lives in `app.py` so the project is self-contained. Models load lazily on first request, so the server starts in <1 s.

---

## License

MIT License. See the [`LICENSE`](../LICENSE) file at the repository root.

---
