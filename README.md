# RFP Automation (Gemini 2.5 + LlamaIndex)

A Streamlit app and script to parse an RFP PDF, extract required sections/questions, retrieve answers from your context PDFs, and generate a final markdown response.

## Features
- Gemini 2.5 LLM for reasoning and generation
- LlamaParse for robust PDF parsing
- Chroma vector DB + FastEmbed embeddings for local, fast retrieval
- Streamlit UI with live progress, resume support, rate-limit backoff
- Persists questions and incremental answers to disk

## Setup
1. Create `.env` at repo root (do not commit this):
```ini
GOOGLE_API_KEY=YOUR_GEMINI_KEY
LLAMA_CLOUD_API_KEY=YOUR_LLAMA_CLOUD_KEY
```
2. Create and use a virtual environment:
```bash
python -m venv .venv
# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
# Or run without activating:
# .\.venv\Scripts\python.exe -m pip install -r requirements.txt
pip install -r requirements.txt
```

## Run the Streamlit UI
```bash
# If activated
streamlit run RFP_Automation/streamlit_app.py
# Or, without activation
.\.venv\Scripts\python.exe -m streamlit run RFP_Automation/streamlit_app.py
```
Then upload:
- RFP template PDF
- One or more context PDFs

Outputs are saved under `data_out_rfp_ui/workflow_output/`:
- `all_keys.txt` — extracted questions
- `combined_answers.jsonl` — appended per question (resume-safe)
- `final_output.md` — final generated response

## Script (notebook-style) version
See `RFP_Automation/copy_of_generate_rfp.py`. This uses notebook magics and async awaits and is best run in a notebook environment.

## Notes
- Free-tier rate limits may cause 429s. The UI paces requests and will resume from where it left off.
- To speed up parsing, set these to `False` in `LlamaParse(...)`: `high_res_ocr`, `outlined_table_extraction`, `output_tables_as_HTML`.

## License
MIT
