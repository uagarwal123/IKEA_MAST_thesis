# IKEA MAST 

This repository contains the shared foundation developed by the group for our theses on
multi-agent system (MAS) failure detection. It includes two parts: the
[MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST)-based
**LLM-as-a-Judge pipeline** for detecting failure modes in MAS traces (adapted from the
original MAST repository and evaluated against the MAD dataset published there); 
and the **initial data analysis of the MAST traces** (`data_understanding/`), exploring the dataset's failure-mode
distributions and characteristics.

## Setup

```bash
pip install -r requirements.txt
```

Running the notebooks requires authentication to Google Cloud (GCP), as the judge calls
Claude and Gemini models hosted on Vertex AI.

## What's in this repo

- **`llm_interface.py`** — the core judge. Wraps multiple model backends (Anthropic and
  Gemini via Vertex AI, UvA AI Chat, local Ollama) behind one call. Builds the judge
  prompt from the MAST definitions and few-shot examples, sends a trace to the model, and
  parses the response into binary verdicts for all 14 failure modes, with token cost and
  latency.

- **`experiments/stage1_llm_judge/`** — the experiment, run as two notebooks in order:
  - **(1) judge run** — runs the judge configurations defined in `config.yaml` over the
    MAD trace dataset, evaluates predictions against the annotations, and saves per-trace
    predictions, per-mode metrics, and a summary to `saved_results/` (with checkpointing so
    interrupted runs resume).
  - **(2) analysis** — reads those saved CSVs and builds the cross-model comparison table
    (per-mode F1, macro/micro F1, kappa with bootstrapped CIs, cost, latency) and a
    quality-vs-cost figure comparing the judge configurations.

- **`prompts/`** — the failure-mode definitions and few-shot examples fed to the judge.

- **`parsers/`** — utilities for parsing raw model output into structured verdicts.

- **`data_understanding/`** — exploratory analysis of the MAD trace dataset.

## Data

Traces and annotations come from the MAD dataset released with MAST
(`mcemri/MAD` on Hugging Face), downloaded automatically by the notebook.
