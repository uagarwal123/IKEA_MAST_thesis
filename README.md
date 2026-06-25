# IKEA MAST 

This repository contains the shared foundation developed by the group for our theses on
multi-agent system (MAS) failure detection. It includes three parts: the
[MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST)-based
**LLM-as-a-Judge pipeline** for detecting failure modes in MAS traces (adapted from the
original MAST repository and evaluated against the MAD dataset published there); the
**AG2 multi-agent system setup** quick multi agent system setup; and the **initial data analysis
of the MAST traces** (`data_understanding/`), exploring the dataset's failure-mode
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

- **`experiments/stage1_llm_judge/`** — the main benchmarking notebook. Runs one or more
  judge configurations (defined in `config.yaml`) over the MAD trace dataset and evaluates
  predictions against the annotations, reporting accuracy, precision, recall, F1, and
  Cohen's kappa per failure mode, with bootstrapped confidence intervals. Outputs are saved
  to `saved_results/` (predictions, per-mode metrics, summary), with checkpointing so
  interrupted runs resume.

- **`prompts/`** — the failure-mode definitions and few-shot examples fed to the judge.

- **`parsers/`** — utilities for parsing raw model output into structured verdicts.

- **`data_understanding/`** — exploratory analysis of the MAD trace dataset.

## Data

Traces and annotations come from the MAD dataset released with MAST
(`mcemri/MAD` on Hugging Face), downloaded automatically by the notebook.
