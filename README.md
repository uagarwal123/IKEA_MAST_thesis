# agent-evaluation-framework

This is the repo for our bachelor thesis project (UvA × IKEA). We are researching failure modes in multi-agent system (MAS) traces, using the [MAD dataset](https://huggingface.co/datasets/mcemri/MAST-Data) and the MAST taxonomy of 14 failure modes.

## Setup

```bash
git clone https://github.com/uagarwal123/agent-evaluation-framework.git
cd agent-evaluation-framework
python -m venv .venv

# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

**Credentials** depend on which backend you use:

| Backend | What you need |
|---|---|
| `anthropic` | `gcloud auth application-default login`  Claude is served via Vertex AI |
| `genai` | `gcloud auth application-default login`  Gemini via Vertex AI |
| `ollama` | Ollama running at `http://localhost:11434` |


## Data understanding

Three notebooks cover dataset exploration. Read them in this order:

| Notebook | What it covers |
|---|---|
| `data_understanding/general_eda/eda.ipynb` | Traces per framework, FM prevalence and co-occurrence, token and step-length distributions |
| `data_understanding/fm_1_3_analysis/fm13_token_length_analysis.ipynb` | FM-1.3 (Step Repetition) deep-dive: does token length predict this failure mode? |
| `data_understanding.ipynb` | Unifies all 7 parser outputs into a shared schema; lets you inspect and export traces for any failure mode to a readable markdown file |

The first two notebooks read directly from `data/MAST-Data/MAD_full_dataset.json`. The third requires the parser output JSON files (see below).

## Running the parsers

Each parser is a standalone script. Run from the repo root:

```bash
python parsers/ag2_parser/ag2_parser.py
python parsers/appworld_parser/appworld_parser.py
python parsers/chatdev_parser/chatdev_parser.py
python parsers/hyperagent_parser/hyperagent_parser.py
python parsers/magenticone_parser/magenticone_parser.py
python parsers/metagpt_parser/metagpt_parser.py
python parsers/openmanus_parser/openmanus_parser.py
```

Output is written as JSON next to each parser (e.g. `parsers/ag2_parser/ag2_output_mad.json`). These files are required by `data_understanding.ipynb`.

## Running the LLM-as-a-Judge

1. Copy `experiments/stage1_llm_judge/experiments_template/` to a new folder, e.g. `experiment_2/`
2. Edit `config.yaml`  define one or more experiments under the `experiments:` key. Each entry sets `model`, `backend`, `shots`, `slice_n`, etc.
3. Open `llm_judge_pipeline.ipynb` and run top to bottom

The notebook runs all experiments in the config sequentially. Results are written to `saved_results/`:

| File | What it contains |
|---|---|
| `predictions.csv` | Per-trace, per-failure-mode predictions for all experiments |
| `metrics_per_mode.csv` | Precision, recall, F1 per failure mode per experiment |
| `summary.csv` | Aggregated metrics per experiment |
| `checkpoints/<name>.pkl` | Checkpoint after every trace, results are not lost if the run crashes |

The template notebook contains a detailed explanation of each step, including input/output descriptions and config options.