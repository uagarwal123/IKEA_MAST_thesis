# agent-evaluation-framework

This is the repo for our bachelor thesis project (UvA × IKEA). We are researching failure modes in multi-agent system (MAS) traces, using the [MAD dataset](https://huggingface.co/datasets/mcemri/MAST-Data) and the MAST taxonomy of 14 failure modes.

## Quickstart

```bash
# 1. Clone the repository
git clone <repo>
cd agent-evaluation-framework

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MAD datasets (not shipped in the repo — required before running anything)
python -c "
from huggingface_hub import hf_hub_download
import os, shutil
os.makedirs('data/MAST-Data', exist_ok=True)
for fn in ['MAD_full_dataset.json', 'MAD_human_labelled_dataset.json']:
    p = hf_hub_download(repo_id='mcemri/MAD', filename=fn, repo_type='dataset')
    shutil.copy(os.path.realpath(p), os.path.join('data/MAST-Data', fn))
print('done')
"

# 5. Authenticate for Vertex AI models
gcloud auth application-default login

# 6. Configure an experiment
cp -r experiments/stage1_llm_judge/experiments_template \
      experiments/stage1_llm_judge/my_experiment

# Edit config.yaml

# 7. Run the judge notebook FROM your experiment folder (not the repo root)
jupyter notebook experiments/stage1_llm_judge/my_experiment/llm_judge_pipeline.ipynb
```
## Setup

```bash
git clone <repo>
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt
```

### Download the MAD datasets

The repository does not ship the MAD dataset files. After installing the requirements, download the data from Hugging Face:

```python
from huggingface_hub import hf_hub_download
import os
import shutil

os.makedirs("data/MAST-Data", exist_ok=True)

for fn in ["MAD_full_dataset.json", "MAD_human_labelled_dataset.json"]:
    p = hf_hub_download(repo_id="mcemri/MAD", filename=fn, repo_type="dataset")
    shutil.copy(os.path.realpath(p), os.path.join("data/MAST-Data", fn))
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
3. Open `llm_judge_pipeline.ipynb` **inside your experiment folder** (e.g. `experiments/stage1_llm_judge/my_experiment/`) and run top to bottom. Do not use the notebook at the repo root.

The notebook runs all experiments in the config sequentially. Results are written to `saved_results/`:

| File | What it contains |
|---|---|
| `predictions.csv` | Per-trace, per-failure-mode predictions for all experiments |
| `metrics_per_mode.csv` | Precision, recall, F1 per failure mode per experiment |
| `summary.csv` | Aggregated metrics per experiment |
| `checkpoints/<name>.pkl` | Checkpoint after every trace, results are not lost if the run crashes |

The template notebook contains a detailed explanation of each step, including input/output descriptions and config options.

## Building the comparison table (baseline results)

After running one or more experiments, build the final comparison table and figure:

```bash
jupyter notebook experiments/stage1_llm_judge/run_50_baseline/build_comparison_table.ipynb
```

This notebook collects `predictions.csv` across experiment folders, reconstructs ground truth from the dataset, and produces:

| Output | What it contains |
|---|---|
| `comparison_table.csv` | One row per (model × shots) config; per-mode F1 for all 14 failure modes (long→wide), macro/micro F1, Cohen's kappa, 95% bootstrap CIs (resampled over traces) for macro F1 and kappa, total/mean cost (USD), mean latency per trace |
| `comparison_figure.png` | Judge quality (macro F1) vs mean cost per trace, with bootstrap CI error bars |

All cloud configs are run on the same 30-trace stratified slice (seed = 42) so rows are comparable.

## Ollama (local models)

An Ollama experiment is pre-configured in `experiments/stage1_llm_judge/run_30_ollama/` (config + price-table entries in place). To run local models, start Ollama, `ollama pull <model>`, set the model name in that folder's `config.yaml`, and run the notebook.

**Note:** small local models (e.g. 3B) do not reliably follow the required structured output format, so their predictions fail to parse and metrics collapse to zero. This is a known limitation, not a bug — larger models are expected to perform better.