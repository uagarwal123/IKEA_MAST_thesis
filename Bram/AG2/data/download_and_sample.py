from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path

N = 1
OUTPUT = Path(__file__).parent / "gsm_plus_sample.json"

ds = load_dataset("qintongli/GSM-Plus", split="testmini")
df = ds.to_pandas()

groups = df["perturbation_type"].value_counts()
n_groups = len(groups)
per_group = N // n_groups
remainder = N % n_groups

samples = []
for i, (ptype, count) in enumerate(groups.items()):
    k = per_group + (1 if i < remainder else 0)
    samples.append(df[df["perturbation_type"] == ptype].sample(n=min(k, count), random_state=42))

sample = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
sample = sample.drop(columns=["seed_question", "seed_solution", "seed_answer"], errors="ignore")

records = sample.to_dict(orient="records")
OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {len(records)} samples to {OUTPUT}")
print(sample["perturbation_type"].value_counts().to_string())
