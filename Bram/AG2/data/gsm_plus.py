from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path

SEED = 42


def create_sample(n: int) -> Path:
    output = Path(__file__).parent / "samples" / f"gsm_plus_n{n}_seed{SEED}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Sample already exists: {output}")
        return output

    ds = load_dataset("qintongli/GSM-Plus", split="testmini")
    df = ds.to_pandas()

    groups = df["perturbation_type"].value_counts()
    n_groups = len(groups)
    per_group = n // n_groups
    remainder = n % n_groups

    samples = []
    for i, (ptype, count) in enumerate(groups.items()):
        k = per_group + (1 if i < remainder else 0)
        samples.append(df[df["perturbation_type"] == ptype].sample(n=min(k, count), random_state=SEED))

    sample = pd.concat(samples).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample = sample.drop(columns=["seed_question", "seed_solution", "seed_answer"], errors="ignore")

    records = sample.to_dict(orient="records")
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} samples to {output}")
    print(sample["perturbation_type"].value_counts().to_string())
    return output


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    create_sample(n)
