from datasets import load_dataset
import pandas as pd
import numpy as np
import json
from pathlib import Path

SEED = 42
DATASET_NAME = "Hothan/OlympiadBench"
DATASET_CONFIG = "OE_TO_maths_en_COMP"


def create_sample(n: int, seed: int = SEED) -> Path:
    output = Path(__file__).parent / "samples" / f"olympiad_n{n}_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Sample already exists: {output}")
        return output

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    df = ds.to_pandas()

    df = df.rename(columns={"problem": "question", "subfield": "category"})

    def extract_answer(ans):
        if isinstance(ans, np.ndarray):
            ans = ans.tolist()
        if isinstance(ans, list):
            return "; ".join(str(a).strip("$").strip() for a in ans)
        return str(ans).strip("$").strip()

    df["answer"] = df["final_answer"].apply(extract_answer)

    groups = df["category"].value_counts()
    n_groups = len(groups)
    per_group = n // n_groups
    remainder = n % n_groups

    samples = []
    for i, (cat, count) in enumerate(groups.items()):
        k = per_group + (1 if i < remainder else 0)
        if k > 0:
            samples.append(
                df[df["category"] == cat].sample(n=min(k, count), random_state=seed)
            )

    sample = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)

    keep = [c for c in ["id", "question", "answer", "category", "solution", "source", "difficulty"] if c in sample.columns]
    records = json.loads(sample[keep].to_json(orient="records", force_ascii=False))

    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} samples to {output}")
    print(sample["category"].value_counts().to_string())
    return output


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    create_sample(n)
