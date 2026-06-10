import sys
import json
import pickle
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "LLM_models_interface"))
from llm_interface import LLMJudge, load_configs, FAILURE_MODES  # type: ignore

import pandas as pd

# ── configuration ────────────────────────────────────────────────────────────
RUN_DIR    = Path("Bram/AG2/results/baseline_olympiad_gpt41_n50_20260609")
CONFIG_PATH = Path(__file__).parent / "config.yaml"
# ─────────────────────────────────────────────────────────────────────────────

def main():
    configs = load_configs(str(CONFIG_PATH))

    with open(RUN_DIR / "raw_traces.json") as f:
        raw_traces = json.load(f)
    with open(RUN_DIR / "parsed_traces.json") as f:
        parsed_traces = json.load(f)
    traces = list(zip(raw_traces, parsed_traces))

    out_dir = RUN_DIR / "saved_results"
    os.makedirs(out_dir / "checkpoints", exist_ok=True)

    all_predictions = []
    summary_rows = []

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Experiment: {cfg.name} — {cfg.model}")
        print(f"{'='*60}")

        judge = LLMJudge(cfg)
        checkpoint_path = out_dir / "checkpoints" / f"{cfg.name}.pkl"

        results = []
        for i, (raw, parsed) in enumerate(traces):
            trace_id   = parsed["metadata"]["trace_id"]
            trace_text = json.dumps(raw)

            if len(trace_text) + len(judge.examples) > 1_048_570:
                trace_text = trace_text[:1_048_570 - len(judge.examples)]

            try:
                response = judge.judge_trace(trace_id, trace_text)
                results.append(response)
                print(f"  {len(results)}/{len(traces)}")

                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)

                if len(results) % 10 == 0:
                    backup = out_dir / "checkpoints" / f"{cfg.name}_backup_{len(results)}.pkl"
                    with open(backup, "wb") as f:
                        pickle.dump(results, f)

            except Exception as e:
                print(f"Error on trace {i} ({trace_id}): {e}")
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)

        print(f"Progress: {len(results)}/{len(traces)} traces completed")

        rows = [
            {
                "trace_id":   r.trace_id,
                "model":      r.model_id,
                "name":       cfg.name,
                "tokens_in":  r.tokens_in,
                "tokens_out": r.tokens_out,
                "latency_s":  r.latency_s,
                "cost_usd":   r.cost_usd,
                **r.annotations,
            }
            for r in results
        ]
        all_predictions.extend(rows)

        if results:
            summary_rows.append({
                "name":             cfg.name,
                "model":            cfg.model,
                "total_cost_usd":   sum(r.cost_usd   for r in results),
                "mean_cost_usd":    sum(r.cost_usd   for r in results) / len(results),
                "total_latency_s":  sum(r.latency_s  for r in results),
                "mean_latency_s":   sum(r.latency_s  for r in results) / len(results),
            })

    pd.DataFrame(all_predictions).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved predictions.csv and summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
