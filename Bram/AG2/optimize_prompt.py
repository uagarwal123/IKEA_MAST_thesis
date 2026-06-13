"""Iterative prompt optimization loop for AG2 MathChat on OlympiadBench.

Orchestrates the existing run (run.py) and judge (judge/judge.py) pipelines:
  1. run MathChat on a fixed random subset of N tasks with the current prompt
  2. judge all traces for the 14 MAST failure modes
  3. log FM prevalence, accuracy, cost and tokens to metrics.json
  4. stop on convergence (total FM prevalence change < threshold) or max iterations
  5. otherwise ask gpt-4.1 for an improved prompt and repeat

Output layout (under optimization_config.yaml's output_dir):
  iteration_0/
    prompt.txt              prompt used this iteration
    raw_traces.json         from run.py (plus parsed_traces.json, summary.csv, code/)
    judge_results/          predictions.csv, summary.csv, checkpoints/
    metrics.json            FM prevalence (14 modes), accuracy, cost, tokens
    prompt_generation.json  metadata of the LLM call that produced the next prompt
  iteration_1/ ...
  summary.json              all iterations side by side

Resumable: existing iteration dirs are reused (run.py resumes via summary.csv,
the judge via its checkpoints, finished iterations via metrics.json).
"""

import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from data.olympiad import create_sample as create_olympiad_sample

load_dotenv()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# judge/ has no __init__.py and run.py shadows common names, so load by path
run_module = _load_module("ag2_run", BASE_DIR / "run.py")
judge_module = _load_module("ag2_judge", BASE_DIR / "judge" / "judge.py")
FAILURE_MODES = judge_module.FAILURE_MODES

FM_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No or Incorrect Verification",
    "3.3": "Weak Verification",
}

# (price_in, price_out) per 1M tokens, for the prompt-generator call
PRICES_PER_MTOK = {"gpt-4.1": (2.00, 8.00)}


def _read_predictions(it_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(it_dir / "judge_results" / "predictions.csv")
    # restrict to the first judge experiment in case the config lists several
    return pred[pred["name"] == pred["name"].iloc[0]]


def compute_metrics(it_dir: Path, iteration: int) -> dict:
    pred = _read_predictions(it_dir)
    prevalence = {m: float(pred[m].mean()) for m in FAILURE_MODES}
    total = sum(prevalence.values()) / len(prevalence)

    run_summary = pd.read_csv(it_dir / "summary.csv")
    accuracy = float((run_summary["correct"].astype(str) == "True").mean())

    def _sum(df, col):
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    metrics = {
        "iteration": iteration,
        "n_tasks_run": int(len(run_summary)),
        "n_traces_judged": int(len(pred)),
        "accuracy": accuracy,
        "fm_prevalence": prevalence,
        "total_fm_prevalence": total,
        "run_tokens_in": int(_sum(run_summary, "tokens_in")),
        "run_tokens_out": int(_sum(run_summary, "tokens_out")),
        "run_cost_usd": _sum(run_summary, "cost_usd"),
        "judge_tokens_in": int(_sum(pred, "tokens_in")),
        "judge_tokens_out": int(_sum(pred, "tokens_out")),
        "judge_cost_usd": _sum(pred, "cost_usd"),
    }
    (it_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _format_trace(messages: list[dict], max_chars: int) -> str:
    parts = []
    for m in messages:
        speaker = m.get("name") or m.get("role") or "unknown"
        parts.append(f"[{speaker}]\n{(m.get('content') or '').strip()}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [trace truncated]"
    return text


def _collect_failing_examples(it_dir: Path, pred: pd.DataFrame, top_modes: list[str],
                              per_fm: int, max_chars: int) -> tuple[str, dict]:
    raw_traces = json.loads((it_dir / "raw_traces.json").read_text(encoding="utf-8"))
    parsed_traces = json.loads((it_dir / "parsed_traces.json").read_text(encoding="utf-8"))
    raw_by_id = {p["metadata"]["trace_id"]: raw_traces[i] for i, p in enumerate(parsed_traces)}

    blocks = []
    used_ids = {}
    for mode in top_modes:
        ids = pred.loc[pred[mode] == 1, "trace_id"].tolist()[:per_fm]
        used_ids[mode] = ids
        for tid in ids:
            if tid not in raw_by_id:
                continue
            blocks.append(
                f"--- Failing trace for FM-{mode} ({FM_NAMES[mode]}), trace_id={tid} ---\n"
                + _format_trace(raw_by_id[tid], max_chars)
            )
    return "\n\n".join(blocks), used_ids


def _split_reasoning(text: str) -> tuple[str, str]:
    """Separate the optimizer's <reasoning> block from the prompt that follows it."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), text[match.end():]
    # unclosed tag: treat everything up to the first blank line after it as reasoning
    match = re.search(r"<reasoning>(.*?)(\n\s*\n)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), text[match.end():]
    return "", text


def _compact_fm_table(prevalence: dict) -> str:
    nonzero = sorted(((m, p) for m, p in prevalence.items() if p > 0),
                     key=lambda kv: kv[1], reverse=True)
    if not nonzero:
        return "all 14 modes 0%"
    listed = ", ".join(f"FM-{m} {p:.1%}" for m, p in nonzero)
    return f"{listed}; remaining {14 - len(nonzero)} modes 0%"


def _build_history_section(history: list[dict], opt_dir: Path, max_entries: int = 4) -> str:
    """Compact summary of previous iterations: prompt change, FM table, objective."""
    previous = history[:-1][-max_entries:]
    blocks = []
    for m in previous:
        i = m["iteration"]
        if i == 0:
            change = "Initial prompt (baseline)."
        else:
            gen_path = opt_dir / f"iteration_{i - 1}" / "prompt_generation.json"
            change = "(no change record found)"
            if gen_path.exists():
                gen = json.loads(gen_path.read_text(encoding="utf-8"))
                change = gen.get("reasoning") or "(no reasoning logged)"
        blocks.append(
            f"Iteration {i}:\n"
            f"  Prompt change: {change}\n"
            f"  FM prevalence: {_compact_fm_table(m['fm_prevalence'])}\n"
            f"  Total FM prevalence (objective): {m['total_fm_prevalence']:.3f} | accuracy: {m['accuracy']:.0%}"
        )
    return "\n\n".join(blocks)


def _sanitize_prompt(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # the runner appends the problem statement directly after the prompt
    if not re.search(r"Problem:\s*$", text):
        text += "\nProblem:"
    return text.rstrip() + "\n"


def _build_optimizer_prompt(current_prompt: str, metrics: dict, examples_text: str,
                            top_modes: list[str], assistant_model: str,
                            history_text: str) -> str:
    ranked = sorted(metrics["fm_prevalence"].items(), key=lambda kv: kv[1], reverse=True)
    table = "\n".join(f"  FM-{m} ({FM_NAMES[m]}): {p:.1%}" for m, p in ranked)
    top_list = ", ".join(f"FM-{m} ({FM_NAMES[m]})" for m in top_modes)
    history_section = (
        "== OPTIMIZATION HISTORY (earlier iterations, oldest first) ==\n"
        f"{history_text}\n\n"
        if history_text else ""
    )
    return f"""You are optimizing the instruction prompt of a two-agent AG2 MathChat system: an AssistantAgent powered by {assistant_model} and a UserProxyAgent that executes Python code locally. The system solves olympiad-level open-ended competition math problems (OlympiadBench). The prompt below is prepended to each problem statement as the first message.

An LLM judge annotated every conversation trace with the 14 MAST failure modes. Your goal is to revise the prompt so that overall failure-mode prevalence (equally weighted across all 14 modes) goes down, without hurting answer accuracy.

== CURRENT PROMPT ==
{current_prompt}
== END CURRENT PROMPT ==

{history_section}== RESULTS THIS ITERATION (n={metrics['n_traces_judged']} tasks, accuracy {metrics['accuracy']:.0%}) ==
Failure-mode prevalence (fraction of traces exhibiting each mode, sorted descending):
{table}

== EXAMPLE FAILING TRACES for the dominant failure modes ({top_list}) ==
{examples_text}

== TASK ==
Write a revised version of the prompt that addresses the dominant failure patterns visible above. Learn from the optimization history: keep what lowered the objective, avoid repeating changes that did not help. Constraints:
- The prompt is prepended to the problem statement, so it must end with the line "Problem:".
- Keep the mechanics the harness depends on: Python code in ```python fenced blocks, 'print' for all outputs, fractions/radical forms instead of decimals, the final answer in \\boxed{{}}, and writing TERMINATE after the boxed answer.
- Keep it concise: do not exceed roughly twice the length of the current prompt.

Output format:
First a <reasoning> block of 2-4 sentences describing what you are changing in the prompt and why (referencing the failure modes you target). Then, after </reasoning>, the new prompt text and nothing else — no other explanation, no surrounding code fences:

<reasoning>
...what is changing and why...
</reasoning>
...new prompt text..."""


def generate_new_prompt(ocfg: dict, it_dir: Path, metrics: dict, current_prompt: str,
                        history: list[dict], opt_dir: Path) -> str:
    opt = ocfg.get("optimizer", {})
    model = opt.get("model", "gpt-4.1")
    top_k = opt.get("top_k_fms", 3)
    per_fm = opt.get("examples_per_fm", 3)
    max_chars = opt.get("max_example_chars", 8000)

    ranked = sorted(metrics["fm_prevalence"].items(), key=lambda kv: kv[1], reverse=True)
    top_modes = [m for m, p in ranked[:top_k] if p > 0]

    pred = _read_predictions(it_dir)
    examples_text, used_ids = _collect_failing_examples(it_dir, pred, top_modes, per_fm, max_chars)
    history_text = _build_history_section(history, opt_dir,
                                          max_entries=opt.get("history_max_entries", 4))
    meta_prompt = _build_optimizer_prompt(
        current_prompt, metrics, examples_text, top_modes, ocfg["model"]["name"], history_text
    )

    client = OpenAI(base_url=ocfg["model"]["base_url"], api_key=os.environ["UVA_API_KEY"])
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        temperature=opt.get("temperature", 0.7),
        messages=[{"role": "user", "content": meta_prompt}],
    )
    latency = round(time.time() - t0, 3)
    reasoning, prompt_text = _split_reasoning(response.choices[0].message.content or "")
    new_prompt = _sanitize_prompt(prompt_text)

    usage = response.usage
    price_in, price_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    meta = {
        "model": model,
        "temperature": opt.get("temperature", 0.7),
        "reasoning": reasoning,
        "top_failure_modes": top_modes,
        "example_trace_ids": used_ids,
        "tokens_in": usage.prompt_tokens if usage else None,
        "tokens_out": usage.completion_tokens if usage else None,
        "cost_usd": ((usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1_000_000)
        if usage else None,
        "latency_s": latency,
        "timestamp": datetime.now().isoformat(),
    }
    (it_dir / "prompt_generation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return new_prompt


def write_summary(opt_dir: Path, ocfg: dict, history: list[dict], stop_reason: str):
    best = min(history, key=lambda m: m["total_fm_prevalence"]) if history else None
    summary = {
        "updated": datetime.now().isoformat(),
        "config": ocfg,
        "n_iterations": len(history),
        "stop_reason": stop_reason,
        "best_iteration": best["iteration"] if best else None,
        "best_total_fm_prevalence": best["total_fm_prevalence"] if best else None,
        "iterations": history,
    }
    (opt_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(config_path: Path):
    ocfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    n = ocfg.get("n_tasks_per_iteration", 20)
    max_iterations = ocfg.get("max_iterations", 5)
    threshold = ocfg.get("convergence_threshold", 0.02)
    seed = ocfg.get("benchmark_subset_seed", 42)

    opt_dir = (BASE_DIR / ocfg.get("output_dir", "results/optimization_run")).resolve()
    opt_dir.mkdir(parents=True, exist_ok=True)
    judge_config = (BASE_DIR / ocfg.get("judge_config", "judge/config.yaml")).resolve()

    # one fixed subset for all iterations, so prompts are compared on the same tasks
    data_path = create_olympiad_sample(n, seed=seed)

    current_prompt = (BASE_DIR / ocfg["initial_prompt"]).read_text(encoding="utf-8")

    history = []
    stop_reason = "max_iterations"
    for k in range(max_iterations):
        it_dir = opt_dir / f"iteration_{k}"
        it_dir.mkdir(exist_ok=True)

        prompt_path = it_dir / "prompt.txt"
        if prompt_path.exists():
            current_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt_path.write_text(current_prompt, encoding="utf-8")

        metrics_path = it_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"\n=== Iteration {k}: metrics.json found, skipping run/judge ===")
        else:
            print(f"\n=== Iteration {k}: running MathChat on {n} tasks ===")
            run_cfg = {
                "stage": f"opt_iter{k}",
                "benchmark": ocfg.get("benchmark", "olympiad"),
                "n": n,
                "max_auto_reply": ocfg.get("max_auto_reply", 10),
                "model": ocfg["model"],
                "prompt": "prompt.txt",
                "data_path": str(data_path),
                "output_dir": str(it_dir),
            }
            it_cfg_path = it_dir / "iteration_config.yaml"
            with open(it_cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(run_cfg, f, allow_unicode=True)

            run_module.run(it_cfg_path)

            print(f"\n=== Iteration {k}: judging traces ===")
            judge_module.main(
                run_dir=it_dir,
                config_path=judge_config,
                out_dir=it_dir / "judge_results",
            )
            metrics = compute_metrics(it_dir, k)

        history.append(metrics)
        write_summary(opt_dir, ocfg, history, stop_reason="running")

        total = metrics["total_fm_prevalence"]
        print(f"\nIteration {k}: total FM prevalence={total:.3f}, accuracy={metrics['accuracy']:.0%}")

        if k > 0:
            delta = abs(total - history[k - 1]["total_fm_prevalence"])
            if delta < threshold:
                stop_reason = f"converged: |delta|={delta:.4f} < {threshold}"
                print(f"Converged ({stop_reason}), stopping.")
                break
        if k == max_iterations - 1:
            break

        next_prompt_path = opt_dir / f"iteration_{k + 1}" / "prompt.txt"
        if next_prompt_path.exists():
            current_prompt = next_prompt_path.read_text(encoding="utf-8")
            print(f"Prompt for iteration {k + 1} already exists, reusing it.")
        else:
            print(f"Generating improved prompt for iteration {k + 1}...")
            current_prompt = generate_new_prompt(ocfg, it_dir, metrics, current_prompt,
                                                 history, opt_dir)

    write_summary(opt_dir, ocfg, history, stop_reason)
    best = min(history, key=lambda m: m["total_fm_prevalence"])
    print(f"\nDone after {len(history)} iteration(s). Stop reason: {stop_reason}")
    print(f"Best iteration: {best['iteration']} "
          f"(total FM prevalence {best['total_fm_prevalence']:.3f}, accuracy {best['accuracy']:.0%})")
    print(f"Results in {opt_dir}")


if __name__ == "__main__":
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "optimization_config.yaml"
    main(cfg_path)
