"""
Export traces flagged for a specific failure mode to a readable .txt file.
Configure FM_CODE, RESULTS_DIR, and OUTPUT_FILE below, then run the script.
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))  # AG2/ — for paths.py
import paths  # type: ignore

# ── configuration ────────────────────────────────────────────────────────────
FM_CODE      = "3.3"
RUN_ID       = "baseline_olympiad_gpt41_n50_20260609"
RESULTS_DIR  = paths.run_dir(RUN_ID) / paths.JUDGE_SUBDIR
OUTPUT_FILE  = None   # None → auto-named as fm<code>_traces.txt in the run dir
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parents[3] / "LLM_models_interface"))
from llm_interface import JudgeResponse  # noqa: F401  (needed for pickle)

import pandas as pd

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

SEP = "=" * 80


def load_judge_results(results_dir: Path) -> dict[str, str]:
    """Return mapping trace_id -> raw_text from all checkpoint pickles."""
    checkpoint_dir = results_dir / "checkpoints"
    raw_texts: dict[str, str] = {}
    for pkl in checkpoint_dir.glob("*.pkl"):
        if "backup" in pkl.name:
            continue
        with open(pkl, "rb") as f:
            responses: list[JudgeResponse] = pickle.load(f)
        for r in responses:
            raw_texts[r.trace_id] = r.raw_text
    return raw_texts


def format_conversation(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        name = msg.get("name") or msg.get("role", "unknown")
        content = msg.get("content") or ""
        parts.append(f"[{name}]\n{content}")
    return "\n\n".join(parts)


def main():
    fm_code     = FM_CODE
    results_dir = Path(RESULTS_DIR)
    run_dir     = results_dir.parent

    if fm_code not in FM_NAMES:
        print(f"Unknown FM code '{fm_code}'. Valid codes: {', '.join(FM_NAMES)}")
        sys.exit(1)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Load predictions
    pred = pd.read_csv(results_dir / "predictions.csv")
    if fm_code not in pred.columns:
        print(f"Column '{fm_code}' not found in predictions.csv. Available: {list(pred.columns)}")
        sys.exit(1)

    flagged_ids = set(pred.loc[pred[fm_code] == 1, "trace_id"])
    if not flagged_ids:
        print(f"No traces flagged for FM {fm_code}.")
        sys.exit(0)

    # Load traces
    with open(run_dir / "raw_traces.json") as f:
        raw_traces = json.load(f)
    with open(run_dir / "parsed_traces.json") as f:
        parsed_traces = json.load(f)

    # Build lookup by trace_id
    raw_by_id   = {p["metadata"]["trace_id"]: r for r, p in zip(raw_traces, parsed_traces)}
    meta_by_id  = {p["metadata"]["trace_id"]: p["metadata"] for p in parsed_traces}

    # Load judge reasoning
    judge_texts = load_judge_results(results_dir)

    # Filter + sort flagged traces
    flagged = sorted(flagged_ids, key=lambda tid: int(tid.rsplit("_", 1)[-1]))

    # Determine output path
    if OUTPUT_FILE:
        out_path = Path(OUTPUT_FILE)
    else:
        safe_code = fm_code.replace(".", "")
        out_path = run_dir / f"fm{safe_code}_traces.txt"

    lines = []
    lines.append(f"Total FM-{fm_code} flagged: {len(flagged)}")

    for i, trace_id in enumerate(flagged, 1):
        meta = meta_by_id.get(trace_id, {})
        raw  = raw_by_id.get(trace_id, [])

        answer   = meta.get("final_answer", "?")
        expected = meta.get("expected_answer", "?")
        correct  = meta.get("correct", "?")

        lines.append("")
        lines.append(SEP)
        lines.append(f"TRACE {i}/{len(flagged)}")
        lines.append(f"trace_id : {trace_id}")
        lines.append(f"correct  : {correct}  (answer: {answer}  expected: {expected})")
        lines.append("")
        lines.append("--- CONVERSATION ---")
        lines.append(format_conversation(raw))
        lines.append("")
        lines.append("--- JUDGE REASONING ---")
        lines.append(judge_texts.get(trace_id, "(no judge output found)"))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(flagged)} traces to {out_path}")


if __name__ == "__main__":
    main()
