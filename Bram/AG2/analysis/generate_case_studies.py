"""
Generate case study output files for all three Stage 2 interventions.

Output structure:
  case_studies/
    fm26_intervention/   (v1 run - FM-2.6)
    fm11_intervention/   (v2 run - FM-1.1)
    fm33_intervention/   (v3 run - FM-3.3)

Each folder:
  improved.txt                    baseline wrong → intervention correct
  degraded.txt                    baseline correct → intervention wrong
  fm_reduced_accuracy_unchanged.txt  target FM reduced but both wrong
  fm_increased.txt                target FM increased vs baseline
  fm_decreased.txt                target FM decreased vs baseline
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))  # AG2/ — for paths.py
import paths  # type: ignore

sys.path.insert(0, str(paths.REPO_ROOT / "LLM_models_interface"))
from llm_interface import JudgeResponse  # noqa: F401 – needed for pickle  # type: ignore

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS = paths.RESULTS_DIR
OUT_ROOT = paths.AG2_DIR / "case_studies"

BASELINE_DIR = RESULTS / "baseline_olympiad_gpt41_n50_20260609"

INTERVENTIONS = [
    {
        "label": "FM-2.6 Action-Reasoning Mismatch",
        "folder": "fm26_intervention",
        "run_dir": RESULTS / "stage_2_v1_olympiad_gpt41_n50_20260611",
        "target_fm": "2.6",
        "base_ckpt": "olympiad_gemini25flash_zero_shot.pkl",
        "int_ckpt": "olympiad_gemini25flash_zero_shot.pkl",
    },
    {
        "label": "FM-1.1 Disobey Task Specification",
        "folder": "fm11_intervention",
        "run_dir": RESULTS / "stage_2_v2_olympiad_gpt41_n50_20260612",
        "target_fm": "1.1",
        "base_ckpt": "olympiad_gemini25flash_zero_shot.pkl",
        "int_ckpt": "olympiad_gemini25flash_few_shot.pkl",
    },
    {
        "label": "FM-3.3 Weak Verification",
        "folder": "fm33_intervention",
        "run_dir": RESULTS / "stage_2_v3_olympiad_gpt41_n50_20260613",
        "target_fm": "3.3",
        "base_ckpt": "olympiad_gemini25flash_zero_shot.pkl",
        "int_ckpt": "olympiad_gemini25flash_few_shot.pkl",
    },
]

FM_CODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

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


# ── helpers ───────────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """Return (trace_by_qid, pred_by_tid, summ_by_tid)."""
    with open(run_dir / "parsed_traces.json", encoding="utf-8") as f:
        parsed = json.load(f)

    pred = pd.read_csv(run_dir / paths.JUDGE_SUBDIR / "predictions.csv")
    summ = pd.read_csv(run_dir / "summary.csv")
    summ["correct"] = summ["correct"].map(
        {"True": True, "False": False, True: True, False: False}
    ).astype(bool)

    trace_by_qid: dict[int, dict] = {}
    for t in parsed:
        qid = int(t["metadata"]["task_id"])
        trace_by_qid[qid] = t

    pred_by_tid = {row["trace_id"]: row for _, row in pred.iterrows()}
    summ_by_tid = {row["trace_id"]: row for _, row in summ.iterrows()}
    return trace_by_qid, pred_by_tid, summ_by_tid


def load_judge_reasoning(ckpt_path: Path) -> dict[str, str]:
    """Return {trace_id: raw_text} from a judge checkpoint pickle."""
    with open(ckpt_path, "rb") as f:
        results = pickle.load(f)
    return {r.trace_id: r.raw_text for r in results}


def fmt_conversation(steps: list[dict]) -> str:
    parts = []
    for step in steps:
        parts.append(f"[{step['agent']}]")
        parts.append(step["content"].rstrip())
        parts.append("")
    return "\n".join(parts)


def fmt_judge(pred_row: pd.Series, raw_text: str) -> str:
    labels = "  ".join(
        f"{fm}: {int(pred_row[fm])}" for fm in FM_CODES
    )
    return f"{labels}\n\nReasoning:\n{raw_text.strip()}"


def fmt_trace_pair(
    question_id: int,
    base_trace: dict,
    int_trace: dict,
    base_pred: pd.Series,
    int_pred: pd.Series,
    base_summ: pd.Series,
    int_summ: pd.Series,
    base_raw: str,
    int_raw: str,
    target_fm: str,
) -> str:
    base_tid = base_trace["metadata"]["trace_id"]
    int_tid = int_trace["metadata"]["trace_id"]
    base_correct = bool(base_summ["correct"])
    int_correct = bool(int_summ["correct"])
    base_fm_val = int(base_pred[target_fm])
    int_fm_val = int(int_pred[target_fm])

    header = "\n".join([
        f"question_id  : {question_id}",
        f"trace_id     : baseline={base_tid}  |  intervention={int_tid}",
        f"correct      : baseline={base_correct}  |  intervention={int_correct}",
        f"FM-{target_fm} ({FM_NAMES[target_fm]}) : baseline={base_fm_val}  |  intervention={int_fm_val}",
    ])

    conv_base = fmt_conversation(base_trace["steps"])
    conv_int = fmt_conversation(int_trace["steps"])
    judge_base = fmt_judge(base_pred, base_raw)
    judge_int = fmt_judge(int_pred, int_raw)

    return "\n".join([
        SEP,
        header,
        SEP,
        "",
        "=== BASELINE CONVERSATION ===",
        conv_base,
        "=== BASELINE JUDGE ===",
        judge_base,
        "",
        "=== INTERVENTION CONVERSATION ===",
        conv_int,
        "=== INTERVENTION JUDGE ===",
        judge_int,
        "",
    ])


def write_bucket(
    path: Path,
    pairs: list[str],
    label: str,
    intervention_label: str,
    category: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join([
        f"Intervention : {intervention_label}",
        f"Category     : {category}",
        f"Count        : {len(pairs)}",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for block in pairs:
            f.write(block)
    print(f"  {path.name}: {len(pairs)} trace(s)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading baseline …")
    base_by_qid, base_pred_by_tid, base_summ_by_tid = load_run(BASELINE_DIR)

    for cfg in INTERVENTIONS:
        run_dir: Path = cfg["run_dir"]
        target_fm: str = cfg["target_fm"]
        out_dir = OUT_ROOT / cfg["folder"]
        label: str = cfg["label"]

        print(f"\n{'-'*60}")
        print(f"Intervention: {label}")
        print(f"Run dir     : {run_dir.name}")
        print(f"Output dir  : {out_dir}")

        # load run data
        int_by_qid, int_pred_by_tid, int_summ_by_tid = load_run(run_dir)

        # load judge reasoning
        base_ckpt = BASELINE_DIR / paths.JUDGE_SUBDIR / "checkpoints" / cfg["base_ckpt"]
        int_ckpt = run_dir / paths.JUDGE_SUBDIR / "checkpoints" / cfg["int_ckpt"]
        base_reasoning = load_judge_reasoning(base_ckpt)
        int_reasoning = load_judge_reasoning(int_ckpt)

        # find shared question_ids
        shared_qids = sorted(set(base_by_qid) & set(int_by_qid))
        print(f"Paired tasks: {len(shared_qids)}")

        # buckets
        improved: list[str] = []
        degraded: list[str] = []
        fm_reduced_acc_unchanged: list[str] = []
        fm_increased: list[str] = []
        fm_decreased: list[str] = []

        for qid in shared_qids:
            base_trace = base_by_qid[qid]
            int_trace = int_by_qid[qid]

            base_tid = base_trace["metadata"]["trace_id"]
            int_tid = int_trace["metadata"]["trace_id"]

            base_pred = base_pred_by_tid[base_tid]
            int_pred = int_pred_by_tid[int_tid]
            base_summ = base_summ_by_tid[base_tid]
            int_summ = int_summ_by_tid[int_tid]

            base_raw = base_reasoning.get(base_tid, "(reasoning not found)")
            int_raw = int_reasoning.get(int_tid, "(reasoning not found)")

            base_correct = bool(base_summ["correct"])
            int_correct = bool(int_summ["correct"])
            base_fm = int(base_pred[target_fm])
            int_fm = int(int_pred[target_fm])

            block = fmt_trace_pair(
                qid,
                base_trace, int_trace,
                base_pred, int_pred,
                base_summ, int_summ,
                base_raw, int_raw,
                target_fm,
            )

            if not base_correct and int_correct:
                improved.append(block)
            if base_correct and not int_correct:
                degraded.append(block)
            if base_fm == 1 and int_fm == 0 and not base_correct and not int_correct:
                fm_reduced_acc_unchanged.append(block)
            if base_fm == 0 and int_fm == 1:
                fm_increased.append(block)
            if base_fm == 1 and int_fm == 0:
                fm_decreased.append(block)

        write_bucket(
            out_dir / "improved.txt", improved, label,
            label, "baseline wrong → intervention correct",
        )
        write_bucket(
            out_dir / "degraded.txt", degraded, label,
            label, "baseline correct → intervention wrong",
        )
        write_bucket(
            out_dir / "fm_reduced_accuracy_unchanged.txt", fm_reduced_acc_unchanged, label,
            label, f"FM-{target_fm} reduced but both wrong (accuracy unchanged)",
        )
        write_bucket(
            out_dir / "fm_increased.txt", fm_increased, label,
            label, f"FM-{target_fm} increased vs baseline",
        )
        write_bucket(
            out_dir / "fm_decreased.txt", fm_decreased, label,
            label, f"FM-{target_fm} decreased vs baseline",
        )

    print(f"\nDone. Output in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
