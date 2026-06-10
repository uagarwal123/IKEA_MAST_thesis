"""
baseline_utils.py
─────────────────
All logic for the baseline experiment.
The notebook calls these functions and contains no logic itself.
"""

import re
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    cohen_kappa_score, accuracy_score,
)

OFFICIAL_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]


# ── Caching ────────────────────────────────────────────────────────────────

def _cache_key(model_name: str, trace_id: str) -> str:
    """Make a short unique filename-safe key for a (model, trace) pair."""
    raw = f"{model_name}::{trace_id}"
    short = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe_model = model_name.replace("/", "-").replace(":", "-")
    safe_trace = str(trace_id).replace("/", "-")[:20]
    return f"{safe_model}__{safe_trace}__{short}"


def load_from_cache(cache_dir: Path, model_name: str, trace_id: str):
    """Return a cached JudgeResponse, or None if not cached yet."""
    p = cache_dir / f"{_cache_key(model_name, trace_id)}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def save_to_cache(cache_dir: Path, model_name: str, trace_id: str, response) -> None:
    """Save a JudgeResponse to disk immediately after each API call."""
    p = cache_dir / f"{_cache_key(model_name, trace_id)}.pkl"
    with open(p, "wb") as f:
        pickle.dump(response, f)


# ── Ground truth ───────────────────────────────────────────────────────────

def build_ground_truth(traces: list[dict]) -> dict[str, list[int]]:
    """
    Extract ground-truth labels from traces.
    Uses mast_annotation if present (full dataset),
    otherwise uses majority vote over 3 annotators (human-labelled dataset).
    """
    gt = {m: [] for m in OFFICIAL_MODES}

    if traces and "mast_annotation" in traces[0]:
        # Full dataset: one annotation dict per trace
        for record in traces:
            for code in OFFICIAL_MODES:
                gt[code].append(record["mast_annotation"].get(code, 0))
    else:
        # Human-labelled dataset: majority vote over annotator_1/2/3
        for record in traces:
            for ann in record.get("annotations", []):
                match = re.match(r"(\d+\.\d+)", ann.get("failure mode", ""))
                if not match:
                    continue
                code = match.group(1)
                if code not in OFFICIAL_MODES:
                    continue
                votes = [
                    ann.get("annotator_1", 0),
                    ann.get("annotator_2", 0),
                    ann.get("annotator_3", 0),
                ]
                gt[code].append(1 if sum(votes) >= 2 else 0)

    return gt


# ── Bootstrap CI ───────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, metric_fn, n: int = 1000, ci: float = 0.95):
    """Return (lo, hi) confidence interval for metric_fn via bootstrap."""
    rng = np.random.default_rng(42)
    yt, yp = np.array(y_true), np.array(y_pred)
    idx = np.arange(len(yt))
    scores = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        scores.append(metric_fn(yt[s], yp[s]))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_summary_row(cfg_name: str, model: str,
                        results: list, traces: list[dict],
                        n_bootstrap: int = 1000) -> tuple[dict, list, list]:
    """
    Compute all metrics for one experiment.
    Returns three things:
      - summary_row   : one dict with aggregate metrics (one row in the decision table)
      - per_mode_rows : one dict per failure mode with precision/recall/F1
      - prediction_rows: one dict per trace with raw predictions
    """
    gt = build_ground_truth(traces)
    fm_preds = {m: [r.annotations.get(m, 0) for r in results]
                for m in OFFICIAL_MODES}

    # Flatten across all modes for aggregate metrics
    y_true_flat, y_pred_flat = [], []
    for m in OFFICIAL_MODES:
        y_true_flat.extend(gt[m])
        y_pred_flat.extend(fm_preds[m])
    yt = np.array(y_true_flat)
    yp = np.array(y_pred_flat)

    # Bootstrap CIs on micro F1 and kappa
    f1_lo,    f1_hi    = bootstrap_ci(
        yt, yp, lambda a, b: f1_score(a, b, zero_division=0), n=n_bootstrap)
    kappa_lo, kappa_hi = bootstrap_ci(
        yt, yp, cohen_kappa_score, n=n_bootstrap)

    # Detect dataset from trace structure: full dataset has mast_annotation, human has annotations
    dataset = "human" if traces and "annotations" in traces[0] else "full"

    summary_row = {
        "name":            cfg_name,
        "dataset":         dataset,
        "model":           model,
        "accuracy":        accuracy_score(yt, yp),
        "precision":       precision_score(yt, yp, zero_division=0),
        "recall":          recall_score(yt, yp, zero_division=0),
        "micro_f1":        f1_score(yt, yp, zero_division=0),
        "macro_f1":        float(np.mean([
                               f1_score(gt[m], fm_preds[m], zero_division=0)
                               for m in OFFICIAL_MODES])),
        "kappa":           cohen_kappa_score(yt, yp),
        "f1_ci_lo":        f1_lo,    "f1_ci_hi":    f1_hi,
        "kappa_ci_lo":     kappa_lo, "kappa_ci_hi": kappa_hi,
        "total_cost_usd":  sum(r.cost_usd  for r in results),
        "mean_cost_usd":   sum(r.cost_usd  for r in results) / len(results),
        "total_latency_s": sum(r.latency_s for r in results),
        "mean_latency_s":  sum(r.latency_s for r in results) / len(results),
    }

    per_mode_rows = [
        {
            "name":      cfg_name,
            "model":     model,
            "mode":      m,
            "precision": precision_score(gt[m], fm_preds[m], zero_division=0),
            "recall":    recall_score(gt[m],    fm_preds[m], zero_division=0),
            "f1":        f1_score(gt[m],        fm_preds[m], zero_division=0),
            "support":   sum(gt[m]),
        }
        for m in OFFICIAL_MODES
    ]

    prediction_rows = [
        {
            "name":       cfg_name,
            "model":      r.model_id,
            "trace_id":   r.trace_id,
            "tokens_in":  r.tokens_in,
            "tokens_out": r.tokens_out,
            "latency_s":  r.latency_s,
            "cost_usd":   r.cost_usd,
            **r.annotations,
        }
        for r in results
    ]

    return summary_row, per_mode_rows, prediction_rows


# ── Decision table ─────────────────────────────────────────────────────────

def build_decision_table(summary_rows: list[dict],
                         per_mode_rows: list[dict]) -> pd.DataFrame:

    # Per-mode F1 as wide table
    pm = pd.DataFrame(per_mode_rows)

    if pm.empty:
        return pd.DataFrame(summary_rows)

    f1_wide = pm.pivot(index="name", columns="mode", values="f1")
    f1_wide.columns = [f"f1_{c}" for c in f1_wide.columns]

    # Aggregate metrics only — no CI columns here
    agg = pd.DataFrame(summary_rows).set_index("name")[[
        "model",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "micro_f1",
        "kappa",
        "total_cost_usd",
        "mean_cost_usd",
        "total_latency_s",
        "mean_latency_s",
    ]]

    # Join aggregate metrics with per-FM F1 columns
    decision = agg.join(f1_wide)

    # Clean column order
    ordered_cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "micro_f1",
        "kappa",
        "total_cost_usd",
        "mean_cost_usd",
        "total_latency_s",
        "mean_latency_s",
    ] + [f"f1_{m}" for m in OFFICIAL_MODES]

    decision = decision[ordered_cols]

    # Sort best models first, cheaper one first if tied
    decision = decision.sort_values(
        by=["macro_f1", "total_cost_usd"],
        ascending=[False, True]
    )

    return decision


# ── Figure ─────────────────────────────────────────────────────────────────

def plot_baseline_figure(decision_table: pd.DataFrame,
                         save_path: str = None):
    """
    Clean baseline figure:
      Left  — aggregate performance metrics by experiment
      Right — total cost vs macro-F1 trade-off
      Bottom — per-failure-mode F1 heatmap
    """

    # Per-FM columns from the clean decision table
    mode_cols = [c for c in decision_table.columns if c.startswith("f1_")]

    exp_names = decision_table.index.tolist()

    aggregate_cols = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "micro_f1",
        "kappa",
    ]

    aggregate_cols = [c for c in aggregate_cols if c in decision_table.columns]

    fig = plt.figure(figsize=(18, max(8, len(exp_names) * 1.2 + 5)))

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.3],
        width_ratios=[1.2, 1],
        hspace=0.4,
        wspace=0.3
    )

    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_tradeoff = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, :])

    # ── Panel 1: aggregate metrics ─────────────────────────────
    metric_data = decision_table[aggregate_cols]

    x = np.arange(len(exp_names))
    width = 0.12

    for i, col in enumerate(aggregate_cols):
        ax_metrics.bar(
            x + i * width,
            metric_data[col],
            width,
            label=col
        )

    ax_metrics.set_title("Aggregate performance by experiment")
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_xticks(x + width * (len(aggregate_cols) - 1) / 2)
    ax_metrics.set_xticklabels(exp_names, rotation=30, ha="right")
    ax_metrics.legend(fontsize=8)
    ax_metrics.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 2: cost vs macro F1 ──────────────────────────────
    costs = decision_table["total_cost_usd"]
    macro_f1s = decision_table["macro_f1"]

    ax_tradeoff.scatter(costs, macro_f1s, s=90)

    for exp in exp_names:
        ax_tradeoff.annotate(
            exp,
            (
                decision_table.loc[exp, "total_cost_usd"],
                decision_table.loc[exp, "macro_f1"]
            ),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax_tradeoff.set_title("Cost vs macro F1")
    ax_tradeoff.set_xlabel("Total cost USD")
    ax_tradeoff.set_ylabel("Macro F1")
    ax_tradeoff.set_ylim(0, 1)
    ax_tradeoff.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 3: per-mode F1 heatmap ───────────────────────────
    heatmap_data = decision_table[mode_cols]

    im = ax_heatmap.imshow(
        heatmap_data,
        aspect="auto",
        vmin=0,
        vmax=1
    )

    ax_heatmap.set_title("Per-failure-mode F1")
    ax_heatmap.set_xlabel("Failure mode")
    ax_heatmap.set_ylabel("Experiment")

    ax_heatmap.set_xticks(np.arange(len(mode_cols)))
    ax_heatmap.set_xticklabels(
        [c.replace("f1_", "") for c in mode_cols],
        rotation=45,
        ha="right"
    )

    ax_heatmap.set_yticks(np.arange(len(exp_names)))
    ax_heatmap.set_yticklabels(exp_names)

    for i in range(len(exp_names)):
        for j in range(len(mode_cols)):
            value = heatmap_data.iloc[i, j]
            ax_heatmap.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    cbar = fig.colorbar(im, ax=ax_heatmap)
    cbar.set_label("F1")

    fig.suptitle("LLM-as-a-Judge baseline comparison", fontsize=16, y=0.98)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()