"""Central path definitions for the AG2 experiment pipeline.

Single source of truth for where things live, so scripts and notebooks no longer
hardcode run directories or depend on the current working directory. All paths are
derived from this file's location (``__file__``), not from ``cwd``.

Run-id convention (active runs follow this): ``{stage}_{benchmark}_{model}_n{N}_{YYYYMMDD}``
e.g. ``stage_2_v3_olympiad_gpt41_n50_20260613``.

Typical use:
    import paths
    run = paths.run_dir("stage_2_v3_olympiad_gpt41_n50_20260613")
    judge_out = run / paths.JUDGE_SUBDIR
"""

from pathlib import Path

# ── core directories ─────────────────────────────────────────────────────────
AG2_DIR = Path(__file__).resolve().parent           # Bram/AG2
REPO_ROOT = AG2_DIR.parents[1]                       # repo root (for parsers, LLM_models_interface)

DATA_DIR = AG2_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"                   # cached benchmark samples (*.json)
PROMPTS_DIR = AG2_DIR / "prompts"
RESULTS_DIR = AG2_DIR / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"

# ── conventional sub-directory name for judge output inside a run ─────────────
JUDGE_SUBDIR = "judge_results"


def run_dir(run_id: str) -> Path:
    """Path to a single run directory under results/."""
    return RESULTS_DIR / run_id


def latest_run(prefix: str = "") -> Path:
    """Most recent run directory (by name) optionally filtered by prefix.

    Run ids end in a date, so lexicographic max == most recent for a given prefix.
    """
    runs = [
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name != "archive" and d.name.startswith(prefix)
    ]
    if not runs:
        raise FileNotFoundError(f"No run directories under {RESULTS_DIR} matching prefix {prefix!r}")
    return max(runs, key=lambda d: d.name)
