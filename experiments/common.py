"""Shared configuration for the lightweight experiments."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "data"
FIG_DIR = ROOT / "results" / "figures"
SUMMARY_PATH = ROOT / "results" / "RESULTS_SUMMARY.md"
SEED = 20260726

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def bandit_policies(q: float, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Three-action pair with shared (or slightly mismatched) unsupported mass."""

    q_target = q + epsilon / 2.0
    q_baseline = q - epsilon / 2.0
    if not (0.0 <= q_baseline <= q_target <= 1.0):
        raise ValueError("q and epsilon produce invalid mixture weights")
    target = np.array([0.8 * (1.0 - q_target), 0.2 * (1.0 - q_target), q_target])
    baseline = np.array(
        [0.2 * (1.0 - q_baseline), 0.8 * (1.0 - q_baseline), q_baseline]
    )
    return target, baseline
