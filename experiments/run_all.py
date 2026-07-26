"""Run every lightweight experiment and regenerate paper-ready artifacts."""

from __future__ import annotations

import numpy as np

from bandit_experiments import (
    experiment_bandit_certificates, experiment_bandit_population,
)
from bandit_stress import experiment_bandit_efficiency, experiment_random_bandits
from common import SEED, SUMMARY_PATH, ensure_dirs
from report import write_summary
from sequential_experiments import (
    experiment_sequential_certificates, experiment_sequential_population,
)
from tree_experiments import experiment_tree_identity


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    bp = experiment_bandit_population()
    bc = experiment_bandit_certificates(rng)
    sp = experiment_sequential_population()
    sc = experiment_sequential_certificates(rng)
    ti = experiment_tree_identity(rng)
    rb = experiment_random_bandits(rng)
    be = experiment_bandit_efficiency(rng)
    write_summary(bp, bc, sp, sc, ti, rb, be)
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
