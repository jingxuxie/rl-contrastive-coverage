"""Run every lightweight experiment and regenerate paper-ready artifacts.

Each experiment executes in a fresh Python process.  Besides keeping the
results deterministic, process isolation releases NumPy/Matplotlib allocations
between experiments and makes the full reproduction path reliable on small
machines.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bandit_experiments import (  # noqa: E402
    experiment_bandit_certificates,
    experiment_bandit_population,
)
from bandit_stress import (  # noqa: E402
    experiment_bandit_efficiency,
    experiment_random_bandits,
)
from common import SEED, SUMMARY_PATH, ensure_dirs  # noqa: E402
from library_experiments import experiment_policy_library_selection  # noqa: E402
from report import write_summary  # noqa: E402
from sequential_experiments import (  # noqa: E402
    experiment_crossfit_augmented_mse,
    experiment_sequential_certificates,
    experiment_sequential_efficiency,
    experiment_sequential_population,
)
from tabular_experiments import experiment_stochastic_tabular_crossfit  # noqa: E402
from tree_experiments import experiment_tree_identity  # noqa: E402

STEP_NAMES = (
    "stochastic_tabular",
    "bandit_population",
    "bandit_certificate",
    "sequential_population",
    "sequential_certificate",
    "tree_identity",
    "random_bandits",
    "bandit_efficiency",
    "sequential_efficiency",
    "sequential_crossfit",
    "policy_library",
)


def _rng(index: int) -> np.random.Generator:
    child_seed = np.random.SeedSequence(SEED).spawn(8)[index]
    return np.random.default_rng(child_seed)


def run_step(name: str) -> Any:
    """Execute one named experiment using its fixed child random stream."""

    functions: dict[str, Callable[[], Any]] = {
        "stochastic_tabular": lambda: experiment_stochastic_tabular_crossfit(_rng(6)),
        "bandit_population": experiment_bandit_population,
        "bandit_certificate": lambda: experiment_bandit_certificates(_rng(0)),
        "sequential_population": experiment_sequential_population,
        "sequential_certificate": lambda: experiment_sequential_certificates(_rng(1)),
        "tree_identity": lambda: experiment_tree_identity(_rng(2)),
        "random_bandits": lambda: experiment_random_bandits(_rng(3)),
        "bandit_efficiency": lambda: experiment_bandit_efficiency(_rng(4)),
        "sequential_efficiency": experiment_sequential_efficiency,
        "sequential_crossfit": lambda: experiment_crossfit_augmented_mse(_rng(5)),
        "policy_library": lambda: experiment_policy_library_selection(_rng(7)),
    }
    try:
        function = functions[name]
    except KeyError as exc:
        raise ValueError(f"unknown experiment step: {name}") from exc
    return function()


def _child(step: str, output: Path) -> None:
    ensure_dirs()
    result = run_step(step)
    with output.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    ensure_dirs()
    outputs: list[Any] = []
    with tempfile.TemporaryDirectory(prefix="contrastive-coverage-") as directory:
        temp_dir = Path(directory)
        for index, name in enumerate(STEP_NAMES, start=1):
            print(f"[{index:02d}/{len(STEP_NAMES):02d}] {name.replace('_', ' ')}", flush=True)
            output = temp_dir / f"{index:02d}-{name}.pkl"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(SRC), environment.get("PYTHONPATH", "")])
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--step",
                    name,
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            with output.open("rb") as handle:
                outputs.append(pickle.load(handle))

    (
        stochastic_tabular,
        bandit_population,
        bandit_certificate,
        sequential_population,
        sequential_certificate,
        tree_identity,
        random_bandits,
        bandit_efficiency,
        sequential_efficiency,
        sequential_crossfit,
        policy_library,
    ) = outputs
    write_summary(
        bandit_population,
        bandit_certificate,
        sequential_population,
        sequential_certificate,
        tree_identity,
        random_bandits,
        bandit_efficiency,
        sequential_efficiency,
        sequential_crossfit,
        stochastic_tabular,
        policy_library,
    )
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=STEP_NAMES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.step is None:
        main()
    else:
        if arguments.output is None:
            raise SystemExit("--output is required with --step")
        _child(arguments.step, arguments.output)
