"""Integration tests for the sweeper using subprocess."""

import os
import subprocess
import sys

import pytest


EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
PYTHON = sys.executable


def _run_example(example_dir, script, tmp_path, extra_overrides=None):
    """Run an example and return combined stdout+stderr."""
    args = [PYTHON, script, "-m", f"hydra.sweep.dir={tmp_path}"]
    if extra_overrides:
        args.extend(extra_overrides)
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=os.path.join(EXAMPLES_DIR, example_dir),
        timeout=120,
    )
    combined = result.stdout + result.stderr
    return result, combined


class TestSweeperIntegration:
    def test_simple_example(self, tmp_path):
        result, output = _run_example("simple", "my_app.py", tmp_path)
        assert result.returncode == 0, output
        assert "Best parameters" in output
        assert "Best value" in output
        assert os.path.exists(os.path.join(tmp_path, "optimization_results.yaml"))

    def test_pruning_example(self, tmp_path):
        result, output = _run_example("pruning_basic", "my_app.py", tmp_path)
        assert result.returncode == 0, output
        assert "Best parameters" in output
        assert "pruned" in output.lower()

    def test_multi_objective_example(self, tmp_path):
        result, output = _run_example("multi_objective", "my_app.py", tmp_path)
        assert result.returncode == 0, output
        assert "Pareto solutions" in output

    def test_pruning_pl_example(self, tmp_path):
        result, output = _run_example(
            "pruning_pytorch_lightning",
            "train.py",
            tmp_path,
            extra_overrides=[
                "hydra.sweeper.storage=null",
                "hydra.sweeper.study_name=null",
            ],
        )
        assert result.returncode == 0, output
        assert "Best parameters" in output
