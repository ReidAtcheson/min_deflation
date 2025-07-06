import subprocess
import sys
from pathlib import Path


def test_regression_experiment_runs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "regression_experiment.py"
    cmd = [sys.executable, str(script), "--samples", "3", "--eps-exp-low", "-3", "--eps-exp-high", "-2", "--m-exp-low", "1", "--m-exp-high", "2", "--seed", "0"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "intercept" in result.stdout
