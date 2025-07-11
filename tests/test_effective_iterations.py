import subprocess
import sys
from pathlib import Path


def test_effective_iterations_runs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "compare_effective_iterations.py"
    out_file = tmp_path / "out.svg"
    result = subprocess.run(
        [sys.executable, str(script), "--m", "8", "--nnz-per-row", "2", "--out", str(out_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_file.exists()
    assert "wrote" in result.stdout
