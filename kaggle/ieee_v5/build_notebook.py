import json
import pathlib

REV = 'd5d4744cc923133cae6a63e26466c4edcc021535'

markdown = """# IZQAR IEEE-CIS v5 - seed-ensemble research CV (GPU)

Research mode only: the final 15% tail is physically excluded and never
evaluated; three expanding chronological folds, identical to v4.

New lever: `--members 3` fits seeds 42/43/44 per fold and rank-averages their
validation scores. v4 established that frequency features are rejected (none
0.5754 vs causal 0.5600 vs causal_log 0.5569), so this arm uses `--freq none`
and differs from the v4 "none" arm only by the ensemble.
"""

setup = f'''from pathlib import Path
import json, subprocess, sys

EXPECTED_REVISION = "{REV}"
root = Path("/kaggle/working/izqar_ieee_v5")
if root.exists():
    raise FileExistsError("Use fresh session")
subprocess.run(["git", "clone", "--branch", "feat/ieee-cis-research", "https://github.com/Night011/izqar.git", str(root)], check=True)
subprocess.run(["git", "checkout", "--detach", EXPECTED_REVISION], cwd=root, check=True)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
assert actual == EXPECTED_REVISION, f"unexpected revision {{actual}}"
backend = root / "backend"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "-r", str(backend / "requirements-ml.txt")], check=True)
matches = list(Path("/kaggle/input").rglob("train_transaction.csv"))
assert len(matches) == 1, f"expected one train_transaction.csv, got {{matches}}"
data_dir = matches[0].parent
assert (data_dir / "train_identity.csv").is_file(), "train_identity.csv missing"
print("input:", data_dir, "bytes:", matches[0].stat().st_size, "revision:", actual)
import xgboost; print("xgboost", xgboost.__version__)
'''

run = '''import subprocess, sys, json
from pathlib import Path

out = Path("/kaggle/working/ieee_v5_ens3.joblib")
subprocess.run([sys.executable, str(backend / "scripts" / "train_ieee_v3.py"),
                "--data-dir", str(data_dir), "--output", str(out),
                "--mode", "research", "--freq", "none", "--device", "cuda",
                "--estimators", "1500", "--members", "3"], cwd=backend, check=True)
report_path = out.with_suffix(".cv.metrics.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
print("== ensemble members=3 ==", json.dumps(report["mean_metrics"], indent=2))
Path("/kaggle/working/ieee_v5_ens3.cv.metrics.json").write_text(
    report_path.read_text(encoding="utf-8"), encoding="utf-8")
print("written /kaggle/working/ieee_v5_ens3.cv.metrics.json")
'''

cells = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': markdown.splitlines(True)},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
     'source': setup.splitlines(True)},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
     'source': run.splitlines(True)},
]
notebook = {
    'cells': cells,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                'name': 'python3'},
                 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 4,
}
pathlib.Path('izqar-ieee-v5-seed-ensemble-gpu-cv-v1.ipynb').write_text(
    json.dumps(notebook, indent=1), encoding='utf-8')
print('notebook written')
