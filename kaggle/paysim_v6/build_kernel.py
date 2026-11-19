import json
import pathlib

REV = 'b1e26906b33127e443e34fef621f6d0fee9230e0'

markdown = """# IZQAR PaySim v6 - per-type operating point (validation only)

The sealed run serves one global max-F1 cutoff, which leaves TRANSFER recall at
1.000 and CASH_OUT at 0.470. This run asks whether a per-transaction-type cutoff,
each holding its own false-alert budget, does better.

`experiment` mode only: the sealed final test is not loaded or scored. Both
policies are fitted on the first half of the validation window and measured on
the second half, so the comparison is out-of-sample.
"""

setup = f'''from pathlib import Path
import json, subprocess, sys

EXPECTED_REVISION = "{REV}"
root = Path("/kaggle/working/izqar_paysim_v6")
if root.exists():
    raise FileExistsError("Use fresh session")
subprocess.run(["git", "clone", "--branch", "feat/paysim-thresholds", "https://github.com/Night011/izqar.git", str(root)], check=True)
subprocess.run(["git", "checkout", "--detach", EXPECTED_REVISION], cwd=root, check=True)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
assert actual == EXPECTED_REVISION, f"unexpected revision {{actual}}"
backend = root / "backend"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "-r", str(backend / "requirements-ml.txt")], check=True)
matches = list(Path("/kaggle/input").rglob("*.csv"))
print("candidates:", [str(m) for m in matches])
csvs = [m for m in matches if m.stat().st_size > 100_000_000]
assert len(csvs) == 1, f"expected one PaySim csv, got {{csvs}}"
data_path = csvs[0]
print("input:", data_path, "bytes:", data_path.stat().st_size, "revision:", actual)
'''

run = '''import subprocess, sys, json
from pathlib import Path

output = Path("/kaggle/working/paysim_v6_experiment.joblib")
subprocess.run([sys.executable, str(backend / "scripts" / "train_paysim.py"),
                "--data", str(data_path), "--output", str(output),
                "--estimators", "180", "--full-data", "--mode", "experiment",
                "--feature-set", "pre_transaction_v2",
                "--model-version", "paysim-xgboost-v6"], cwd=backend, check=True)
report_path = output.with_suffix(".metrics.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
policies = report["validation_threshold_policies"]
print("policies:", json.dumps(policies, indent=2))
Path("/kaggle/working/paysim_v6_threshold_policies.json").write_text(
    json.dumps(policies, indent=2), encoding="utf-8")
print("written /kaggle/working/paysim_v6_threshold_policies.json")
print("validation_metrics:", json.dumps(report.get("validation_metrics"), indent=2))
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
target = pathlib.Path('izqar-paysim-v6-thresholds-v1.ipynb')
target.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
metadata = {
    'id': 'hoogle57/izqar-paysim-v6-thresholds-v1',
    'title': 'IZQAR PaySim v6 Thresholds v1',
    'code_file': target.name,
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': 'true',
    'enable_gpu': 'false',
    'enable_internet': 'true',
    'dataset_sources': ['ealaxi/paysim1'],
    'competition_sources': [],
    'kernel_sources': [],
}
pathlib.Path('kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('paysim v6 kernel written')
