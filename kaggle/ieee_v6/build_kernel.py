import json
import pathlib

REV = '6ca43f2d46ea3d57719b78aebc2ec215000730cd'

markdown = """# IZQAR IEEE-CIS v6 - heterogeneous arm blend (GPU CV)

Research mode only: the final 15% tail is physically excluded and never
evaluated; the same three expanding chronological folds as v4/v5.

v5 showed a seed ensemble adds nothing (0.5757 vs 0.5754). This run scores the
two feature arms separately on the same folds (`none` and `causal`), saves their
out-of-fold validation scores, and rank-averages them. `none` is the strongest
single arm; `causal` is weaker alone but uses different features, which is the
diversity that helped on BAF.
"""

setup = f'''from pathlib import Path
import json, subprocess, sys

EXPECTED_REVISION = "{REV}"
root = Path("/kaggle/working/izqar_ieee_v6")
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
'''

run = '''import subprocess, sys, json
from pathlib import Path

for arm in ("none", "causal"):
    out = Path(f"/kaggle/working/ieee_v6_{arm}.joblib")
    subprocess.run([sys.executable, str(backend / "scripts" / "train_ieee_v3.py"),
                    "--data-dir", str(data_dir), "--output", str(out),
                    "--mode", "research", "--freq", arm, "--device", "cuda",
                    "--estimators", "1500",
                    "--save-scores", f"/kaggle/working/scores_{arm}"], cwd=backend, check=True)
    report = json.loads(out.with_suffix(".cv.metrics.json").read_text(encoding="utf-8"))
    print(f"== arm {arm} ==", json.dumps(report["mean_metrics"], indent=2))

subprocess.run([sys.executable, str(backend / "scripts" / "ieee_blend_scores.py"),
                "--scores-dir", "/kaggle/working/scores_none",
                "--scores-dir", "/kaggle/working/scores_causal",
                "--output", "/kaggle/working/ieee_v6_blend.json"], cwd=backend, check=True)
print(Path("/kaggle/working/ieee_v6_blend.json").read_text(encoding="utf-8"))
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
target = pathlib.Path('izqar-ieee-v6-arm-blend-gpu-cv-v1.ipynb')
target.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
metadata = {
    'id': 'hoogle57/izqar-ieee-v6-arm-blend-gpu-cv-v1',
    'title': 'IZQAR IEEE v6 Arm Blend GPU CV v1',
    'code_file': target.name,
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': 'true',
    'enable_gpu': 'true',
    'enable_tpu': 'false',
    'enable_internet': 'true',
    'competition_sources': ['ieee-fraud-detection'],
    'dataset_sources': [],
    'kernel_sources': [],
}
pathlib.Path('kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('notebook and metadata written')
