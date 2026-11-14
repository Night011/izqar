"""Small, dependency-free provenance records for validation-only ML experiments."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _revision() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL, timeout=2).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _peak_rss_mb() -> float | None:
    """Best-effort process peak RSS; Windows safely reports unavailable without psutil."""
    try:
        import resource  # Unix/Kaggle only
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux ru_maxrss is KiB; macOS is bytes.
        return round(value / (1024 * 1024 if platform.system() == 'Darwin' else 1024), 3)
    except (ImportError, AttributeError):
        return None


def start_experiment(*, dataset: str, algorithm: str, features: list[str], split_method: str,
                     hyperparameters: dict, mode: str) -> tuple[dict, float]:
    started = datetime.now(timezone.utc)
    fingerprint = json.dumps({'dataset': dataset, 'algorithm': algorithm, 'features': features,
                              'split_method': split_method, 'hyperparameters': hyperparameters,
                              'mode': mode, 'started_at_utc': started.isoformat()}, sort_keys=True)
    return ({'experiment_id': f"{dataset.lower().replace(' ', '-')}-{hashlib.sha256(fingerprint.encode()).hexdigest()[:12]}",
             'dataset': dataset, 'algorithm': algorithm, 'features': features, 'split_method': split_method,
             'hyperparameters': hyperparameters, 'mode': mode, 'started_at_utc': started.isoformat(),
             'code_revision': _revision(), 'python': platform.python_version(), 'platform': platform.platform()},
            time.monotonic())


def finish_experiment(record: dict, started_monotonic: float) -> dict:
    result = dict(record)
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['training_seconds'] = round(time.monotonic() - started_monotonic, 3)
    result['peak_rss_mb'] = _peak_rss_mb()
    result['peak_rss_note'] = ('OS process high-water mark; unavailable on Windows without an optional profiler.'
                               if result['peak_rss_mb'] is None else 'OS process high-water mark.')
    return result
