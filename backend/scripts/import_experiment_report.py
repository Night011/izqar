"""Import a report-only, validation-only Kaggle experiment without model code."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path


MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
REVISION = re.compile(r'^[0-9a-f]{40}$')


def import_report(archive_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Validate and copy only metadata; intentionally never deserializes a model."""
    if not archive_path.is_file() or archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Experiment archive is missing or exceeds the 2 MiB report limit')
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(members) != 2 or any(Path(name).name != name for name in members):
            raise ValueError('Archive must contain exactly two flat metadata files')
        metrics = [name for name in members if name.endswith('.metrics.json')]
        manifests = [name for name in members if name.endswith('_run.json')]
        if len(metrics) != 1 or len(manifests) != 1:
            raise ValueError('Archive must contain one metrics report and one run manifest')
        report = json.loads(archive.read(metrics[0]))
        run = json.loads(archive.read(manifests[0]))
    if (report.get('run_mode') != 'experiment' or report.get('test_metrics') is not None
            or report.get('baseline_test_metrics') is not None):
        raise ValueError('Only validation-only experiment reports can be imported')
    experiment = report.get('experiment')
    if not isinstance(experiment, dict) or experiment.get('mode') != 'experiment':
        raise ValueError('Experiment provenance is incomplete')
    if (run.get('mode') != 'experiment' or run.get('test_evaluation_performed') is not False
            or not isinstance(run.get('expected_revision'), str)
            or not REVISION.fullmatch(run['expected_revision'])
            or run.get('actual_revision') != run['expected_revision']):
        raise ValueError('Run manifest does not prove a pinned validation-only run')
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / metrics[0]
    run_path = output_dir / manifests[0]
    if report_path.exists() or run_path.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment report')
    shutil.copyfile(archive_path, output_dir / archive_path.name)
    with zipfile.ZipFile(archive_path) as archive:
        report_path.write_bytes(archive.read(metrics[0]))
        run_path.write_bytes(archive.read(manifests[0]))
    return report_path, run_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('models/experiments'))
    args = parser.parse_args()
    report_path, run_path = import_report(args.archive, args.output_dir)
    print(json.dumps({'report': str(report_path), 'run_manifest': str(run_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
