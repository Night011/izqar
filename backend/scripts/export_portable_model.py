"""Re-export a trusted local joblib artifact with stable XGBoost model IO.

Run in the environment that trained the model. This does not retrain or
re-evaluate on test; metrics are copied from the original report.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib


def export(source: Path, output_stem: Path, bundle_path: Path | None = None) -> Path:
    if source.suffix != '.joblib':
        raise ValueError('Source must be a trusted .joblib artifact')
    report_path = source.with_suffix('.metrics.json')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('format_version') != 2 or report.get('model_kind') not in {'paysim', 'ieee_cis', 'baf_base'}:
        raise ValueError('Unsupported source report')
    if not source.is_file():
        raise FileNotFoundError(source)
    names = {
        'preprocessor': output_stem.with_name(output_stem.name + '.preprocessor.joblib'),
        'model': output_stem.with_name(output_stem.name + '.xgboost.json'),
        'report': output_stem.with_name(output_stem.name + '.metrics.json'),
        'zip': bundle_path or output_stem.with_suffix('.zip'),
    }
    source_audit = source.with_name(source.stem + '.validation_audit.json')
    if source_audit.is_file():
        names['audit'] = output_stem.with_name(output_stem.name + '.validation_audit.json')
    if any(path.exists() for path in names.values()):
        raise FileExistsError('Portable output already exists; choose a new name')
    source_artifact = joblib.load(source)
    if source_artifact.get('model_kind') != report['model_kind'] or source_artifact.get('model_version') != report['model_version']:
        raise ValueError('Source artifact and report do not match')
    pipeline = source_artifact['pipeline']
    preprocessing = pipeline.named_steps['preprocessing']
    model = pipeline.named_steps['model']
    names['preprocessor'].parent.mkdir(parents=True, exist_ok=True)
    names['zip'].parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        joblib.dump(preprocessing, names['preprocessor'])
        created.append(names['preprocessor'])
        model.save_model(names['model'])
        created.append(names['model'])
        portable_report = dict(report)
        portable_report['format_version'] = 3
        portable_report['serialization'] = 'portable_xgboost_v1'
        portable_report['source_artifact'] = source.name
        names['report'].write_text(json.dumps(portable_report, ensure_ascii=False, indent=2), encoding='utf-8')
        created.append(names['report'])
        if 'audit' in names:
            audit = json.loads(source_audit.read_text(encoding='utf-8'))
            if audit.get('model_version') != report['model_version']:
                raise ValueError('Source validation audit does not match the model')
            names['audit'].write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
            created.append(names['audit'])
        with zipfile.ZipFile(names['zip'], 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for part in ('preprocessor', 'model', 'report', 'audit'):
                if part not in names:
                    continue
                bundle.write(names[part], arcname=names[part].name)
        created.append(names['zip'])
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return names['zip']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-stem', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    print(export(args.source, args.output_stem, args.bundle))


if __name__ == '__main__':
    main()
