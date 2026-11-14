"""Pure protocol tests for the Kaggle-only joint trainer."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from app.unified_network import TASKS, required_shapes
from train_unified import (baf, balanced_cap, choose_threshold, experiment_provenance, final_evaluation,
                           ieee, metrics, parse_task_weights, resolve_device, transformer)  # noqa: E402


def test_unified_trainer_cli_imports_app_when_run_directly():
    """Kaggle invokes this script by path, not with ``python -m``."""
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_unified.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert '--task-weights' in result.stdout


def test_training_adapter_is_train_fitted_and_handles_unknown_categories():
    train = pd.DataFrame({'amount': [1., 2., 3., 4.], 'kind': ['a', 'a', 'b', 'b']})
    valid = pd.DataFrame({'amount': [2., np.nan], 'kind': ['new', None]})
    pre = transformer(['amount'], ['kind']).fit(train)
    output = pre.transform(valid)
    assert output.shape[0] == 2
    assert np.isfinite(output.toarray() if hasattr(output, 'toarray') else output).all()


def test_task_weights_are_complete_positive_and_explicit():
    assert parse_task_weights(None) == {'paysim': 1.0, 'ieee_cis': 1.0, 'baf_base': 1.0, 'ibm_aml': 1.0}
    custom = parse_task_weights('{"paysim": 1.5, "ieee_cis": 1, "baf_base": 0.5, "ibm_aml": 2}')
    assert custom['paysim'] == 1.5 and custom['ibm_aml'] == 2.0
    with pytest.raises(ValueError, match='exactly'):
        parse_task_weights('{"paysim": 1}')
    with pytest.raises(ValueError, match='positive'):
        parse_task_weights('{"paysim": 1, "ieee_cis": 1, "baf_base": 0, "ibm_aml": 1}')


def test_unified_experiment_provenance_is_explicit_and_copies_weights():
    weights = {'paysim': 1.0, 'ieee_cis': 1.25, 'baf_base': 1.0, 'ibm_aml': 1.0}
    report = experiment_provenance(weights)
    weights['ieee_cis'] = 99.0
    assert report == {'mode': 'experiment', 'task_weights': {'paysim': 1.0, 'ieee_cis': 1.25,
                                                              'baf_base': 1.0, 'ibm_aml': 1.0}}


def test_capping_preserves_temporal_order_and_validation_metrics_are_complete():
    frame = pd.DataFrame({'time': list(range(20))})
    selected, y = balanced_cap(frame, np.arange(20) % 2, 7)
    assert selected.time.is_monotonic_increasing
    assert len(selected) == len(y) == 7
    truth = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([.01, .2, .6, .95])
    threshold = choose_threshold(truth, scores)
    report = metrics(truth, scores, threshold)
    assert 0 <= threshold <= 1
    assert set(report) >= {'average_precision_pr_auc', 'roc_auc', 'precision', 'recall', 'f1',
                           'confusion_matrix_tn_fp_fn_tp', 'false_alerts_per_10000_operations'}


def test_unified_calibration_diagnostic_is_validation_only_and_not_probability_claim():
    from train_ieee import calibration_diagnostics
    report = calibration_diagnostics(np.array([0, 0, 1, 1], dtype=np.int8),
                                     np.array([.1, .3, .7, .9], dtype=np.float32))
    assert report['brier_score_uncalibrated'] >= 0
    assert sum(bucket['cases'] for bucket in report['bins']) == 4
    assert 'not calibrated' in report['warning']


def test_ieee_adapter_requires_identity_and_enforces_one_to_one_merge(tmp_path):
    transaction = pd.DataFrame({'TransactionID': list(range(100, 140)), 'TransactionDT': list(range(40)),
        'TransactionAmt': [1.] * 40, 'card1': [1] * 40, 'card2': [1] * 40, 'card3': [1] * 40,
        'card5': [1] * 40, 'addr1': [1] * 40, 'dist1': [1.] * 40, 'ProductCD': ['W'] * 40,
        'card4': ['visa'] * 40, 'card6': ['credit'] * 40, 'P_emaildomain': ['x.test'] * 40,
        'isFraud': [i % 5 == 0 for i in range(40)]})
    transaction.to_csv(tmp_path / 'train_transaction.csv', index=False)
    try:
        ieee(tmp_path, 10)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('identity file is required')
    identity = pd.DataFrame({'TransactionID': list(range(100, 140)), 'DeviceType': ['desktop'] * 40,
                             'DeviceInfo': ['browser'] * 40, 'id_31': ['chrome'] * 40})
    identity.to_csv(tmp_path / 'train_identity.csv', index=False)
    result = ieee(tmp_path, 10)
    assert 'DeviceType' in result[0].columns and result[4][-1] == 'id_31'
    identity.iloc[[0, 0]].to_csv(tmp_path / 'train_identity.csv', index=False)
    try:
        ieee(tmp_path, 10)
    except ValueError as exc:
        assert 'unique' in str(exc)
    else:
        raise AssertionError('duplicate identity IDs must fail')


def test_baf_adapter_unpacks_partition_and_boundary_contract(monkeypatch, tmp_path):
    rows = pd.DataFrame({
        'fraud_bool': [0, 1, 0, 1, 0, 1],
        'month': [0, 0, 1, 1, 2, 2],
        **{name: [1.] * 6 for name in ('income', 'name_email_similarity', 'prev_address_months_count',
                                        'current_address_months_count', 'customer_age', 'intended_balcon_amount',
                                        'proposed_credit_limit', 'session_length_in_minutes', 'bank_months_count',
                                        'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards',
                                        'foreign_request', 'keep_alive_session')},
        **{name: ['value'] * 6 for name in ('payment_type', 'employment_status', 'housing_status', 'source', 'device_os')},
    })
    monkeypatch.setattr('train_unified.read_base', lambda _: rows)
    train, valid, sealed = rows.iloc[:2], rows.iloc[2:4], rows.iloc[4:]
    monkeypatch.setattr('train_unified.split_months', lambda _: ((train, valid, sealed), {'test_month': 2}))
    result = baf(tmp_path / 'Base.csv', 10)
    assert len(result[0]) == len(train)
    assert len(result[2]) == len(valid)
    assert result[6]['method'].endswith('final month sealed')


def test_auto_device_resolves_to_cpu_without_cuda():
    assert resolve_device('auto', cuda_available=False) == 'cpu'
    assert resolve_device('auto', cuda_available=True) == 'cuda'
    try:
        resolve_device('cuda', cuda_available=False)
    except RuntimeError as exc:
        assert 'unavailable' in str(exc)
    else:
        raise AssertionError('explicit CUDA must fail when unavailable')


def test_final_evaluation_reads_frozen_export_once_without_training(tmp_path, monkeypatch):
    dimensions = {task: 1 for task in TASKS}
    manifest = {'model_kind': 'izqar_unified_mtl', 'model_version': 'test-run', 'input_dimensions': dimensions,
                'tasks': {task: {'threshold': .5} for task in TASKS}}
    (tmp_path / 'unified.metrics.json').write_text(json.dumps(manifest), encoding='utf-8')
    joblib.dump({task: FunctionTransformer(validate=False) for task in TASKS}, tmp_path / 'unified_preprocessors.joblib')
    np.savez_compressed(tmp_path / 'unified_weights.npz', **{
        key: np.zeros(shape, dtype=np.float32) for key, shape in required_shapes(dimensions).items()
    })
    frame = pd.DataFrame({'amount': [1., 2., 3., 4.]})
    target = np.array([0, 0, 1, 1], dtype=np.int8)
    loaded = (frame, target, frame, target, [], ['amount'], {'method': 'test'}, frame, target)
    for name in ('paysim', 'ieee', 'baf', 'ibm'):
        monkeypatch.setattr(f'train_unified.{name}', lambda *args, **kwargs: loaded)
    args = SimpleNamespace(output_dir=tmp_path, paysim=tmp_path / 'paysim.csv', ieee_dir=tmp_path,
                           baf=tmp_path / 'baf.csv', ibm_aml=tmp_path / 'ibm.csv', max_train_rows=4,
                           max_final_rows=None)
    result = final_evaluation(args)
    assert set(result['tasks']) == set(TASKS)
    assert (tmp_path / 'unified.final_test.json').is_file()
    assert result['tasks']['paysim']['threshold_from_validation'] == .5
    assert result['tasks']['paysim']['final_calibration']['brier_score_uncalibrated'] >= 0
    assert 'not calibrated' in result['tasks']['paysim']['final_calibration']['warning']
    with pytest.raises(FileExistsError, match='refusing'):
        final_evaluation(args)
