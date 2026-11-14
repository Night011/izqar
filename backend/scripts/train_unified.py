"""Jointly train IZQAR Unified Intelligence on four distinct fraud tasks.

The labels are never merged: each dataset has its own head. Dataset adapters
are fitted on their train period only; the shared backbone receives gradients
from every head in round-robin batches. This script is for Kaggle CPU/GPU
sessions where PyTorch is available. The resulting NPZ weights run locally
through ``app.unified_network`` without PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_baf import CATEGORICAL as BAF_CATS, FEATURES as BAF_FEATURES, NUMERIC as BAF_NUMS, read_base, split_months
from train_ibm_aml import load_splits as ibm_splits, time_boundaries as ibm_boundaries, validate_file as validate_ibm
from app.unified_network import UnifiedNetwork

TASKS = ('paysim', 'ieee_cis', 'baf_base', 'ibm_aml')
RNG = 42


def metrics(y, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'average_precision_pr_auc': round(float(average_precision_score(y, scores)), 6),
            'roc_auc': round(float(roc_auc_score(y, scores)), 6),
            'precision': round(float(precision_score(y, predicted, zero_division=0)), 6),
            'recall': round(float(recall_score(y, predicted, zero_division=0)), 6),
            'f1': round(float(f1_score(y, predicted, zero_division=0)), 6),
            'confusion_matrix_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
            'false_positive_rate': round(float(fp / max(1, tn + fp)), 6),
            'false_negative_rate': round(float(fn / max(1, fn + tp)), 6),
            'false_alerts_per_10000_operations': round(float(10_000 * fp / len(y)), 3),
            'cases': int(len(y)), 'positive_cases': int(y.sum())}


def choose_threshold(y, scores):
    p, r, cuts = precision_recall_curve(y, scores)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(cuts[int(np.argmax(f))])


def balanced_cap(frame, target, cap):
    """Chronology is retained; deterministic uniform thinning prevents RAM spikes."""
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True), target
    index = np.linspace(0, len(frame) - 1, cap, dtype=np.int64)
    return frame.iloc[index].reset_index(drop=True), np.asarray(target)[index]


def transformer(numbers, categories):
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numbers),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='__missing__')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=25))]), categories),
    ])


def paysim(path, cap, include_test=False):
    required = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud']
    data = pd.read_csv(path, usecols=required, dtype={'step':'int32','amount':'float32','oldbalanceOrg':'float32','oldbalanceDest':'float32','isFraud':'int8','type':'category'})
    data = data.sort_values('step', kind='stable').reset_index(drop=True)
    first, second = data.step.iloc[int(.70 * len(data))], data.step.iloc[int(.85 * len(data))]
    train, valid = data[data.step < first], data[(data.step >= first) & (data.step < second)]
    test = data[data.step >= second]
    fields = ['type','amount','oldbalanceOrg','oldbalanceDest']
    result = (train[fields], train.isFraud.to_numpy(), valid[fields], valid.isFraud.to_numpy(), fields[:1], fields[1:],
              {'method':'70/15/15 chronological step; final 15% sealed and never loaded for tuning', 'excluded':['isFraud','isFlaggedFraud','newbalanceOrig','newbalanceDest','nameOrig','nameDest','step']})
    return (*result, test[fields], test.isFraud.to_numpy()) if include_test else result


def ieee(data_dir, cap, include_test=False):
    tx = pd.read_csv(data_dir / 'train_transaction.csv', usecols=['TransactionID','TransactionDT','TransactionAmt','card1','card2','card3','card5','addr1','dist1','ProductCD','card4','card6','P_emaildomain','isFraud'], low_memory=False)
    tx = tx.sort_values(['TransactionDT','TransactionID'], kind='stable').reset_index(drop=True)
    first, second = tx.TransactionDT.iloc[int(.70 * len(tx))], tx.TransactionDT.iloc[int(.85 * len(tx))]
    identity_path = data_dir / 'train_identity.csv'
    if not identity_path.is_file():
        raise FileNotFoundError('Unified IEEE adapter requires train_identity.csv alongside train_transaction.csv')
    identity = pd.read_csv(identity_path, usecols=lambda name: name in {'TransactionID', 'DeviceType', 'DeviceInfo', 'id_31'}, low_memory=False)
    if identity.TransactionID.isna().any() or not identity.TransactionID.is_unique:
        raise ValueError('IEEE identity TransactionID must be unique and complete')
    tx = tx.merge(identity, on='TransactionID', how='left', validate='one_to_one')
    fields_num = ['TransactionAmt','card1','card2','card3','card5','addr1','dist1']
    fields_cat = ['ProductCD','card4','card6','P_emaildomain','DeviceType','DeviceInfo','id_31']
    fields = fields_num + fields_cat
    train, valid = tx[tx.TransactionDT < first], tx[(tx.TransactionDT >= first) & (tx.TransactionDT < second)]
    test = tx[tx.TransactionDT >= second]
    result = (train[fields], train.isFraud.to_numpy(), valid[fields], valid.isFraud.to_numpy(), fields_cat, fields_num,
              {'method':'70/15/15 chronological TransactionDT; identity left-joined one_to_one; final 15% sealed and never loaded for tuning', 'excluded':['isFraud','TransactionID','TransactionDT']})
    return (*result, test[fields], test.isFraud.to_numpy()) if include_test else result


def baf(path, cap, include_test=False):
    data = read_base(path)
    # ``split_months`` returns both the three chronological partitions and
    # their audit boundaries.  Keep the last month sealed: this validation
    # experiment must not materialise it at all.
    (train, valid, sealed_test), _boundaries = split_months(data)
    result = (train[BAF_FEATURES], train.fraud_bool.to_numpy(), valid[BAF_FEATURES], valid.fraud_bool.to_numpy(), BAF_CATS, BAF_NUMS,
              {'method':'earlier months train, penultimate month validation; final month sealed', 'excluded':['fraud_bool','month','device_fraud_count','days_since_request','credit_risk_score']})
    return (*result, sealed_test[BAF_FEATURES], sealed_test.fraud_bool.to_numpy()) if include_test else result


def ibm(path, cap, include_test=False):
    validate_ibm(path)
    first, second, total, available = ibm_boundaries(path)
    parts, counts = ibm_splits(path, first, second, available, cap, include_test)
    train, valid = parts['train'], parts['validation']
    numbers = ['amount_paid','log_amount_paid','same_bank','same_currency','hour','weekday']
    cats = ['receiving_currency','payment_currency','payment_format']
    result = (*train, *valid, cats, numbers,
              {'method':'70/15/15 timestamp; final 15% sealed and never loaded for tuning', 'rows_total':total, 'excluded':['Is Laundering','Account','Account.1','Amount Received','raw bank identifiers']})
    return (*result, *parts['test']) if include_test else result


def to_dense(transform, frame):
    value = transform.transform(frame)
    return value.toarray().astype('float32') if hasattr(value, 'toarray') else np.asarray(value, dtype='float32')


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Resolve ``auto`` explicitly so CPU-only Kaggle sessions are supported."""
    if requested == 'auto':
        return 'cuda' if cuda_available else 'cpu'
    if requested == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA requested but unavailable')
    return requested


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def final_evaluation(args) -> dict:
    """Evaluate an immutable validation-selected export once on sealed test windows.

    This path intentionally has no training code: thresholds, preprocessing and
    weights are read from the existing experiment output.  Results go into a
    distinct report so the serving manifest cannot accidentally turn test
    metrics into tuning input.
    """
    output = args.output_dir
    manifest_path = output / 'unified.metrics.json'
    weights_path = output / 'unified_weights.npz'
    preprocessors_path = output / 'unified_preprocessors.joblib'
    final_path = output / 'unified.final_test.json'
    if final_path.exists():
        raise FileExistsError(f'Final evaluation already exists at {final_path}; refusing to re-evaluate this run')
    if not all(path.is_file() for path in (manifest_path, weights_path, preprocessors_path)):
        raise FileNotFoundError('Final evaluation requires an existing validation-selected Unified export')
    report = json.loads(manifest_path.read_text(encoding='utf-8'))
    if report.get('model_kind') != 'izqar_unified_mtl' or set(report.get('tasks', {})) != set(TASKS):
        raise ValueError('Unsupported Unified manifest for final evaluation')
    preprocessors = joblib.load(preprocessors_path)
    if set(preprocessors) != set(TASKS):
        raise ValueError('Unified preprocessors do not contain exactly four tasks')
    network = UnifiedNetwork.from_npz(weights_path, report['input_dimensions'])
    sources = {
        'paysim': lambda: paysim(args.paysim, args.max_train_rows, include_test=True),
        'ieee_cis': lambda: ieee(args.ieee_dir, args.max_train_rows, include_test=True),
        'baf_base': lambda: baf(args.baf, args.max_train_rows, include_test=True),
        'ibm_aml': lambda: ibm(args.ibm_aml, args.max_train_rows, include_test=True),
    }
    outputs = {}
    for name in TASKS:
        loaded = sources[name]()
        x_test, y_test = loaded[7], loaded[8]
        x_test, y_test = balanced_cap(x_test, y_test, args.max_final_rows)
        if len(np.unique(y_test)) != 2:
            raise ValueError(f'{name} final test must contain both classes')
        scores = network.predict_scores(name, to_dense(preprocessors[name], x_test))
        threshold = report['tasks'][name].get('threshold')
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f'{name} has no valid validation-selected threshold')
        outputs[name] = {'threshold_from_validation': float(threshold), 'final_test_metrics': metrics(y_test, scores, float(threshold))}
    final = {
        'format_version': 1,
        'evaluation_type': 'one_time_sealed_final_test',
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_version': report.get('model_version'),
        'manifest_sha256': sha256_file(manifest_path),
        'weights_sha256': sha256_file(weights_path),
        'preprocessors_sha256': sha256_file(preprocessors_path),
        'max_final_rows': args.max_final_rows,
        'tasks': outputs,
        'warning': ('One-time final evaluation only. These results must not be used to select thresholds, epochs, '
                    'features, model architecture, or a replacement for specialised models.'),
    }
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paysim', type=Path, required=True)
    parser.add_argument('--ieee-dir', type=Path, required=True)
    parser.add_argument('--baf', type=Path, required=True)
    parser.add_argument('--ibm-aml', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['experiment', 'final'], default='experiment',
                        help='experiment trains using train/validation only; final evaluates an existing frozen export once')
    parser.add_argument('--max-train-rows', type=int, default=150_000)
    parser.add_argument('--max-validation-rows', type=int, default=75_000)
    parser.add_argument('--max-final-rows', type=int, default=None,
                        help='optional deterministic cap for a one-time final test; omit for the entire sealed window')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_validation_rows, args.epochs, args.batch_size) < 1 or (args.max_final_rows is not None and args.max_final_rows < 1):
        parser.error('all limits must be positive')
    if args.mode == 'final':
        result = final_evaluation(args)
        print(json.dumps({'final_evaluation': str(args.output_dir / 'unified.final_test.json'), 'tasks': result['tasks']}, ensure_ascii=False, indent=2))
        return
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError('PyTorch is required only for Kaggle training; local inference does not require it.') from exc
    random.seed(RNG); np.random.seed(RNG); torch.manual_seed(RNG)
    device = resolve_device(args.device, torch.cuda.is_available())
    sources = {'paysim': lambda: paysim(args.paysim, args.max_train_rows), 'ieee_cis': lambda: ieee(args.ieee_dir, args.max_train_rows),
               'baf_base': lambda: baf(args.baf, args.max_train_rows), 'ibm_aml': lambda: ibm(args.ibm_aml, args.max_train_rows)}
    prepared, preprocessors, dimensions, metadata = {}, {}, {}, {}
    for name in TASKS:
        xtr, ytr, xva, yva, cats, nums, audit = sources[name]()
        xtr, ytr = balanced_cap(xtr, ytr, args.max_train_rows); xva, yva = balanced_cap(xva, yva, args.max_validation_rows)
        if len(np.unique(ytr)) != 2 or len(np.unique(yva)) != 2:
            raise ValueError(f'{name} selected train and validation samples must each contain both classes; increase the row limits')
        pre = transformer(nums, cats).fit(xtr)
        prepared[name] = (to_dense(pre, xtr), ytr.astype('float32'), to_dense(pre, xva), yva.astype('int8'))
        preprocessors[name] = pre; dimensions[name] = int(prepared[name][0].shape[1]); metadata[name] = {**audit, 'train_used':len(ytr), 'validation_used':len(yva), 'positive_train':int(ytr.sum()), 'feature_schema': {'all': nums + cats, 'numerical': nums, 'categorical': cats}}
    class MTL(nn.Module):
        def __init__(self):
            super().__init__(); self.encoders = nn.ModuleDict({n:nn.Linear(dimensions[n],32) for n in TASKS}); self.shared = nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU()); self.heads=nn.ModuleDict({n:nn.Linear(16,1) for n in TASKS})
        def forward(self, name, x): return self.heads[name](self.shared(F.relu(self.encoders[name](x)))).squeeze(1)
    model = MTL().to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    started = time.monotonic()
    # Verify each task reaches the same backbone before optimisation.
    gradient_audit = {}
    for name in TASKS:
        x, y, _, _ = prepared[name]; take = min(len(x), args.batch_size); optimiser.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(name, torch.tensor(x[:take], device=device)), torch.tensor(y[:take], device=device), pos_weight=torch.tensor([(len(y)-y.sum())/max(y.sum(),1)],device=device)); loss.backward()
        gradient_audit[name] = round(float(model.shared[0].weight.grad.norm().detach().cpu()), 8)
    if any(value <= 0 for value in gradient_audit.values()): raise RuntimeError(f'No shared-backbone gradient: {gradient_audit}')
    best, best_state, history = float('-inf'), None, []
    for epoch in range(args.epochs):
        model.train(); losses = {}
        for name in TASKS:
            x, y, _, _ = prepared[name]; order = np.random.permutation(len(x)); limit = min(len(x), 50_000)
            values=[]
            for start in range(0, limit, args.batch_size):
                ix=order[start:start+args.batch_size]; xb=torch.tensor(x[ix],device=device); yb=torch.tensor(y[ix],device=device)
                weight=torch.tensor([(len(y)-y.sum())/max(y.sum(),1)],device=device); optimiser.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model(name, xb),yb,pos_weight=weight); loss.backward(); optimiser.step(); values.append(float(loss.detach().cpu()))
            losses[name]=round(float(np.mean(values)),6)
        model.eval(); score=0; epoch_metrics={}
        with torch.no_grad():
            for name in TASKS:
                _,_,x,y=prepared[name]; s=torch.sigmoid(model(name,torch.tensor(x,device=device))).cpu().numpy(); epoch_metrics[name]=round(float(average_precision_score(y,s)),6); score += epoch_metrics[name]
        history.append({'epoch':epoch+1,'loss':losses,'validation_pr_auc':epoch_metrics})
        if score > best:
            best, best_state = score, {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch + 1, 'state_dict': best_state, 'optimizer_state': optimiser.state_dict(),
                        'validation_pr_auc_sum': best, 'seed': RNG}, args.output_dir/'unified_checkpoint.pt')
    model.load_state_dict(best_state); args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_dir/'unified_weights.npz', **{k:v.numpy().astype('float32') for k,v in best_state.items()})
    joblib.dump(preprocessors, args.output_dir/'unified_preprocessors.joblib')
    outputs={}
    model.eval()
    with torch.no_grad():
        for name in TASKS:
            _,_,x,y=prepared[name]; scores=torch.sigmoid(model(name,torch.tensor(x,device=device))).cpu().numpy(); threshold=choose_threshold(y,scores); outputs[name]={'threshold':threshold,'validation_metrics':metrics(y,scores,threshold)}
    report={'format_version':1,'model_kind':'izqar_unified_mtl','model_version':'izqar-unified-mtl-v1','trained_at_utc':datetime.now(timezone.utc).isoformat(),'serialization':'numpy_shared_backbone_v1','tasks':outputs,'input_dimensions':dimensions,'gradient_audit_shared_backbone_l2':gradient_audit,'training':{'epochs':args.epochs,'batch_size':args.batch_size,'device':device,'loss':'per-task positive-weighted BCE, equal task update rounds','seed':RNG,'best_validation_pr_auc_sum':best,'seconds':round(time.monotonic()-started,3),'history':history},'task_metadata':metadata,'warning':'Experimental multi-task model. Validation only; no final tests were used. Scores are uncalibrated and individual specialized models remain the primary detectors unless comparison supports replacement.'}
    (args.output_dir/'unified.metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output_dir),'tasks':outputs,'gradient_audit':gradient_audit},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
