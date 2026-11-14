"""IZQAR local-only hackathon MVP API."""
from __future__ import annotations
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from .config import API_KEY, CORS_ORIGINS, BACKEND_ROOT, MODEL_PATH, PAYSIM_MODEL_PATH, BAF_MODEL_PATH, IBM_AML_MODEL_PATH, UNIFIED_MODEL_PATH, IBM_AML_GRAPH_REPORT_PATH, validate_config
from .database import Base, engine, get_db
from .models import Transaction, InvestigationEvent, InvestigationMlAssessment
from .schemas import (TransferIn, TransactionOut, SimulationIn, ReviewIn, CommentIn,
                      InvestigationEventOut, InvestigationListOut, InvestigationDetailOut,
                      IeeeFeaturesIn, PaySimFeaturesIn, BafFeaturesIn, IbmAmlFeaturesIn, UnifiedFeaturesIn,
                      MlAssessmentIn, InvestigationMlAssessmentOut)
from .risk import assess
from .graph import build_graph
from .ibm_graph_report import read_public_summary
from .model_registry import list_registry
from .model_registry import FORBIDDEN
from .ml import load_model, load_paysim_model, load_baf_model, load_ibm_aml_model, load_unified_model, predict_ieee, predict_paysim, predict_baf, predict_ibm_aml, predict_unified


MAX_API_BODY_BYTES = 64 * 1024


class ApiBodyLimitMiddleware:
    """Bound request bytes even when a client omits or lies about Content-Length."""

    def __init__(self, app, max_body_bytes: int = MAX_API_BODY_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope.get('path', '').startswith('/api/v1/'):
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get('headers', [])}
        raw_length = headers.get(b'content-length')
        if raw_length is not None:
            try:
                declared = int(raw_length)
                if declared < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse(status_code=400, content={'detail': 'Invalid Content-Length'})(scope, receive, send)
                return
            if declared > self.max_body_bytes:
                await JSONResponse(status_code=413, content={'detail': f'Request body exceeds {self.max_body_bytes} byte limit'})(scope, receive, send)
                return
        received = 0
        chunks = []
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            if message['type'] != 'http.request':
                continue
            chunk = message.get('body', b'')
            received += len(chunk)
            if received > self.max_body_bytes:
                await JSONResponse(status_code=413, content={'detail': f'Request body exceeds {self.max_body_bytes} byte limit'})(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get('more_body', False):
                break

        async def replay_receive():
            if chunks:
                body = chunks.pop(0)
                return {'type': 'http.request', 'body': body, 'more_body': bool(chunks)}
            return {'type': 'http.disconnect'}

        # The body is bounded before FastAPI gets a receive callable, so its
        # JSON parser cannot convert an over-limit transport error into 400.
        await self.app(scope, replay_receive, send)

@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title='IZQAR Fraud Prevention MVP', version='0.1.0', lifespan=lifespan)
# Add CORS last: it wraps the limiter so browser clients receive CORS headers
# even for an over-limit request rejected before FastAPI parses its JSON body.
app.add_middleware(ApiBodyLimitMiddleware, max_body_bytes=MAX_API_BODY_BYTES)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False,
                   allow_methods=['GET', 'POST', 'PATCH'], allow_headers=['X-API-Key', 'Content-Type'])
app.mount('/demo', StaticFiles(directory=Path(__file__).parent / 'static', html=True), name='demo')


def require_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail='Неверный или отсутствующий X-API-Key')

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'izqar-api'}

# Every data endpoint requires the configured demo API key.
@app.post('/api/v1/transfers', response_model=TransactionOut, dependencies=[Depends(require_key)])
def transfer(payload: TransferIn, db: Session = Depends(get_db)):
    return store_transfer(payload, db, 'manual')


def store_transfer(payload: TransferIn, db: Session, scenario: str):
    if payload.sender == payload.recipient:
        raise HTTPException(status_code=422, detail='Нельзя отправлять средства на тот же счёт')
    result = assess(payload, db)
    tx = Transaction(sender=payload.sender, recipient=payload.recipient,
                     amount=payload.amount, currency=payload.currency,
                     device_changed=payload.device_changed, external_flag=payload.external_flag,
                     scenario=scenario, risk_score=result.score, risk_level=result.level,
                     decision=result.decision, reasons=result.reasons, indicators=result.indicators)
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx

@app.get('/api/v1/transactions', response_model=list[TransactionOut], dependencies=[Depends(require_key)])
def transactions(limit: int = Query(default=60, ge=1, le=200), db: Session = Depends(get_db)):
    return list(db.scalars(select(Transaction).order_by(Transaction.created_at.desc(), Transaction.id.desc()).limit(limit)))

@app.patch('/api/v1/transactions/{tx_id}/review', response_model=TransactionOut, dependencies=[Depends(require_key)])
def review(tx_id: str, payload: ReviewIn, db: Session = Depends(get_db)):
    tx = db.get(Transaction, tx_id)
    if tx is None:
        raise HTTPException(status_code=404, detail='Транзакция не найдена')
    previous = tx.review_state
    tx.review_state = payload.review_state
    db.add(InvestigationEvent(transaction_id=tx.id, action='decision',
                              from_state=previous, to_state=payload.review_state,
                              comment=payload.comment))
    db.commit()
    db.refresh(tx)
    return tx


@app.get('/api/v1/investigations', response_model=InvestigationListOut, dependencies=[Depends(require_key)])
def investigations(
    status: Literal['all', 'pending', 'in_review', 'step_up', 'escalated', 'allowed',
                    'confirmed_fraud', 'false_positive'] = 'all',
    risk_level: Literal['all', 'low', 'medium', 'high'] = 'all',
    search: str = Query(default='', max_length=64),
    sort: Literal['newest', 'highest_risk'] = 'highest_risk',
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
    db: Session = Depends(get_db),
):
    # The queue starts with suspicious transactions, but resolved and actively
    # investigated cases remain visible. Filtering is server-side and bounded.
    filters = [or_(Transaction.risk_level.in_(['medium', 'high']),
                   Transaction.review_state != 'pending')]
    if status != 'all':
        filters.append(Transaction.review_state == status)
    if risk_level != 'all':
        filters.append(Transaction.risk_level == risk_level)
    if search:
        literal = search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        term = f'%{literal}%'
        filters.append(or_(Transaction.sender.ilike(term, escape='\\'),
                           Transaction.recipient.ilike(term, escape='\\'),
                           Transaction.id.ilike(term, escape='\\')))
    total = db.scalar(select(func.count(Transaction.id)).where(*filters)) or 0
    ordering = ((Transaction.risk_score.desc(), Transaction.created_at.desc(), Transaction.id.desc())
                if sort == 'highest_risk' else (Transaction.created_at.desc(), Transaction.id.desc()))
    items = list(db.scalars(select(Transaction).where(*filters).order_by(*ordering).offset(offset).limit(limit)))
    return {'total': total, 'items': items}


@app.get('/api/v1/investigations/{tx_id}', response_model=InvestigationDetailOut,
         dependencies=[Depends(require_key)])
def investigation_detail(tx_id: str, db: Session = Depends(get_db)):
    tx = db.get(Transaction, tx_id)
    if tx is None:
        raise HTTPException(status_code=404, detail='Транзакция не найдена')
    events_total = db.scalar(select(func.count(InvestigationEvent.id))
                             .where(InvestigationEvent.transaction_id == tx_id)) or 0
    events = list(reversed(list(db.scalars(select(InvestigationEvent)
                                           .where(InvestigationEvent.transaction_id == tx_id)
                                           .order_by(InvestigationEvent.created_at.desc(), InvestigationEvent.id.desc())
                                           .limit(500)))))
    accounts = (tx.sender, tx.recipient)
    related = list(db.scalars(select(Transaction)
                              .where(Transaction.id != tx_id,
                                     or_(Transaction.sender.in_(accounts),
                                         Transaction.recipient.in_(accounts)))
                              .order_by(Transaction.created_at.desc(), Transaction.id.desc())
                              .limit(50)))
    assessments = list(reversed(list(db.scalars(select(InvestigationMlAssessment)
                                                 .where(InvestigationMlAssessment.transaction_id == tx_id)
                                                 .order_by(InvestigationMlAssessment.created_at.desc(), InvestigationMlAssessment.id.desc())
                                                 .limit(100)))))
    return {'transaction': tx, 'events': events, 'events_total': events_total,
            'related_transactions': related, 'ml_assessments': assessments}


@app.post('/api/v1/investigations/{tx_id}/comments', response_model=InvestigationEventOut,
          dependencies=[Depends(require_key)])
def add_investigation_comment(tx_id: str, payload: CommentIn, db: Session = Depends(get_db)):
    tx = db.get(Transaction, tx_id)
    if tx is None:
        raise HTTPException(status_code=404, detail='Транзакция не найдена')
    event = InvestigationEvent(transaction_id=tx.id, action='comment',
                               from_state=tx.review_state, to_state=tx.review_state,
                               comment=payload.comment)
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@app.post('/api/v1/investigations/{tx_id}/ml-assessments', response_model=InvestigationMlAssessmentOut,
          dependencies=[Depends(require_key)])
def add_ml_assessment(tx_id: str, payload: MlAssessmentIn, db: Session = Depends(get_db)):
    """Run a selected model server-side and preserve provenance, not raw inputs."""
    tx = db.get(Transaction, tx_id)
    if tx is None:
        raise HTTPException(status_code=404, detail='Транзакция не найдена')
    forbidden = sorted(set(payload.features) & FORBIDDEN)
    if forbidden:
        raise HTTPException(status_code=422, detail=f'Forbidden assessment features: {forbidden}')
    model_map = {
        'paysim': predict_paysim, 'ieee_cis': predict_ieee, 'baf_base': predict_baf, 'ibm_aml': predict_ibm_aml,
        'unified_paysim': lambda features: predict_unified('paysim', features),
        'unified_ieee_cis': lambda features: predict_unified('ieee_cis', features),
        'unified_baf_base': lambda features: predict_unified('baf_base', features),
        'unified_ibm_aml': lambda features: predict_unified('ibm_aml', features),
    }
    try:
        prediction = model_map[payload.model](payload.features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f'ML model is incompatible: {exc}') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if prediction is None:
        raise HTTPException(status_code=503, detail='Selected ML model is not installed')
    assessment = InvestigationMlAssessment(
        transaction_id=tx.id, model_key=payload.model, model_version=prediction['model_version'],
        source=prediction['source'], score_kind=prediction['score_kind'], risk_score=prediction['risk_score'],
        threshold=prediction['threshold'], flagged=prediction['flagged'], warning=prediction['warning'],
        feature_names=sorted(payload.features),
    )
    db.add(assessment)
    db.add(InvestigationEvent(transaction_id=tx.id, action='ml_assessment', from_state=tx.review_state,
                              to_state=tx.review_state, comment=f'ML assessment: {payload.model}'))
    db.commit()
    db.refresh(assessment)
    return assessment

@app.get('/api/v1/metrics', dependencies=[Depends(require_key)])
def metrics(db: Session = Depends(get_db)):
    count = db.scalar(select(func.count(Transaction.id))) or 0
    total_volume = db.scalar(select(func.sum(Transaction.amount))) or 0
    levels = dict(db.execute(select(Transaction.risk_level, func.count(Transaction.id)).group_by(Transaction.risk_level)).all())
    reviewed = db.scalar(select(func.count(Transaction.id)).where(Transaction.review_state != 'pending')) or 0
    return {'transactions': count, 'simulated_volume_kzt': round(total_volume, 2),
            'high': levels.get('high', 0), 'medium': levels.get('medium', 0),
            'low': levels.get('low', 0), 'reviewed': reviewed,
            'disclaimer': 'Синтетические операции; показатели не являются метриками реального банка.'}

@app.get('/api/v1/graph', dependencies=[Depends(require_key)])
def graph(limit: int = Query(default=500, ge=1, le=1000), window_minutes: int = Query(default=1440, ge=1, le=43_200), db: Session = Depends(get_db)):
    return build_graph(db, limit, window_minutes)


@app.get('/api/v1/graph/ibm-aml-report', dependencies=[Depends(require_key)])
def ibm_aml_graph_report():
    """Expose only aggregate signals from the optional offline IBM analysis."""
    return read_public_summary(IBM_AML_GRAPH_REPORT_PATH)

@app.get('/api/v1/model/status', dependencies=[Depends(require_key)])
def model_status():
    def summary(artifact):
        if artifact is None:
            return None
        return {'features': len(artifact['features']), 'feature_names': artifact['features'],
                'test_metrics': artifact.get('test_metrics'), 'threshold': artifact['threshold'],
                'model_version': artifact.get('model_version', 'unknown'),
                'dataset': artifact.get('dataset', 'unknown'),
                'trained_at_utc': artifact.get('trained_at_utc'),
                'preprocessing_compatibility': artifact.get('preprocessing_compatibility'),
                'split_counts': artifact.get('split_counts', {}),
                'score_kind': 'uncalibrated_ml_score',
                'validation_ranking_metrics': artifact.get('validation_ranking_metrics'),
                'validation_operating_profiles': artifact.get('validation_operating_profiles'),
                # These are present only in newer PaySim reports.  They remain
                # validation diagnostics, never live scoring inputs or labels.
                'validation_type_slices': artifact.get('validation_type_slices'),
                'validation_robustness': artifact.get('validation_robustness'),
                'feature_importance': artifact.get('feature_importance', {}).get('top_30', [])[:10]}
    try:
        artifact = load_model()
    except RuntimeError as exc:
        artifact, ieee_error = None, str(exc)
    else:
        ieee_error = None
    try:
        paysim = load_paysim_model()
    except RuntimeError as exc:
        paysim, paysim_error = None, str(exc)
    else:
        paysim_error = None
    try:
        baf = load_baf_model()
    except RuntimeError as exc:
        baf, baf_error = None, str(exc)
    else:
        baf_error = None
    try:
        ibm_aml = load_ibm_aml_model()
    except RuntimeError as exc:
        ibm_aml, ibm_aml_error = None, str(exc)
    else:
        ibm_aml_error = None
    try:
        unified = load_unified_model()
    except RuntimeError as exc:
        unified, unified_error = None, str(exc)
    else:
        unified_error = None
    unified_summary = None
    if unified:
        tasks = {task: dict(item) for task, item in unified['report']['tasks'].items()}
        if unified['final_evaluation']:
            for task, item in unified['final_evaluation']['tasks'].items():
                tasks[task]['final_test_metrics'] = item['final_test_metrics']
        unified_summary = {'model_version': unified['report']['model_version'], 'tasks': tasks,
                           # Expose the serving schema, not labels or preprocessing internals.  The
                           # React console needs it to construct an exact, validated request for a
                           # selected Unified task rather than guessing a cross-dataset payload.
                           'task_schemas': {task: schema['all'] for task, schema in unified['schema'].items()},
                           'score_kind': 'uncalibrated_ml_score', 'experimental': True,
                           'preprocessing_compatibility': unified['preprocessing_compatibility'],
                           'final_evaluation_available': bool(unified['final_evaluation']),
                           'final_evaluation_error': unified['final_evaluation_error']}
    return {'ready': bool(artifact or paysim or baf or ibm_aml), 'mode': 'separate_models',
            'ieee_cis': summary(artifact),
            'paysim': summary(paysim),
            'baf_base': summary(baf),
            'ibm_aml': summary(ibm_aml),
            'unified_mtl': unified_summary,
            'message': 'No trained ML artifacts available; demo rules remain available.' if not (artifact or paysim or baf or ibm_aml) and not (ieee_error or paysim_error or baf_error or ibm_aml_error) else None,
            'artifact_errors': {'ieee_cis': ieee_error, 'paysim': paysim_error, 'baf_base': baf_error, 'ibm_aml': ibm_aml_error, 'unified_mtl': unified_error}}


@app.get('/api/v1/model/registry', dependencies=[Depends(require_key)])
def model_registry():
    return list_registry({'ieee_cis': MODEL_PATH, 'paysim': PAYSIM_MODEL_PATH,
                          'baf_base': BAF_MODEL_PATH, 'ibm_aml': IBM_AML_MODEL_PATH,
                          'izqar_unified_mtl': UNIFIED_MODEL_PATH},
                         BACKEND_ROOT / 'models')


@app.post('/api/v1/unified/{task}/predict', dependencies=[Depends(require_key)])
def unified_predict(task: str, payload: UnifiedFeaturesIn):
    try:
        artifact = load_unified_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f'Incompatible Unified model artifact: {exc}') from exc
    if artifact is None:
        raise HTTPException(status_code=503, detail='Unified Intelligence artifact is not installed; it remains experimental')
    try:
        return predict_unified(task, payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post('/api/v1/ieee/predict', dependencies=[Depends(require_key)])
def ieee_predict(payload: IeeeFeaturesIn):
    try:
        artifact = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f'Incompatible IEEE-CIS model artifact: {exc}') from exc
    if artifact is None:
        raise HTTPException(status_code=503, detail='Сначала обучите модель IEEE-CIS; см. README')
    try:
        return predict_ieee(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post('/api/v1/paysim/predict', dependencies=[Depends(require_key)])
def paysim_predict(payload: PaySimFeaturesIn):
    try:
        artifact = load_paysim_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f'Incompatible PaySim model artifact: {exc}') from exc
    if artifact is None:
        raise HTTPException(status_code=503, detail='Сначала обучите модель PaySim; см. README')
    try:
        return predict_paysim(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/api/v1/baf/predict', dependencies=[Depends(require_key)])
def baf_predict(payload: BafFeaturesIn):
    try:
        artifact = load_baf_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f'Incompatible BAF model artifact: {exc}') from exc
    if artifact is None:
        raise HTTPException(status_code=503, detail='BAF Base model is not trained or installed; see README')
    try:
        return predict_baf(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/api/v1/ibm-aml/predict', dependencies=[Depends(require_key)])
def ibm_aml_predict(payload: IbmAmlFeaturesIn):
    try:
        artifact = load_ibm_aml_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f'Incompatible IBM AML model artifact: {exc}') from exc
    if artifact is None:
        raise HTTPException(status_code=503, detail='IBM AML model is not trained or installed; see README')
    try:
        return predict_ibm_aml(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post('/api/v1/demo/simulate', response_model=list[TransactionOut], dependencies=[Depends(require_key)])
def simulate(payload: SimulationIn, db: Session = Depends(get_db)):
    # Each scenario uses the same real assessment path as a manually submitted transfer.
    samples = {
        'normal': [('CLIENT-101', 'SHOP-021', 12_400, False, False)],
        'new_recipient': [('CLIENT-101', 'UNFAMILIAR-909', 2_450_000, True, True)],
        'velocity': [
            ('CLIENT-202', 'SHOP-301', 125_000, False, False),
            ('CLIENT-202', 'SHOP-302', 150_000, False, False),
            ('CLIENT-202', 'SHOP-303', 180_000, False, False),
            ('CLIENT-202', 'SHOP-304', 395_000, True, False),
        ],
        'fan_in': [
            ('CLIENT-401', 'MULE-001', 415_000, False, False),
            ('CLIENT-402', 'MULE-001', 310_000, False, False),
            ('CLIENT-403', 'MULE-001', 500_000, False, False),
            ('CLIENT-404', 'MULE-001', 655_000, True, False),
            ('MULE-001', 'MULE-002', 1_100_000, False, True),
            ('MULE-001', 'MULE-003', 650_000, False, False),
        ],
        'chain': [
            ('SOURCE-110', 'LINK-111', 750_000, False, False),
            ('LINK-111', 'LINK-112', 700_000, False, False),
            ('LINK-112', 'LINK-113', 640_000, False, False),
            ('LINK-113', 'DEST-114', 610_000, False, False),
        ],
        'cycle': [
            ('CYCLE-201', 'CYCLE-202', 420_000, False, False),
            ('CYCLE-202', 'CYCLE-203', 400_000, False, False),
            ('CYCLE-203', 'CYCLE-201', 385_000, True, False),
        ],
    }
    result = []
    for sender, recipient, amount, device_changed, external_flag in samples[payload.scenario]:
        result.append(store_transfer(TransferIn(sender=sender, recipient=recipient, amount=amount,
                                                device_changed=device_changed, external_flag=external_flag),
                                     db, payload.scenario))
    return result
