"""IZQAR local-only hackathon MVP API."""
from __future__ import annotations
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from .config import API_KEY, CORS_ORIGINS, validate_config
from .database import Base, engine, get_db
from .models import Transaction
from .schemas import TransferIn, TransactionOut, SimulationIn, ReviewIn, IeeeFeaturesIn, PaySimFeaturesIn
from .risk import assess
from .graph import build_graph
from .ml import load_model, load_paysim_model, predict_ieee, predict_paysim

@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title='IZQAR Fraud Prevention MVP', version='0.1.0', lifespan=lifespan)
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
    tx.review_state = payload.review_state
    db.commit()
    db.refresh(tx)
    return tx

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

@app.get('/api/v1/model/status', dependencies=[Depends(require_key)])
def model_status():
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
    return {'ready': bool(artifact or paysim), 'mode': 'separate_models',
            'ieee_cis': None if not artifact else {'features': len(artifact['features']), 'test_metrics': artifact.get('test_metrics', {}), 'threshold': artifact['threshold']},
            'paysim': None if not paysim else {'features': len(paysim['features']), 'test_metrics': paysim.get('test_metrics', {}), 'threshold': paysim['threshold']},
            'message': 'No trained ML artifacts available; demo rules remain available.' if not (artifact or paysim) and not (ieee_error or paysim_error) else None,
            'artifact_errors': {'ieee_cis': ieee_error, 'paysim': paysim_error}}

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
    }
    result = []
    for sender, recipient, amount, device_changed, external_flag in samples[payload.scenario]:
        result.append(store_transfer(TransferIn(sender=sender, recipient=recipient, amount=amount,
                                                device_changed=device_changed, external_flag=external_flag),
                                     db, payload.scenario))
    return result
