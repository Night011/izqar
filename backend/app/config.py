"""Read configuration from the repository .env; refuse weak demo credentials."""
from __future__ import annotations
import os
import re
from pathlib import Path
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / 'frontend').exists() else BACKEND_ROOT
load_dotenv(PROJECT_ROOT / '.env')

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./izqar.db')
API_KEY = os.environ.get('IZQAR_API_KEY', '')
MODEL_PATH = Path(os.environ.get('IZQAR_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'ieee_model.joblib')))
PAYSIM_MODEL_PATH = Path(os.environ.get('IZQAR_PAYSIM_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'paysim_model.joblib')))
BAF_MODEL_PATH = Path(os.environ.get('IZQAR_BAF_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'baf_base_v1_portable.metrics.json')))
CORS_ORIGINS = [s.strip() for s in os.environ.get('IZQAR_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if s.strip()]


def validate_config() -> None:
    if len(API_KEY) < 24 or re.match(r'^(CHANGE_ME|replace-me|your[-_]?api[-_]?key|example)', API_KEY, re.IGNORECASE):
        raise RuntimeError('Set IZQAR_API_KEY in .env to a random secret of at least 24 characters (see README).')
