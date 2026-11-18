"""Read configuration from the repository .env; refuse weak demo credentials."""
from __future__ import annotations
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlsplit

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / 'frontend').exists() else BACKEND_ROOT
load_dotenv(PROJECT_ROOT / '.env')

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./izqar.db')
API_KEY = os.environ.get('IZQAR_API_KEY', '')
MODEL_PATH = Path(os.environ.get('IZQAR_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'ieee_model.joblib')))
PAYSIM_MODEL_PATH = Path(os.environ.get('IZQAR_PAYSIM_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'paysim_model.joblib')))
BAF_MODEL_PATH = Path(os.environ.get('IZQAR_BAF_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'baf_base_v1_portable.metrics.json')))
IBM_AML_MODEL_PATH = Path(os.environ.get('IZQAR_IBM_AML_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'ibm_aml_hi_small_v1_portable.metrics.json')))
UNIFIED_MODEL_PATH = Path(os.environ.get('IZQAR_UNIFIED_MODEL_PATH', str(BACKEND_ROOT / 'models' / 'unified.metrics.json')))
IBM_AML_GRAPH_REPORT_PATH = Path(os.environ.get('IZQAR_IBM_AML_GRAPH_REPORT_PATH', str(BACKEND_ROOT / 'models' / 'experiments' / 'ibm_graph_report.json')))
CORS_ORIGINS = [s.strip() for s in os.environ.get('IZQAR_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if s.strip()]


def validate_cors_origins(origins: list[str]) -> None:
    """Require explicit browser origins, never a wildcard or URL path.

    The local MVP uses an API key rather than browser identity.  A permissive
    CORS policy would therefore make accidental deployment exposure easier.
    Operators may explicitly configure a trusted HTTP(S) origin, but must use
    its origin form only (scheme, host, optional port).
    """
    if not origins:
        raise RuntimeError('IZQAR_CORS_ORIGINS must contain at least one explicit HTTP(S) origin.')
    for origin in origins:
        if origin == '*':
            raise RuntimeError('IZQAR_CORS_ORIGINS must not contain wildcard origin *.')
        parsed = urlsplit(origin)
        if (parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.path not in {'', '/'}
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise RuntimeError('Each IZQAR_CORS_ORIGINS entry must be an explicit HTTP(S) origin without path or credentials.')
        if parsed.path == '/':
            raise RuntimeError('Each IZQAR_CORS_ORIGINS entry must omit the trailing slash.')


def validate_config() -> None:
    if len(API_KEY) < 24 or re.match(r'^(CHANGE_ME|replace-me|your[-_]?api[-_]?key|example)', API_KEY, re.IGNORECASE):
        raise RuntimeError('Set IZQAR_API_KEY in .env to a random secret of at least 24 characters (see README).')
    validate_cors_origins(CORS_ORIGINS)
