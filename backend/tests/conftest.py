"""Isolate tests from user data; no real banking data."""
import os
import tempfile
from pathlib import Path
import pytest

_tmpdir = tempfile.TemporaryDirectory()
os.environ['IZQAR_API_KEY'] = 'test-only-secret-key-longer-than-32-characters'
os.environ['DATABASE_URL'] = f"sqlite:///{Path(_tmpdir.name) / 'tests.db'}"
os.environ['IZQAR_MODEL_PATH'] = str(Path(_tmpdir.name) / 'missing_ieee_model.joblib')
os.environ['IZQAR_PAYSIM_MODEL_PATH'] = str(Path(_tmpdir.name) / 'missing_paysim_model.joblib')
os.environ['IZQAR_BAF_MODEL_PATH'] = str(Path(_tmpdir.name) / 'missing_baf_model.joblib')

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.database import Base, engine  # noqa: E402

@pytest.fixture(scope='session', autouse=True)
def close_test_database():
    """Release SQLite's Windows file handle before TemporaryDirectory cleanup."""
    yield
    engine.dispose()
    _tmpdir.cleanup()

@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def auth():
    return {'X-API-Key': os.environ['IZQAR_API_KEY']}
