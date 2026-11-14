"""Configuration must fail closed before the local API starts."""
import pytest

from app.config import validate_cors_origins


def test_cors_accepts_explicit_origins_only():
    validate_cors_origins(['http://localhost:5173', 'https://analyst.example.test:8443'])
    for invalid in ([], ['*'], ['file:///tmp/app'], ['https://analyst.example.test/path'],
                    ['https://user:password@analyst.example.test'], ['https://analyst.example.test/']):
        with pytest.raises(RuntimeError):
            validate_cors_origins(invalid)
