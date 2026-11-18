"""Exercise the real PowerShell key handling in an isolated temporary project."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest


POWERSHELL = shutil.which('powershell.exe') or shutil.which('powershell')


@pytest.mark.skipif(POWERSHELL is None, reason='Windows PowerShell is unavailable')
def test_start_local_config_preserves_real_secrets_and_repairs_templates(tmp_path):
    source = Path(__file__).resolve().parents[2] / 'scripts' / 'start_local.ps1'
    target = tmp_path / 'scripts' / 'start_local.ps1'
    target.parent.mkdir()
    shutil.copyfile(source, target)
    env_file = tmp_path / '.env'

    def run():
        return subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                               '-File', str(target), '-ConfigureOnly'], capture_output=True,
                              text=True, timeout=15)

    first = run()
    assert first.returncode == 0, first.stderr
    generated = re.search(r'^IZQAR_API_KEY=(\S+)$', env_file.read_text(), re.M).group(1)
    assert len(generated) >= 24
    assert run().returncode == 0
    assert generated in env_file.read_text()

    env_file.write_text('IZQAR_API_KEY=CHANGE_ME\nPOSTGRES_PASSWORD=keep-this-secret\n')
    repaired = run()
    assert repaired.returncode == 0, repaired.stderr
    content = env_file.read_text()
    assert 'CHANGE_ME' not in content and 'POSTGRES_PASSWORD=keep-this-secret' in content

    env_file.write_text('IZQAR_API_KEY=existing-real-key\nPOSTGRES_PASSWORD=keep-this-secret\n')
    rejected = run()
    assert rejected.returncode != 0
    assert 'existing-real-key' in env_file.read_text()
    assert 'POSTGRES_PASSWORD=keep-this-secret' in env_file.read_text()

    env_file.write_text('IZQAR_API_KEY=CHANGE_ME\nIZQAR_API_KEY=another-value\n')
    assert run().returncode != 0
    assert 'IZQAR_API_KEY=CHANGE_ME' in env_file.read_text()
