# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
$ErrorActionPreference = 'Stop'
$project = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $project
if (-not (Test-Path '.env')) {
  $secret = (& py -3 -c 'import secrets; print(secrets.token_urlsafe(32))').Trim()
  if ($LASTEXITCODE -ne 0 -or $secret.Length -lt 24) { throw 'Python 3 is required; unable to generate key.' }
  $envText = "IZQAR_API_KEY=$secret`nDATABASE_URL=sqlite:///./izqar.db`nIZQAR_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`n"
  [System.IO.File]::WriteAllText((Join-Path $project ".env"), $envText, [System.Text.UTF8Encoding]::new($false))
  Write-Host 'Created .env with a new local API key.' -ForegroundColor Green
}
$backend = Join-Path $project 'backend'
Set-Location $backend
if (-not (Test-Path '.venv')) { & py -3 -m venv .venv }
$python = Join-Path $backend '.venv\Scripts\python.exe'
& $python -m pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
Write-Host 'IZQAR demo ready at http://127.0.0.1:8000/demo/' -ForegroundColor Green
Write-Host 'API key: open .env in the project root. Press Ctrl+C to stop.'
Start-Process 'http://127.0.0.1:8000/demo/'
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
