# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
$ErrorActionPreference = 'Stop'
$project = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $project
$envPath = Join-Path $project '.env'
$invalidKey = $true
if (Test-Path $envPath) {
  $match = Select-String -Path $envPath -Pattern '^\s*IZQAR_API_KEY\s*=\s*(.*?)\s*$' | Select-Object -First 1
  if ($match) {
    $currentKey = $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
    $invalidKey = $currentKey.Length -lt 24 -or $currentKey -match '^(CHANGE_ME|replace-me|your[-_]?api[-_]?key|example)'
  }
}
if ($invalidKey) {
  $secret = (& py -3 -c 'import secrets; print(secrets.token_urlsafe(32))').Trim()
  if ($LASTEXITCODE -ne 0 -or $secret.Length -lt 24) { throw 'Python 3 is required; unable to generate a secure API key.' }
  if (Test-Path $envPath) {
    $content = Get-Content -Raw $envPath
    if ($content -match '(?m)^\s*IZQAR_API_KEY\s*=.*$') {
      $content = [regex]::Replace($content, '(?m)^\s*IZQAR_API_KEY\s*=.*$', "IZQAR_API_KEY=$secret")
    } else {
      $content = "IZQAR_API_KEY=$secret`n$content"
    }
    [System.IO.File]::WriteAllText($envPath, $content, [System.Text.UTF8Encoding]::new($false))
    Write-Host 'Repaired empty or template IZQAR_API_KEY in .env; existing non-key settings were retained.' -ForegroundColor Yellow
  } else {
    $envText = "IZQAR_API_KEY=$secret`nDATABASE_URL=sqlite:///./izqar.db`nIZQAR_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`n"
    [System.IO.File]::WriteAllText($envPath, $envText, [System.Text.UTF8Encoding]::new($false))
    Write-Host 'Created .env with a new local API key.' -ForegroundColor Green
  }
}
$backend = Join-Path $project 'backend'
Set-Location $backend
if (-not (Test-Path '.venv')) { & py -3 -m venv .venv }
$python = Join-Path $backend '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Virtual environment creation failed: $python was not found." }
& $python -m pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
$portCheck = [System.Net.Sockets.TcpClient]::new()
try {
  $portCheck.Connect('127.0.0.1', 8000)
  throw 'Port 8000 is already in use. Stop the existing service before starting IZQAR.'
} catch [System.Net.Sockets.SocketException] {
  # Connection refused means the port is free.
} finally {
  $portCheck.Dispose()
}
Write-Host 'Starting IZQAR API and waiting for health check...' -ForegroundColor Cyan
$server = Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $backend -PassThru -WindowStyle Hidden
try {
  $ready = $false
  foreach ($attempt in 1..30) {
    Start-Sleep -Milliseconds 500
    try {
      $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
      if ($health.status -eq 'ok') { $ready = $true; break }
    } catch { }
    if ($server.HasExited) { throw "API process exited with code $($server.ExitCode). Check the terminal output above." }
  }
  if (-not $ready) { throw 'API did not become ready within 15 seconds; browser was not opened.' }
  Write-Host 'IZQAR demo ready at http://127.0.0.1:8000/demo/' -ForegroundColor Green
  Write-Host 'API key: open .env in the project root. Press Ctrl+C to stop the API process.'
  Start-Process 'http://127.0.0.1:8000/demo/'
  Wait-Process -Id $server.Id
} finally {
  if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
