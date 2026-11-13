# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
param([switch]$ConfigureOnly)
$ErrorActionPreference = 'Stop'
$project = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $project
$envPath = Join-Path $project '.env'
$generateKey = -not (Test-Path $envPath)
$keyPattern = '(?m)^\s*IZQAR_API_KEY\s*=\s*(.*?)\s*$'
if (Test-Path $envPath) {
  $content = Get-Content -LiteralPath $envPath -Raw
  $keyEntries = [regex]::Matches($content, $keyPattern)
  if ($keyEntries.Count -gt 1) { throw '.env contains multiple IZQAR_API_KEY entries; resolve them manually.' }
  if ($keyEntries.Count -eq 0) {
    $generateKey = $true
  } else {
    $currentKey = $keyEntries[0].Groups[1].Value.Trim().Trim('"').Trim("'")
    $template = $currentKey -eq '' -or $currentKey -match '^(CHANGE_ME|replace-me|your[-_]?api[-_]?key|example)'
    if ($template) {
      $generateKey = $true
    } elseif ($currentKey.Length -lt 24) {
      throw 'Existing IZQAR_API_KEY is shorter than 24 characters. It was not overwritten; replace it manually with a secure key.'
    }
  }
}
if ($generateKey) {
  $bytes = [byte[]]::new(32)
  $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  try { $random.GetBytes($bytes) } finally { $random.Dispose() }
  $secret = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
  if (Test-Path $envPath) {
    if ($keyEntries.Count -eq 1) {
      $content = [regex]::Replace($content, $keyPattern, "IZQAR_API_KEY=$secret")
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
if ($ConfigureOnly) { Write-Host 'Local demo API key configuration is valid.'; return }
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
