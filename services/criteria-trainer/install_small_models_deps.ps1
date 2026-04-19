$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$req = Join-Path $PSScriptRoot "requirements-small-models.txt"

if (-not (Test-Path $req)) {
  throw "requirements file not found: $req"
}

python -m pip install --upgrade pip
python -m pip install -r $req

Write-Host "[OK] Installed small-model dependencies from $req"
