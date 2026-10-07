[CmdletBinding()]
param([string]$ProjectRoot = '', [switch]$InstallDependencies, [switch]$NoNetwork,
    [string]$WheelhousePath = '', [ValidateSet('production','restoration','validation','evaluation','scene')][string]$RuntimeProfile = 'production')
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
$basePython = Join-Path $root '_internal\python_base\python.exe'
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $root ('release\python-locks\' + $RuntimeProfile + '-win-cp312.txt')
if (-not (Test-Path -LiteralPath (Join-Path $root '_internal\DeepFaceLab\me.py'))) { throw "Not a DFL-PT-WEBUI project: $root" }
if (-not (Test-Path -LiteralPath $basePython)) { throw 'The independent _internal/python_base interpreter is missing.' }
if (-not (Test-Path -LiteralPath $python)) {
    if (-not (Test-Path -LiteralPath $basePython)) { throw 'Place the standalone Python 3.12 distribution in _internal/python_base first.' }
    & $basePython -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
    if ($LASTEXITCODE -ne 0) { throw 'The project-local base interpreter must be Python 3.12.' }
    & $basePython -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project-local .venv.' }
}
# Windows venv records an absolute base path. Rebase it to the packaged local
# interpreter after this project directory is copied or moved.
$configurationPath = Join-Path $venv 'pyvenv.cfg'
if (Test-Path -LiteralPath $configurationPath) {
    $configuration = [IO.File]::ReadAllText($configurationPath)
    $baseDirectory = Split-Path -Parent $basePython
    $configuration = [Regex]::Replace($configuration,'(?m)^home\s*=.*$',('home = ' + $baseDirectory))
    $configuration = [Regex]::Replace($configuration,'(?m)^executable\s*=.*$',('executable = ' + $basePython))
    $configuration = [Regex]::Replace($configuration,'(?m)^command\s*=.*$',('command = ' + $basePython + ' -m venv ' + $venv))
    [IO.File]::WriteAllText($configurationPath,$configuration,(New-Object Text.UTF8Encoding($false)))
}
if ($InstallDependencies) {
    if ($NoNetwork -and -not $WheelhousePath) { throw 'Offline dependency installation requires a verified wheelhouse.' }
    if (-not (Test-Path -LiteralPath $requirements)) { throw "Missing runtime requirements: $requirements" }
    # Production portable archives carry only the locked inference packages.
    # Explicit installation can restore pip from this local Python's bundled
    # wheel without networking or touching another interpreter/environment.
    & $python -I -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pip') is not None else 1)"
    if ($LASTEXITCODE -ne 0) {
        & $python -I -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw 'Could not restore pip from the project-local Python bundled wheel.' }
    }
    $pipOptions = @('--disable-pip-version-check','--require-hashes','--only-binary=:all:','-r',$requirements)
    if ($WheelhousePath) { $pipOptions += @('--no-index','--find-links',$WheelhousePath) }
    else { $pipOptions += @('--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu128') }
    & $python -I -m pip --isolated install @pipOptions
    if ($LASTEXITCODE -ne 0) { throw 'Installing the ME and tool requirements failed.' }
    & $python -I -m pip --isolated check
    if ($LASTEXITCODE -ne 0) { throw 'Locked runtime dependency graph is inconsistent.' }
}
& $python -I -c "import sys,torch,cv2,numpy,scipy,pathlib; root=pathlib.Path(sys.executable).resolve().parents[2]; assert pathlib.Path(sys.base_prefix).resolve() == root/'_internal'/'python_base', sys.base_prefix; assert pathlib.Path(torch.__file__).resolve().is_relative_to(root/'.venv'); assert sys.version_info[:2] == (3,12), sys.version; assert torch.__version__ == '2.9.1+cu128', torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda; print('DFL_PT_RUNTIME_OK|'+sys.version.split()[0]+'|'+torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'The project-local Python / PyTorch runtime validation failed.' }
