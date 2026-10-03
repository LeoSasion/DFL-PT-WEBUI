[CmdletBinding()]
param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot), [switch]$InstallDependencies, [switch]$NoNetwork)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
$basePython = Join-Path $root '_internal\python_base\python.exe'
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $root 'requirements.txt'
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
    if ($NoNetwork) { throw 'NoNetwork prohibits downloading packages.' }
    if (-not (Test-Path -LiteralPath $requirements)) { throw "Missing runtime requirements: $requirements" }
    & $python -m pip install --disable-pip-version-check 'torch==2.9.1' --index-url 'https://download.pytorch.org/whl/cu128'
    if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned PyTorch CUDA 12.8 wheel failed.' }
    & $python -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Installing the ME and tool requirements failed.' }
}
& $python -I -c "import sys,torch,cv2,numpy,scipy,pathlib; root=pathlib.Path(sys.executable).resolve().parents[2]; assert pathlib.Path(sys.base_prefix).resolve() == root/'_internal'/'python_base', sys.base_prefix; assert pathlib.Path(torch.__file__).resolve().is_relative_to(root/'.venv'); assert sys.version_info[:2] == (3,12), sys.version; assert torch.__version__ == '2.9.1+cu128', torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda; print('DFL_PT_RUNTIME_OK|'+sys.version.split()[0]+'|'+torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'The project-local Python / PyTorch runtime validation failed.' }
