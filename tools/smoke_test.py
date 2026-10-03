#!/usr/bin/env python3
"""Read-only checks of the independent PyTorch ME runtime and tool routes."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DFL_ROOT = ROOT / '_internal' / 'DeepFaceLab'
LEGACY_DIR = ROOT / 'legacy-cli'
failures = []

def passed(message):
    print('[PASS] ' + message)

def failed(message):
    failures.append(message)
    print('[FAIL] ' + message)

def run_check(label, arguments, expected=()):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=120)
    output = result.stdout + result.stderr
    if result.returncode == 0 and all(token in output for token in expected):
        passed(label)
    else:
        failed(f'{label}: exit {result.returncode}; {output.strip()}')

def check_required_paths():
    required = [
        '.venv/Scripts/python.exe', '_internal/python_base/python.exe',
        '_internal/setenv.bat', '_internal/DeepFaceLab/main.py',
        '_internal/DeepFaceLab/me.py', '_internal/DeepFaceLab/me_backend/engine.py',
        '_internal/DeepFaceLab/models/Model_XSeg/Model_pytorch.py',
        '_internal/ffmpeg/ffmpeg.exe', '_internal/ffmpeg/ffprobe.exe',
        '启动 WebUI.bat', '传统命令菜单.bat', 'legacy-cli/menu.ps1',
        'legacy-cli/commands.json', 'launcher/setup-runtime.ps1',
        'webui/python/dfl_asset_tool.py', 'requirements.txt',
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    if missing:
        failed('Required files missing: ' + ', '.join(missing))
    else:
        passed('Independent Python, ME, XSeg, FFmpeg and WebUI entry points exist')
    families = {directory.name for directory in (DFL_ROOT / 'models').glob('Model_*')
                if directory.is_dir()}
    if families == {'Model_ME', 'Model_XSeg'}:
        passed('Available model families are ME and the XSeg mask helper')
    else:
        failed('Unexpected model families: ' + ', '.join(sorted(families)))

def check_python_sources():
    files = sorted(DFL_ROOT.rglob('*.py')) + sorted((ROOT / 'webui' / 'python').glob('*.py'))
    errors = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            for node in ast.walk(tree):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                if any(name.split('.')[0] == 'tensorflow' for name in names):
                    errors.append(f'{path.relative_to(ROOT)}: TensorFlow import')
        except Exception as error:
            errors.append(f'{path.relative_to(ROOT)}: {error}')
    if errors:
        failed('Python source validation: ' + '; '.join(errors))
    else:
        passed(f'{len(files)} Python sources parse without TensorFlow imports')

def check_runtime():
    import torch
    expected_python = ROOT / '.venv' / 'Scripts' / 'python.exe'
    if (Path(sys.executable).resolve() != expected_python.resolve()
            or Path(sys.base_prefix).resolve() != (ROOT / '_internal' / 'python_base').resolve()
            or not Path(torch.__file__).resolve().is_relative_to(ROOT / '.venv')):
        failed("Python and PyTorch must come from this repository's independent runtime")
    elif sys.version_info[:2] != (3, 12) or torch.__version__ != '2.9.1+cu128' or torch.version.cuda != '12.8':
        failed(f'Unexpected runtime: Python {sys.version.split()[0]}, PyTorch {torch.__version__}')
    else:
        passed(f'Local Python {sys.version.split()[0]} and PyTorch {torch.__version__} (CUDA {torch.version.cuda})')
    if importlib.util.find_spec('tensorflow') is not None:
        failed('TensorFlow is installed in the new runtime')
    else:
        passed('The new runtime has no TensorFlow package')
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([\w.-]+)==([^\s;]+)', line)
        if not match:
            failed('Unpinned runtime requirement: ' + line)
            continue
        name, version = match.groups()
        try:
            actual = importlib.metadata.version(name)
            if actual != version:
                failed(f'{name}: expected {version}, installed {actual}')
        except importlib.metadata.PackageNotFoundError:
            failed('Missing runtime requirement: ' + name)
    run_check('pip dependency health', [sys.executable, '-m', 'pip', 'check'])

def check_assets():
    hashes = {
        '_internal/DeepFaceLab/facelib/S3FD.npy': 'b4894ecfba8e6461eb1a69490f76184620c816605238ff0ba3216f1143a06c29',
        '_internal/DeepFaceLab/facelib/2DFAN.npy': 'ca2dc7f0b2aa146842e6de2119fedff1142188b7cff5ab702564952d6cba4624',
        '_internal/DeepFaceLab/facelib/3DFAN.npy': 'b50d2faf0fd4d6503aba9d19365e7aff06f2ea96bac37fc5f8f25a191a0a63a9',
        '_internal/DeepFaceLab/facelib/FaceEnhancer.npy': '254958f67c9adfe97a0c9fc7b3c343ba490a1519c01862a50945fa228875476a',
        '_internal/vision_models/face_recognition_sface_2021dec.onnx': '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
    }
    errors = []
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(relative)
            continue
        with path.open('rb') as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != expected:
                errors.append(relative)
    if errors:
        failed('Missing or modified helper weights: ' + ', '.join(errors))
    else:
        passed('Detector, landmark, face enhancer and SFace weights match pinned hashes')

def check_tool_routes():
    commands = json.loads((LEGACY_DIR / 'commands.json').read_text(encoding='utf-8-sig'))
    errors = []
    files = [command['file'] for command in commands]
    for name in files:
        path = LEGACY_DIR / name
        if Path(name).name != name or not path.is_file():
            errors.append('Missing registered tool: ' + name)
            continue
        text = path.read_text(encoding='utf-8-sig')
        if re.search(r'python_common|DeepFaceLab_old|setenv_old|--model\s+(?:SAEHD|Q\d+|AMP|Quick96)', text, re.I):
            errors.append('Legacy runtime or model reference: ' + name)
    if len(files) != len(set(files)) or {int(item['category']) for item in commands} != set(range(1, 9)):
        errors.append('The fixed routes must be unique and cover all eight categories')
    training = [name for name in files if name.endswith('train ME.bat')]
    if len(training) != 1:
        errors.append('Expected one ME training route')
    else:
        text = (LEGACY_DIR / training[0]).read_text(encoding='utf-8-sig')
        for option in ('--training-data-src-dir "%WORKSPACE%\\data_src\\aligned"',
                       '--training-data-dst-dir "%WORKSPACE%\\data_dst\\aligned"', '--model ME'):
            if option not in text:
                errors.append('ME training route is missing ' + option)
    if errors:
        failed('Tool routing: ' + '; '.join(errors))
    else:
        passed(f'{len(commands)} fixed tool routes cover eight categories and use ME training paths')
    run_check('Native tool menu self-check', [os.environ.get('COMSPEC', 'cmd.exe'),
              '/d', '/c', str(ROOT / '传统命令菜单.bat'), '--check'],
              expected=('[OK]', f'{len(commands)} fixed tool routes', '8 categories'))
    run_check('Main CLI exposes ME and XSeg', [sys.executable, str(DFL_ROOT / 'main.py'), 'train', '--help'],
              expected=('{ME,XSeg}',))
    run_check('Qt editor binding self-check', [sys.executable, str(DFL_ROOT / 'main.py'), 'qt', 'selftest'],
              expected=('PySide6',))
    run_check('Packaged FFmpeg starts', [str(ROOT / '_internal' / 'ffmpeg' / 'ffmpeg.exe'), '-version'],
              expected=('ffmpeg version',))

def check_web_cli_flags():
    registry = (ROOT / 'webui' / 'server' / 'command-registry.mjs').read_text(encoding='utf-8-sig')
    used = set(re.findall(r'''["'](--[a-z][a-z-]+)["']''', registry))
    accepted = set()
    tree = ast.parse((DFL_ROOT / 'main.py').read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            accepted.update(argument.value for argument in node.args
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                and argument.value.startswith('--'))
    for command in ('web-train', 'import-tf', 'list-backups', 'restore-backup'):
        help_result = subprocess.run([sys.executable, str(DFL_ROOT / 'me.py'), command, '--help'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        if help_result.returncode:
            failed(f'ME {command} CLI help failed: ' + help_result.stderr.strip())
            return
        accepted.update(re.findall(r'--[a-z][a-z-]+', help_result.stdout))
    missing = sorted(used - accepted)
    if missing:
        failed('Web command registry uses unsupported CLI flags: ' + ', '.join(missing))
    else:
        passed(f'All {len(used)} Web command registry flags exist in the Python CLI')

def main():
    print(f'DFL-PT-WEBUI smoke test\nRepository: {ROOT}\nPython: {sys.executable}\n')
    for check in (check_required_paths, check_python_sources, check_runtime, check_assets, check_tool_routes, check_web_cli_flags):
        try:
            check()
        except Exception as error:
            failed(f'{check.__name__}: {type(error).__name__}: {error}')
    print(f"\nRESULT: {'FAIL' if failures else 'PASS'} ({len(failures)} failed checks)")
    return int(bool(failures))

if __name__ == '__main__':
    sys.exit(main())
