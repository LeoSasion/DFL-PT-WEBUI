import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.web_bridge import load_config_options, load_training_engine
from tests.me_fixtures import make_aligned

ROOT = Path(__file__).resolve().parents[1]


def run_cli(arguments, expected=0):
    result = subprocess.run([sys.executable, str(ROOT / 'me.py'), *map(str, arguments)],
                            capture_output=True, text=True, encoding='utf-8', timeout=120,
                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def configuration_arguments(source, settings, value):
    return ['--config', settings] if source == 'file' else ['--config-json', json.dumps(value)]


@pytest.mark.parametrize('configuration_source', ['file', 'inline'])
def test_training_cli_resume_config_target_and_pretrained_initialization(tmp_path, configuration_source):
    src, dst = make_aligned(tmp_path / 'src'), make_aligned(tmp_path / 'dst', offset=15)
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1,
                      use_rg=True, data_workers=1, ct_mode='rct', random_noise=True)
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps(config.to_dict()))
    model = tmp_path / 'training'
    base = ['train', '--src', src, '--dst', dst, '--model', model, '--device', 'cpu', '--threads', '2']
    configured = configuration_arguments(configuration_source, settings, config.to_dict())
    first_run = run_cli([*base, *configured, '--steps', '3'])
    assert json.loads((model / 'metadata.json').read_text())['iteration'] == 3
    assert (model / 'preview.png').is_file()
    history = [json.loads(line) for line in (model / 'loss-history.jsonl').read_text().splitlines()]
    assert [row['iteration'] for row in history] == [1, 2, 3]
    assert all('src_loss' in row and 'dst_loss' in row for row in history)
    # subprocess text mode normalizes CR to LF; the display unit test checks CR itself.
    assert '[#000001]' in first_run.stdout
    assert '[#000003]' in first_run.stdout  # Final result is flushed before save.
    fine_tune = tmp_path / 'fine-tune.json'
    fine_tune.write_text(json.dumps(dict(lr=.0001, eyes_prio=True)))
    updates = configuration_arguments(configuration_source, fine_tune, dict(lr=.0001, eyes_prio=True))
    checkpoint_before = (model / 'me.pt').read_bytes()
    rejected = run_cli([*base, '--resume', *updates, '--steps', '1'], expected=1)
    assert 'Resume config differs' in rejected.stderr
    assert (model / 'me.pt').read_bytes() == checkpoint_before
    if configuration_source == 'inline':
        for options, code, message in [
            (['--config-json', '[]'], 1, 'ME config must be a JSON object'),
            (['--config-json', '{"unrecognized_training_option":true}'], 1, 'Unsupported ME options'),
            (['--config', settings, '--config-json', '{}'], 2, 'not allowed with argument'),
        ]:
            invalid = run_cli([*base, '--resume', *options, '--steps', '1'], expected=code)
            assert message in invalid.stderr
            assert (model / 'me.pt').read_bytes() == checkpoint_before
    run_cli([*base, '--resume', *updates, '--allow-config-change', '--steps', '100', '--target-iterations', '5'])
    loaded = MEEngine.load(model / 'me.pt', 'cpu')
    assert loaded.iteration == 5 and loaded.config.lr == .0001 and loaded.config.eyes_prio
    initialized = tmp_path / 'initialized'
    run_cli(['train', '--src', src, '--dst', dst, '--model', initialized, *configured,
             '--initialize-from', model, '--device', 'cpu', '--steps', '1', '--threads', '2'])
    assert MEEngine.load(initialized / 'me.pt', 'cpu').iteration == 1


def test_pretraining_data_debug_and_finetune_transition(tmp_path, monkeypatch):
    pretrain = make_aligned(tmp_path / 'pretraining')
    src, dst = make_aligned(tmp_path / 'src'), make_aligned(tmp_path / 'dst', offset=20)
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1, pretrain=True)
    settings = tmp_path / 'pretrain.json'
    settings.write_text(json.dumps(config.to_dict()))
    model = tmp_path / 'pretrained'
    args = ['train', '--src', src, '--dst', dst, '--model', model, '--config', settings, '--device', 'cpu', '--threads', '2']
    failed = run_cli([*args, '--steps', '1'], expected=1)
    assert 'Pretraining requires' in failed.stderr and not (model / 'me.pt').exists()
    run_cli([*args, '--pretraining-data-dir', pretrain, '--steps', '2'])
    run_cli(['train', '--src', src, '--dst', dst, '--model', model, '--device', 'cpu', '--resume',
             '--no-pretrain', '--allow-config-change', '--reset-data-state', '--steps', '1', '--threads', '2'])
    loaded = MEEngine.load(model / 'me.pt', 'cpu')
    assert loaded.iteration == 3 and not loaded.config.pretrain
    checkpoint_before = (model / 'me.pt').read_bytes()
    debug_preview = tmp_path / 'web-preview.png'
    monkeypatch.setenv('DFL_WEB_PREVIEW_FILE', str(debug_preview))
    run_cli(['train', '--src', src, '--dst', dst, '--model', model, '--device', 'cpu', '--resume',
             '--debug-samples', '--threads', '2'])
    assert (model / 'debug-src.png').is_file() and (model / 'debug-dst.png').is_file()
    assert debug_preview.is_file()
    assert (model / 'me.pt').read_bytes() == checkpoint_before
    assert MEEngine.load(model / 'me.pt', 'cpu').iteration == 3


@pytest.mark.parametrize('configuration_source', ['file', 'inline'])
def test_legacy_weight_import_cli_registers_infers_and_trains(tmp_path, configuration_source):
    from tests.test_me_checkpoint_import import tf_fixture
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1)
    source = tmp_path / 'legacy'
    expected, _ = tf_fixture(source, config, counted=True)
    settings = tmp_path / 'import.json'
    settings.write_text(json.dumps(config.to_dict()))
    model = tmp_path / 'converted'
    configured = configuration_arguments(configuration_source, settings, config.to_dict())
    result = run_cli(['import-tf', '--source', source, '--name', 'fixture_ME', *configured,
                      '--model', model, '--threads', '2'])
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(report['imported']['path']) == model / 'me.pt'
    metadata = json.loads((model / 'metadata.json').read_text())
    assert metadata['modelClass'] == 'ME' and metadata['iteration'] == 0
    assert metadata['importedFrom']['network_only'] and metadata['importedFrom']['optimizer_reset']
    loaded = MEEngine.load(model / 'me.pt', 'cpu')
    for name, tensor in expected.network.state_dict().items():
        torch.testing.assert_close(loaded.network.state_dict()[name], tensor, rtol=0, atol=0)
    src, dst = make_aligned(tmp_path / 'src'), make_aligned(tmp_path / 'dst', offset=10)
    run_cli(['infer', '--model', model, '--input', next(src.glob('*.jpg')), '--output', tmp_path / 'prediction',
             '--device', 'cpu', '--threads', '2'])
    assert (tmp_path / 'prediction' / 'aligned-composite.png').is_file()
    run_cli(['train', '--src', src, '--dst', dst, '--model', model, '--resume', '--steps', '1',
             '--device', 'cpu', '--threads', '2'])
    trained = MEEngine.load(model / 'me.pt', 'cpu')
    assert trained.iteration == trained.optimizer_updates == 1
    rejected = run_cli(['import-tf', '--source', source, '--name', 'fixture_ME', *configured,
                        '--model', model], expected=1)
    assert 'existing models are never overwritten' in rejected.stderr
    assert MEEngine.load(model / 'me.pt', 'cpu').iteration == 1


def test_main_training_forwards_inline_configuration(tmp_path):
    src, dst = make_aligned(tmp_path / 'src'), make_aligned(tmp_path / 'dst', offset=10)
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1,
                      use_rg=True, lr_dropout='cpu', optimizer_on_cpu=True)
    result = subprocess.run([sys.executable, str(ROOT / 'main.py'), 'train', '--model', 'ME',
        '--training-data-src-dir', str(src), '--training-data-dst-dir', str(dst),
        '--model-dir', str(tmp_path / 'models'), '--force-model-name', 'inline-main', '--cpu-only',
        '--config-json', json.dumps(config.to_dict()), '--steps', '1', '--threads', '2'],
        capture_output=True, text=True, encoding='utf-8', timeout=120,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = MEEngine.load(tmp_path / 'models' / 'inline-main' / 'me.pt', 'cpu')
    assert loaded.iteration == 1 and loaded.config == config
    assert (tmp_path / 'models' / 'inline-main' / 'preview.png').is_file()


def test_real_control_close_saves_then_resumes_and_predicts(tmp_path):
    src, dst = make_aligned(tmp_path / 'src'), make_aligned(tmp_path / 'dst', offset=10)
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1)
    model = tmp_path / 'controlled'
    control = tmp_path / 'control.jsonl'
    ack = tmp_path / 'control-ack.json'
    heartbeat = tmp_path / 'trainer-heartbeat.json'
    base = ['train', '--src', src, '--dst', dst, '--model', model, '--device', 'cpu',
            '--threads', '2', '--save-every', '100', '--preview-every', '100']
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
           'DFL_WEB_CONTROL_FILE': str(control), 'DFL_WEB_CONTROL_ACK_FILE': str(ack),
           'DFL_WEB_HEARTBEAT_FILE': str(heartbeat)}
    process = subprocess.Popen([sys.executable, str(ROOT / 'me.py'), *map(str, base),
                                '--config-json', json.dumps(config.to_dict())],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env)
    try:
        deadline = time.monotonic() + 60
        history = model / 'loss-history.jsonl'
        while (not history.is_file() or not history.read_text(encoding='utf-8').strip()):
            assert process.poll() is None, 'Isolated Trainer exited before the first iteration'
            assert time.monotonic() < deadline, 'Isolated Trainer did not start training'
            time.sleep(0.1)
        requested_at = '2026-10-03T00:00:00.000Z'
        control.write_text(json.dumps({'operation': 'close', 'requestedAt': requested_at}) + '\n',
                           encoding='utf-8')
        output, _ = process.communicate(timeout=60)
        assert process.returncode == 0, output.decode('utf-8', errors='replace')
        assert b'\r[#000001]' in output  # Actual piped bytes retain CR for the Web terminal.
        result = json.loads(ack.read_text(encoding='utf-8'))
        assert result['operation'] == 'close' and result['status'] == 'completed'
        assert result['requestedAt'] == requested_at
        assert Path(result['checkpoint']) == model / 'me.pt'
        assert result['iteration'] >= 1
        finished = json.loads(heartbeat.read_text(encoding='utf-8'))
        assert finished['phase'] == 'finished' and finished['iteration'] == result['iteration']
        lines = [json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()]
        assert len(lines) == result['iteration']
        run_cli([*base, '--resume', '--steps', '1'])
        assert MEEngine.load(model / 'me.pt', 'cpu').iteration == result['iteration'] + 1
        run_cli(['infer', '--model', model, '--input', next(src.glob('*.jpg')),
                 '--output', tmp_path / 'prediction', '--device', 'cpu', '--threads', '2'])
        assert (tmp_path / 'prediction' / 'aligned-composite.png').is_file()
    finally:
        if process.poll() is None:
            process.terminate()  # Only this test-owned isolated process.
            process.communicate(timeout=10)


def test_configuration_namespace_compatibility_and_conflicting_sources(tmp_path):
    assert load_config_options(SimpleNamespace(config=None)) == {}
    assert load_config_options(SimpleNamespace(config_json='{"lr":0.0001}')) == dict(lr=.0001)
    with pytest.raises(ValueError, match='Use only one'):
        load_config_options(SimpleNamespace(config=tmp_path / 'absent.json', config_json='{}'))
    with pytest.raises(ValueError, match='JSON object'):
        load_training_engine(SimpleNamespace(model=tmp_path / 'model', config_json='null'))
    assert not (tmp_path / 'model').exists()


def test_import_requires_one_configuration_source_and_rejects_nonobject(tmp_path):
    base = ['import-tf', '--source', tmp_path / 'legacy', '--name', 'fixture_ME', '--model', tmp_path / 'converted']
    missing = run_cli(base, expected=2)
    assert 'one of the arguments --config --config-json is required' in missing.stderr
    conflict = run_cli([*base, '--config', tmp_path / 'absent.json', '--config-json', '{}'], expected=2)
    assert 'not allowed with argument' in conflict.stderr
    invalid = run_cli([*base, '--config-json', '[]'], expected=1)
    assert 'ME config must be a JSON object' in invalid.stderr
    assert not (tmp_path / 'converted').exists()
    assert not list(tmp_path.glob('.converted-import-*'))
