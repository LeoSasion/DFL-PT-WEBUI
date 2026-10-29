"""Native BF16 training gates and caller-controlled CUDA kernel caches."""
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))

from core.leras import device as legacy_device
from me_backend.engine import MEEngine, resolve_devices
from tae.engine import TAEEngine


@pytest.mark.parametrize('native', [False, True])
def test_tae_bf16_requires_native_support(native):
    engine = object.__new__(TAEEngine)
    engine.device = torch.device('cuda:0')
    # Old GPUs may accept a BF16 tensor through emulation without native math.
    def supported(*, including_emulation=True):
        return native or including_emulation
    with mock.patch('torch.cuda.device', return_value=nullcontext()), \
            mock.patch('torch.cuda.is_bf16_supported', side_effect=supported) as probe:
        if native:
            engine._validate_precision(SimpleNamespace(precision='bf16'))
        else:
            with pytest.raises(ValueError, match='select FP32'):
                engine._validate_precision(SimpleNamespace(precision='bf16'))
        probe.assert_called_once_with(including_emulation=False)


@pytest.mark.parametrize('native', [False, True])
def test_me_bf16_rejects_emulation_before_training_update(native):
    engine = object.__new__(MEEngine)
    engine.device, engine.device_ids = torch.device('cuda:0'), [0]
    engine.config = SimpleNamespace(use_fp16=False, use_bf16=True)
    engine._training_precision_checked = False
    engine.network = mock.Mock()
    engine.adversarial = mock.Mock()
    class ReachedUpdate(Exception):
        pass
    engine._update = mock.Mock(side_effect=ReachedUpdate)
    def supported(*, including_emulation=True):
        return native or including_emulation
    with mock.patch('torch.cuda.device', return_value=nullcontext()), \
            mock.patch('torch.cuda.is_bf16_supported', side_effect=supported) as probe:
        if native:
            with pytest.raises(ReachedUpdate):
                engine.train_step(None, None)
            engine._update.assert_called_once()
            assert engine._training_precision_checked
        else:
            with pytest.raises(ValueError, match='select FP32'):
                engine.train_step(None, None)
            engine._update.assert_not_called()
            engine.network.train.assert_not_called()
            assert not engine._training_precision_checked
        probe.assert_called_once_with(including_emulation=False)


def test_fp32_does_not_require_bf16_capability():
    engine = object.__new__(TAEEngine)
    engine.device = torch.device('cuda:0')
    with mock.patch('torch.cuda.is_bf16_supported', side_effect=AssertionError('unexpected precision probe')):
        engine._validate_precision(SimpleNamespace(precision='fp32'))


@pytest.mark.parametrize('capability,arches,cuda_version,allowed', [
    ((6, 1), ['sm_70', 'sm_75', 'sm_80', 'sm_120'], '12.8', False),
    ((6, 1), ['sm_50', 'sm_60', 'sm_70', 'sm_90'], '12.6', True),
    ((12, 0), ['sm_50', 'sm_60', 'sm_70', 'sm_90'], '12.6', False),
    ((12, 0), ['sm_70', 'sm_75', 'sm_80', 'sm_120'], '12.8', True),
])
def test_device_selection_rejects_incompatible_wheel_before_allocating(capability, arches, cuda_version, allowed):
    with mock.patch('torch.cuda.is_available', return_value=True), \
            mock.patch('torch.cuda.device_count', return_value=1), \
            mock.patch('torch.cuda.get_device_capability', return_value=capability), \
            mock.patch('torch.cuda.get_arch_list', return_value=arches), \
            mock.patch.object(torch._C, '_cuda_getArchFlags', return_value=' '.join(arches), create=True), \
            mock.patch('torch.cuda.get_device_name', return_value='synthetic GPU'), \
            mock.patch('torch.version.cuda', cuda_version), \
            mock.patch('torch.empty', side_effect=AssertionError('GPU tensor allocated before validation')):
        if allowed:
            assert resolve_devices('cuda:0') == (torch.device('cuda:0'), [0])
        else:
            with pytest.raises(RuntimeError):
                resolve_devices('cuda:0')


@pytest.mark.parametrize('configured', [False, True])
def test_windows_gpu_enumeration_uses_requested_cache_without_global_writes(tmp_path, configured):
    requested = tmp_path / 'requested-cache'
    appdata = tmp_path / 'appdata'
    env = {'CUDA_CACHE_PATH': str(requested)} if configured else {'APPDATA': str(appdata)}
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(legacy_device.sys, 'platform', 'win32'), \
            mock.patch.object(legacy_device.io, 'log_info'), \
            mock.patch('torch.cuda.is_available', return_value=False), \
            mock.patch('torch.backends.mps.is_available', return_value=False):
        assert legacy_device.Devices._enumerate_torch_devices_dict() == {}
        expected = requested if configured else appdata / 'NVIDIA/ComputeCache_ALL'
        assert os.environ['CUDA_CACHE_PATH'] == str(expected)
        assert expected.is_dir()
        if configured:
            assert not appdata.exists()


@pytest.mark.parametrize('existing', [None, '1048576'])
def test_device_initialization_sets_ascii_cache_limit_and_preserves_user_value(existing):
    import __main__
    env = {} if existing is None else {'CUDA_CACHE_MAXSIZE': existing}
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(__main__, '__file__', None, create=True), \
            mock.patch.object(legacy_device.Devices, '_enumerate_torch_devices_dict', return_value={}):
        legacy_device.Devices.initialize_main_env()
        assert os.environ['CUDA_CACHE_MAXSIZE'] == ('2147483647' if existing is None else existing)
        assert all(key.isascii() for key in os.environ)
        assert os.environ['NN_DEVICES_INITIALIZED'] == '1'
        assert os.environ['NN_DEVICES_COUNT'] == '0'
