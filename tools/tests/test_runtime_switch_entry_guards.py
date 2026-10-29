"""Normal entry points stop before loading Torch or touching caches during a switch."""
import builtins
import importlib.util
from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT=Path(__file__).resolve().parents[2]
DFL=ROOT/'_internal/DeepFaceLab'
sys.path.insert(0,str(DFL))
from core import gpu_runtime, torch_runtime_switch
from core.leras.device import Devices
from core.leras.nn import nn


class RuntimeBlocked(RuntimeError):
    pass


def test_runtime_validation_stops_before_native_import(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_runtime_switch,'assert_runtime_available',mock.Mock(side_effect=RuntimeBlocked('switching')))
    original=builtins.__import__
    def imports(name,*args,**kwargs):
        if name=='torch':
            pytest.fail('Torch was imported before checking the runtime transaction')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',imports)
    with pytest.raises(RuntimeBlocked):
        gpu_runtime.validate_runtime(tmp_path)


def test_device_enumeration_does_not_create_a_cache_when_runtime_is_blocked(tmp_path, monkeypatch):
    cache=tmp_path/'cache'
    monkeypatch.setenv('CUDA_CACHE_PATH',str(cache))
    monkeypatch.setattr(torch_runtime_switch,'assert_runtime_available',mock.Mock(side_effect=RuntimeBlocked('switching')))
    with pytest.raises(RuntimeBlocked):
        Devices._enumerate_torch_devices_dict()
    assert not cache.exists()


def test_neural_initialization_checks_transaction_before_device_selection(monkeypatch):
    monkeypatch.setattr(torch_runtime_switch,'assert_runtime_available',mock.Mock(side_effect=RuntimeBlocked('switching')))
    monkeypatch.setattr(nn.DeviceConfig,'BestGPU',mock.Mock(side_effect=AssertionError('selected a GPU before guard')))
    with pytest.raises(RuntimeBlocked):
        nn.initialize()


def test_tae_cli_cannot_read_a_checkpoint_during_a_switch(monkeypatch):
    spec=importlib.util.spec_from_file_location('tae_guard_cli',DFL/'tae.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    monkeypatch.setattr(torch_runtime_switch,'assert_runtime_available',mock.Mock(side_effect=RuntimeBlocked('switching')))
    with mock.patch('tae.checkpoint.read_checkpoint',side_effect=AssertionError('checkpoint opened before guard')) as load:
        with pytest.raises(RuntimeBlocked):
            cli.main(['inspect','synthetic-do-not-read.pt'])
        load.assert_not_called()


def test_me_cli_module_checks_transaction_before_importing_torch(monkeypatch):
    monkeypatch.setattr(torch_runtime_switch,'assert_runtime_available',mock.Mock(side_effect=RuntimeBlocked('switching')))
    original=builtins.__import__
    def imports(name,*args,**kwargs):
        if name=='torch':pytest.fail('ME loaded Torch while its runtime was blocked')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',imports)
    spec=importlib.util.spec_from_file_location('me_guard_cli',DFL/'me.py')
    cli=importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeBlocked):
        spec.loader.exec_module(cli)
