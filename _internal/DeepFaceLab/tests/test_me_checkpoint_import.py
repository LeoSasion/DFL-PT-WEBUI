"""Artificial TF containers prove strict tensor migration without TensorFlow."""
import hashlib
import itertools
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from me_backend.checkpoint_import import (
    LegacyMEImportError, _named_parameters, _prepare_component, import_tf_me_weights, legacy_me_components)
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from core.leras import nn

torch.set_num_threads(2)


def small(archi='liae-ud', **kwargs):
    return MEConfig(archi=archi, resolution=64, ae_dims=32, e_dims=16,
                    d_dims=16, d_mask_dims=16, batch_size=1, **kwargs)


def write_mapping(path, mapping, counted=False):
    with path.open('wb') as stream:
        if counted:
            pickle.dump(len(mapping), stream, protocol=4)
            for name, tensor in mapping.items():
                pickle.dump({name:tensor}, stream, protocol=4)
        else:
            pickle.dump(mapping, stream, protocol=4)


def tf_fixture(directory, config, counted=False, prefix='fixture_ME'):
    directory.mkdir(parents=True, exist_ok=True)
    expected = MEEngine(config, 'cpu', seed=67)
    mappings = {}
    for component in legacy_me_components(config.archi):
        mapping = {}
        for name, layer, parameter_name, parameter in _named_parameters(getattr(expected.network, component)):
            array = parameter.detach().numpy().copy()
            if parameter_name == 'weight' and layer.__class__.__name__ == 'Conv2D':
                array = array.transpose(2, 3, 1, 0).copy()  # Independent inverse OIHW -> HWIO fixture.
            mapping[name+':0'] = array
        path = directory / f'{prefix}_{component}.npy'
        write_mapping(path, mapping, counted)
        mappings[component] = mapping
    return expected, mappings


@pytest.mark.parametrize('counted', [True, False])
def test_tf_containers_import_every_tensor_and_reset_optimizer(tmp_path, counted):
    config = small(optimizer_on_cpu=True, lr_dropout='cpu')
    expected, _ = tf_fixture(tmp_path/'legacy', config, counted)
    # These legacy files must not be decoded or mistaken for optimizer resume.
    (tmp_path/'legacy'/'fixture_ME_src_dst_opt.npy').write_bytes(b'not imported')
    output = tmp_path/'native'/'fixture.pt'
    torch.manual_seed(981)
    rng = torch.get_rng_state()
    report = import_tf_me_weights(tmp_path/'legacy', 'fixture_ME', config, output)
    assert torch.equal(rng, torch.get_rng_state())
    assert report['network_only'] and report['optimizer_reset'] and report['iteration_reset']
    assert not report['legacy_optimizer_imported']
    payload = torch.load(output, map_location='cpu', weights_only=True)
    assert payload['import_metadata'] == {key:value for key,value in report.items() if key != 'path'}
    assert payload['iteration'] == 0 and payload['optimizer']['state'] == {}
    loaded = MEEngine.load(output, 'cpu')
    for name, tensor in expected.network.state_dict().items():
        torch.testing.assert_close(loaded.network.state_dict()[name], tensor, rtol=0, atol=0)
    image = np.linspace(0., 1., 3*64*64, dtype=np.float32).reshape(1,3,64,64)
    for first, second in zip(expected.predict(image), loaded.predict(image)):
        np.testing.assert_array_equal(first, second)
    for component in report['components']:
        original = tmp_path/'legacy'/component['file']
        assert component['sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
        assert component['container'] == ('counted-tensor-records' if counted else 'tensor-dictionary')
    assert 'tensorflow' not in sys.modules


VARIANTS = [family + ('-'+''.join(options) if options else '')
            for family in ('df','liae') for count in range(5)
            for options in itertools.combinations('udtc', count)]


@pytest.mark.parametrize('archi', VARIANTS)
def test_explicit_df_liae_architecture_layouts_import_on_cpu(tmp_path, archi):
    config = small(archi)
    expected, _ = tf_fixture(tmp_path/'legacy', config)
    output = tmp_path/'converted.pt'
    report = import_tf_me_weights(tmp_path/'legacy', 'fixture_ME', config.to_dict(), output)
    assert report['source_architecture'] == archi
    loaded = MEEngine.load(output, 'cpu')
    assert tuple(component['component'] for component in report['components']) == legacy_me_components(archi)
    for name, tensor in expected.network.state_dict().items():
        torch.testing.assert_close(loaded.network.state_dict()[name], tensor, atol=0, rtol=0)


@pytest.mark.parametrize('invalid', ['flat-shape', 'nonfinite', 'integer', 'missing', 'unexpected', 'duplicate-normalized'])
def test_invalid_tensor_mapping_leaves_existing_output_untouched(tmp_path, invalid):
    config = small()
    _, mappings = tf_fixture(tmp_path/'legacy', config)
    mapping = mappings['encoder']
    key = next(iter(mapping))
    if invalid == 'flat-shape':
        mapping[key] = mapping[key].reshape(-1)  # Same element count is deliberately insufficient.
    elif invalid == 'nonfinite':
        mapping[key].flat[0] = np.nan
    elif invalid == 'integer':
        mapping[key] = mapping[key].astype(np.int32)
    elif invalid == 'missing':
        mapping.pop(key)
    elif invalid == 'unexpected':
        mapping['other/weight:0'] = mapping.pop(key)
    else:
        second = next(name for name in mapping if name != key)
        mapping['encoder/'+key] = mapping.pop(second)
    write_mapping(tmp_path/'legacy'/'fixture_ME_encoder.npy', mapping)
    output = tmp_path/'existing.pt'
    output.write_bytes(b'previous checkpoint remains intact')
    with pytest.raises(LegacyMEImportError, match='shape|nonfinite|float16|count|names|duplicate'):
        import_tf_me_weights(tmp_path/'legacy', 'fixture_ME', config, output)
    assert output.read_bytes() == b'previous checkpoint remains intact'
    assert not list(tmp_path.glob('existing.pt.*.tmp'))


@pytest.mark.parametrize('invalid', ['wrong-count', 'duplicate-record', 'multi-record', 'truncated', 'trailing'])
def test_malformed_streams_are_rejected(tmp_path, invalid):
    config = small()
    _, mappings = tf_fixture(tmp_path/'legacy', config, counted=True)
    mapping = mappings['encoder']
    path = tmp_path/'legacy'/'fixture_ME_encoder.npy'
    if invalid == 'truncated':
        path.write_bytes(path.read_bytes()[:-50])
    elif invalid == 'trailing':
        with path.open('ab') as stream:
            stream.write(b'extra')
    else:
        records = list(mapping.items())
        with path.open('wb') as stream:
            pickle.dump(len(records)+(1 if invalid == 'wrong-count' else 0), stream)
            for index, (key, value) in enumerate(records):
                if invalid == 'duplicate-record' and index == 1:
                    key, value = records[0]
                record = {key:value}
                if invalid == 'multi-record' and index == 0:
                    record['extra'] = value
                pickle.dump(record, stream, protocol=4)
    output = tmp_path/'missing-parent'/'rejected.pt'
    with pytest.raises(LegacyMEImportError):
        import_tf_me_weights(tmp_path/'legacy', 'fixture_ME', config, output)
    assert not output.exists() and not output.parent.exists()


def test_pickle_code_cannot_execute(tmp_path):
    config = small()
    tf_fixture(tmp_path/'legacy', config)
    marker = tmp_path/'unsafe-executed'

    class Unsafe:
        def __reduce__(self):
            return eval, (f"open({str(marker)!r}, 'w').write('executed')",)

    path = tmp_path/'legacy'/'fixture_ME_encoder.npy'
    path.write_bytes(pickle.dumps(Unsafe()))
    with pytest.raises(LegacyMEImportError, match='Unsupported pickle global'):
        import_tf_me_weights(tmp_path/'legacy', 'fixture_ME', config, tmp_path/'bad.pt')
    assert not marker.exists()


@pytest.mark.parametrize('archi', ['quick', 'rg', 'liae-rg', 'df-uud', 'df-z', 'df-', 'df-ud-x'])
def test_unsupported_old_architectures_rejected(archi):
    with pytest.raises(LegacyMEImportError, match='Unsupported legacy ME architecture'):
        legacy_me_components(archi)


def test_missing_files_and_ambiguous_names_rejected(tmp_path):
    config = small()
    (tmp_path/'legacy').mkdir()
    with pytest.raises(LegacyMEImportError, match='Missing TF ME component'):
        import_tf_me_weights(tmp_path/'legacy', 'fixture_ME', config, tmp_path/'bad.pt')
    with pytest.raises(LegacyMEImportError, match='exact legacy saved-model'):
        import_tf_me_weights(tmp_path/'legacy', '../fixture_ME', config, tmp_path/'bad.pt')


def test_handwritten_tf_conv_indices_and_dense_orientation():
    nn.initialize(nn.DeviceConfig.CPU(), data_format='NCHW')
    conv = nn.Conv2D(2, 5, kernel_size=3, name='conv')
    dense = nn.Dense(7, 4, name='dense')
    parameters = [conv.weight, conv.bias, dense.weight, dense.bias]
    model = SimpleNamespace(name='manual', layers=[conv,dense], build=lambda: None,
                            get_weights=lambda: parameters)
    # These fixed legacy keys, shapes and values do not come from Torch's
    # state_dict or the importer's traversal/conversion implementation.
    hwio = np.arange(3*3*2*5, dtype=np.float32).reshape(3,3,2,5)
    matrix = np.arange(7*4, dtype=np.float32).reshape(7,4)
    mapping = {'conv/weight:0':hwio, 'conv/bias:0':np.arange(5,dtype=np.float16),
               'dense/weight:0':matrix, 'dense/bias:0':np.arange(4,dtype=np.float32)}
    prepared = {id(parameter):tensor for parameter,tensor in _prepare_component(model,mapping,'manual.npy')}
    converted = prepared[id(conv.weight)]
    assert converted.shape == (5,2,3,3)
    for y,x,i,o in itertools.product(range(3),range(3),range(2),range(5)):
        assert converted[o,i,y,x].item() == hwio[y,x,i,o]
    np.testing.assert_array_equal(prepared[id(dense.weight)].numpy(), matrix)
    assert prepared[id(conv.bias)].dtype == torch.float32


def test_legacy_discriminator_import_is_explicitly_rejected(tmp_path):
    with pytest.raises(LegacyMEImportError, match='GAN/true-face powers disabled'):
        import_tf_me_weights(tmp_path, 'fixture_ME', small(gan_power=.1), tmp_path/'bad.pt')
