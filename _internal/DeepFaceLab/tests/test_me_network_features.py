"""CPU contracts for ME architecture, recomputation, and adversarial training."""
from itertools import combinations

import numpy as np
import pytest
import torch

from core.leras import nn
from me_backend.adversarial import AdversarialTraining, UNetPatchDiscriminator
from me_backend.config import MEConfig
from me_backend.export import DFMNetwork
from me_backend.losses import me_losses, training_mask, total_variation_mse
from me_backend.network import MENetwork

torch.set_num_threads(2)


def small(**kwargs):
    return MEConfig(resolution=64,ae_dims=32,e_dims=16,d_dims=16,d_mask_dims=16,
                    batch_size=1,**kwargs)


def batch(seed=23):
    rng = torch.Generator().manual_seed(seed)
    image = torch.rand((1,3,64,64),generator=rng)
    mask = torch.zeros(1,1,64,64)
    mask[:,:,8:56,12:52] = 1
    return image,image.clone(),mask,mask.clone()


class _LegacyLiae(torch.nn.Module):
    """The pre-feature generator topology, retained as a checkpoint fixture."""
    def __init__(self, config):
        super().__init__()
        nn.initialize(nn.DeviceConfig.CPU(),data_format='NCHW')
        arch = nn.DeepFakeArchi(config.resolution,opts='ud')
        self.encoder = arch.Encoder(3,config.e_dims,name='encoder')
        size = self.encoder.get_out_res(config.resolution)**2*self.encoder.get_out_ch()
        self.inter_AB = arch.Inter(size,config.ae_dims,config.ae_dims*2,name='inter_AB')
        self.inter_B = arch.Inter(size,config.ae_dims,config.ae_dims*2,name='inter_B')
        self.decoder = arch.Decoder(config.ae_dims*4,config.d_dims,config.d_mask_dims,name='decoder')
        for module in self.children():
            module.get_weights()

    def forward(self, source, destination):
        src_ab = self.inter_AB(self.encoder(source))
        dst_code = self.encoder(destination)
        dst_ab,dst_b = self.inter_AB(dst_code),self.inter_B(dst_code)
        src,src_mask = self.decoder(torch.cat((src_ab,src_ab),1))
        dst,dst_mask = self.decoder(torch.cat((dst_b,dst_ab),1))
        swap,swap_mask = self.decoder(torch.cat((dst_ab,dst_ab),1))
        return dict(src=src,src_mask=src_mask,dst=dst,dst_mask=dst_mask,
                    swap=swap,swap_mask=swap_mask)


def test_legacy_liae_checkpoint_keeps_keys_and_predictions(tmp_path):
    legacy = _LegacyLiae(small()).eval()
    path = tmp_path/'prototype-generator.pt'
    torch.save(legacy.state_dict(),path)
    current = MENetwork(small()).eval()
    saved = torch.load(path,weights_only=True)
    assert saved.keys() == current.state_dict().keys()
    current.load_state_dict(saved,strict=True)
    source,destination = batch(),batch(24)
    with torch.no_grad():
        expected = legacy(source[0],destination[0])
        actual = current(source[0],destination[0])
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)


OPTIONS = [''.join(chars) for length in range(5) for chars in combinations('udtc',length)]


@pytest.mark.parametrize('base',['df','liae'])
@pytest.mark.parametrize('options',OPTIONS)
def test_all_architecture_options_have_finite_full_resolution_outputs(base,options):
    archi = base+('-'+options if options else '')
    config = small(archi=archi)
    network = MENetwork(config).eval()
    source,destination = batch(),batch(24)
    with torch.no_grad():
        outputs = network(source[0],destination[0],face_style=True)
        prediction = network.predict(destination[0])
    for name in ('src','dst','swap','swap_no_code_grad'):
        assert outputs[name].shape == (1,3,64,64)
        assert torch.isfinite(outputs[name]).all()
    for name in ('src_mask','dst_mask','swap_mask'):
        assert outputs[name].shape == (1,1,64,64)
        assert torch.isfinite(outputs[name]).all()
    for actual,name in zip(prediction,('swap','swap_mask','dst_mask')):
        torch.testing.assert_close(actual,outputs[name])
    if base == 'df':
        assert outputs['src_code'].shape[1] == config.ae_dims
        assert outputs['dst_code'].shape == outputs['src_code'].shape


@pytest.mark.parametrize('archi',['df','df-udtc','liae-ud','liae-udtc'])
def test_rg_keeps_weights_values_gradients_and_recomputes(archi):
    direct = MENetwork(small(archi=archi))
    recomputed = MENetwork(small(archi=archi,use_rg=True))
    recomputed.load_state_dict(direct.state_dict(),strict=True)
    calls = [0]
    hook = recomputed.encoder.register_forward_pre_hook(
        lambda module,args:calls.__setitem__(0,calls[0]+1))
    source,destination = batch(),batch(24)
    expected = direct(source[0],destination[0],face_style=True)
    actual = recomputed(source[0],destination[0],face_style=True)
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
    assert calls[0] == 2
    sum(value.square().mean() for value in expected.values()).backward()
    sum(value.square().mean() for value in actual.values()).backward()
    assert calls[0] > 2
    hook.remove()
    for (name,left),(other,right) in zip(direct.named_parameters(),recomputed.named_parameters()):
        assert name == other and left.dtype == right.dtype == torch.float32
        torch.testing.assert_close(right.grad,left.grad,rtol=1e-5,atol=1e-6)


def test_rg_frozen_intermediate_still_updates_encoder():
    network = MENetwork(small(use_rg=True,random_warp=False))
    source,destination = batch(),batch(24)
    outputs = network(source[0],destination[0])
    (outputs['src'].mean()+outputs['dst'].mean()).backward()
    assert all(parameter.grad is None for parameter in network.inter_AB.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum()>0
               for parameter in network.encoder.parameters())


@pytest.mark.parametrize('archi,gan,true_face,masked,blur',[
    ('liae-ud',.1,0,True,False),
    ('df-ud',0,.05,True,False),
    ('df-udtc',.1,.05,False,True),
    ('df-ud',.1,.05,True,True),
])
def test_adversarial_parameters_and_generator_actually_update(archi,gan,true_face,masked,blur):
    config = small(archi=archi,gan_power=gan,gan_dims=4,true_face_power=true_face,
                   gan_noise=.2,masked_training=masked,blur_out_mask=blur,use_rg=True)
    network = MENetwork(config)
    adversarial = AdversarialTraining(config,network)
    assert adversarial.enabled
    assert not ({id(p) for p in network.parameters()} & {id(p) for p in adversarial.parameters()})
    source,destination = batch(),batch(24)
    outputs = network(source[0],destination[0])
    generator_before = {k:p.detach().clone() for k,p in network.named_parameters()}
    discriminator_before = {k:p.detach().clone() for k,p in adversarial.named_parameters()}
    generator_opt = torch.optim.Adam(network.parameters(),lr=1e-4)
    discriminator_opt = torch.optim.Adam(adversarial.parameters(),lr=1e-4)
    adversarial.requires_grad_(False)
    generator_loss = adversarial.generator_loss(outputs,source,destination)
    assert generator_loss.shape == (1,) and torch.isfinite(generator_loss).all()
    # The adversarial objective itself must reach generator weights, independently
    # of the reconstruction objective used for the full optimizer step below.
    encoder_weight = next(network.encoder.parameters())
    gradient = torch.autograd.grad(generator_loss.mean(),encoder_weight,retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    if gan:
        decoder = network.decoder_src if archi.startswith('df') else network.decoder
        gradient = torch.autograd.grad(generator_loss.mean(),next(decoder.parameters()),retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    src,dst = me_losses(config,source,destination,outputs)
    (src+dst+generator_loss).mean().backward()
    assert all(p.grad is None for p in adversarial.parameters())
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    generator_opt.step()
    generator_opt.zero_grad(set_to_none=True)
    adversarial.requires_grad_(True)
    discriminator_loss = adversarial.discriminator_loss(outputs,source,destination)
    assert discriminator_loss.ndim == 0 and torch.isfinite(discriminator_loss)
    discriminator_loss.backward()
    assert all(p.grad is None for p in network.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adversarial.parameters())
    discriminator_opt.step()
    assert any(not torch.equal(p,generator_before[k]) for k,p in network.named_parameters())
    for prefix in ('gan','true_face'):
        if getattr(adversarial,prefix) is not None:
            assert any(not torch.equal(p,discriminator_before[k])
                       for k,p in adversarial.named_parameters() if k.startswith(prefix+'.'))


def test_gan_patch_controls_real_architecture_and_has_two_heads():
    narrow = UNetPatchDiscriminator(3,base_ch=4)
    wide = UNetPatchDiscriminator(32,base_ch=4)
    assert narrow.receptive_field == 3 and abs(wide.receptive_field-32)<=1
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in narrow.parameters())
    image = torch.rand(2,3,63,65)
    center,dense = wide(image)
    assert center.shape[0:2] == (2,1) and dense.shape == (2,1,63,65)
    (center.mean()+dense.mean()).backward()
    assert all(p.grad is not None for p in wide.parameters())


def test_gan_uses_source_mask_and_ignores_destination_labels():
    config = small(gan_power=.1,gan_dims=4,gan_noise=0)
    network = MENetwork(config)
    adversarial = AdversarialTraining(config,network)
    source,destination = batch(),batch(24)
    outputs = network(source[0],destination[0])
    captured = []
    hook = adversarial.gan.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    loss = adversarial.discriminator_loss(outputs,source,destination)
    hook.remove()
    mask = training_mask(source[2],config.resolution)
    torch.testing.assert_close(captured[0],outputs['src'].detach()*mask)
    torch.testing.assert_close(captured[1],source[1]*mask)
    other = tuple(torch.zeros_like(value) for value in destination)
    torch.testing.assert_close(adversarial.discriminator_loss(outputs,source,other),loss)


def test_gan_total_variation_uses_original_per_image_sum():
    image = torch.tensor([[[[0.,1.],[2.,3.]]],[[[0.,0.],[0.,0.]]]])
    torch.testing.assert_close(total_variation_mse(image),torch.tensor([10.,0.]))


def test_adversarial_checkpoint_and_optimizer_resume_exactly(tmp_path):
    config = small(archi='df-ud',gan_power=.1,gan_dims=4,true_face_power=.1,gan_noise=.25)
    network = MENetwork(config)
    adversarial = AdversarialTraining(config,network)
    optimizer = torch.optim.Adam(adversarial.parameters(),lr=1e-4)
    source,destination = batch(),batch(24)
    with torch.no_grad():
        outputs = network(source[0],destination[0])
    def step(container,opt):
        opt.zero_grad(set_to_none=True)
        loss = container.discriminator_loss(outputs,source,destination)
        loss.backward()
        opt.step()
        return loss.detach()
    step(adversarial,optimizer)
    path = tmp_path/'adversarial.pt'
    torch.save(dict(network=adversarial.state_dict(),optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state()),path)
    state = torch.load(path,weights_only=True)
    expected = step(adversarial,optimizer)
    resumed = AdversarialTraining(config,network)
    resumed.load_state_dict(state['network'],strict=True)
    resumed_opt = torch.optim.Adam(resumed.parameters(),lr=1e-4)
    resumed_opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    torch.testing.assert_close(step(resumed,resumed_opt),expected,rtol=0,atol=0)
    for key,value in adversarial.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key],value,rtol=0,atol=0)


def test_true_face_rejects_liae_and_disabled_adversarial_is_parameter_free():
    with pytest.raises(ValueError,match='df'):
        small(true_face_power=.1)
    network = MENetwork(small())
    disabled = AdversarialTraining(small(),network)
    assert not disabled.enabled and list(disabled.parameters()) == []
    source,destination = batch(),batch(24)
    outputs = network(source[0],destination[0])
    assert torch.equal(disabled.generator_loss(outputs,source,destination),torch.zeros(1))


@pytest.mark.parametrize('archi',['df-ud','liae-ud'])
def test_face_style_branch_only_updates_decoder(archi):
    network = MENetwork(small(archi=archi,use_rg=True))
    source,destination = batch(),batch(24)
    network(source[0],destination[0],face_style=True)['swap_no_code_grad'].mean().backward()
    assert all(p.grad is None for p in network.encoder.parameters())
    if archi.startswith('df'):
        assert all(p.grad is None for p in network.inter.parameters())
        decoder = network.decoder_src
    else:
        assert all(p.grad is None for p in network.inter_AB.parameters())
        assert all(p.grad is None for p in network.inter_B.parameters())
        decoder = network.decoder
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in decoder.parameters())


def test_amp_config_keeps_fp32_generator_state_dict():
    ordinary = MENetwork(small())
    amp = MENetwork(small(use_fp16=True,use_rg=True))
    assert ordinary.state_dict().keys() == amp.state_dict().keys()
    amp.load_state_dict(ordinary.state_dict(),strict=True)
    assert all(parameter.dtype == torch.float32 for parameter in amp.parameters())
    source,destination = batch(),batch(24)
    # This CPU's oneDNN cannot backpropagate BF16; use native CPU kernels to
    # exercise autocast/checkpoint context restoration without requiring CUDA.
    with torch.backends.mkldnn.flags(enabled=False):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            outputs = amp(source[0],destination[0])
            loss = outputs['src'].float().square().mean()+outputs['dst'].float().square().mean()
        loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in amp.parameters() if p.grad is not None)


@pytest.mark.parametrize('archi',['df-udt','liae-udtc'])
def test_rg_variants_export_dfm_without_training_checkpoint_ops(archi,tmp_path):
    ort = pytest.importorskip('onnxruntime')
    network = MENetwork(small(archi=archi,use_rg=True)).eval()
    model = DFMNetwork(network)
    face = torch.rand(1,64,64,3)
    path = tmp_path/'feature-model.dfm'
    with torch.no_grad():
        expected = model(face)
    torch.onnx.export(model,face,str(path),input_names=['in_face:0'],
        output_names=['out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0'],
        dynamic_axes={name:{0:'batch'} for name in ('in_face:0','out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0')},
        opset_version=17,dynamo=False)
    session = ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    actual = session.run(None,{'in_face:0':face.numpy()})
    for left,right in zip(actual,expected):
        np.testing.assert_allclose(left,right.numpy(),rtol=2e-4,atol=2e-5)
    doubled = face.repeat(2,1,1,1)
    assert session.run(None,{'in_face:0':doubled.numpy()})[1].shape == (2,64,64,3)
