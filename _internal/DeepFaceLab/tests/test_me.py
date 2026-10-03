import copy
import numpy as np
import pytest
import torch
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.data import AlignedDataset, read_aligned
from me_backend.losses import priority_mask, ms_ssim_loss
from me_backend.optimizer import MEOptimizer
from tests.me_fixtures import make_aligned

torch.set_num_threads(2)


def small(**kwargs):
    return MEConfig(resolution=64,ae_dims=32,e_dims=16,d_dims=16,d_mask_dims=16,batch_size=1,**kwargs)


def test_aligned_masks_and_rng_resume(tmp_path):
    config = small(eyes_prio=True,mouth_prio=True)
    folder = make_aligned(tmp_path/'src')
    image,mask,encoded = read_aligned(folder/'000.jpg',config)
    assert image.shape == (64,64,3) and mask.shape == (64,64,1)
    assert encoded.max()>2 and np.any((encoded>1)&(encoded<=2))
    dataset = AlignedDataset(folder,config,True,123)
    dataset.batch()
    state = copy.deepcopy(dataset.state_dict())
    expected = dataset.batch()
    resumed = AlignedDataset(folder,config,True,0)
    resumed.load_state_dict(state)
    for a,b in zip(expected,resumed.batch()):
        np.testing.assert_array_equal(a,b)
    state['fingerprint']='wrong'
    with pytest.raises(ValueError,match='changed'):
        resumed.load_state_dict(state)


def test_eye_mouth_are_separate():
    encoded = torch.tensor([0.,1.,1.5,2.,2.5,3.])
    eyes = priority_mask(encoded,True,False)
    mouth = priority_mask(encoded,False,True)
    torch.testing.assert_close(eyes,torch.tensor([0.,0.,.5,1.,.5,0.]))
    torch.testing.assert_close(mouth,torch.tensor([0.,0.,0.,0.,.5,1.]))
    torch.testing.assert_close(eyes+mouth,priority_mask(encoded,True,True))


def test_dropout_resamples():
    p = torch.nn.Parameter(torch.zeros(10000))
    opt = MEOptimizer([p],small(lr_dropout='y'))
    p.grad = torch.ones_like(p)
    opt.step()
    changed1 = p.detach().clone()!=0
    before = p.detach().clone()
    opt.step()
    changed2 = p!=before
    assert .25<float(changed1.float().mean())<.35
    assert torch.any(changed1 & ~changed2) and torch.any(changed2 & ~changed1)


@pytest.mark.parametrize('kind',['SSIM','MS-SSIM','MS-SSIM+L1'])
def test_train_options_finite(kind,tmp_path):
    config = small(loss_function=kind,eyes_prio=True,mouth_prio=True,blur_out_mask=True,
                   background_power=.2,face_style_power=.1,bg_style_power=.1)
    dataset = AlignedDataset(make_aligned(tmp_path/'src'),config,True,4)
    engine = MEEngine(config,'cpu',3)
    batch = dataset.batch()
    before = next(engine.network.encoder.parameters()).detach().clone()
    result = engine.train_step(batch,batch)
    assert np.isfinite(result['src_loss']) and np.isfinite(result['dst_loss'])
    assert not torch.equal(before,next(engine.network.encoder.parameters()))


def test_exact_checkpoint_resume_and_strict_load(tmp_path):
    config = small(lr_dropout='y')
    dataset = AlignedDataset(make_aligned(tmp_path/'src'),config,True,3)
    engine = MEEngine(config,'cpu',2)
    batch = dataset.batch()
    engine.train_step(batch,batch)
    path = tmp_path/'model.pt'
    engine.save(path,dict(src=dataset.state_dict()))
    prediction = engine.predict(batch[1])
    expected = engine.train_step(batch,batch)
    expected_weights = {k:v.clone() for k,v in engine.network.state_dict().items()}
    resumed = MEEngine.load(path,'cpu')
    for a,b in zip(prediction,resumed.predict(batch[1])):
        np.testing.assert_array_equal(a,b)
    assert expected == resumed.train_step(batch,batch)
    for k,v in resumed.network.state_dict().items():
        torch.testing.assert_close(v,expected_weights[k],rtol=0,atol=0)
    payload = torch.load(path,weights_only=True)
    payload['model'].pop(next(iter(payload['model'])))
    torch.save(payload,tmp_path/'bad.pt')
    with pytest.raises(RuntimeError,match='Missing key'):
        MEEngine.load(tmp_path/'bad.pt','cpu')
    payload = torch.load(path,weights_only=True)
    payload['optimizer']['state']={}
    torch.save(payload,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='Incomplete optimizer'):
        MEEngine.load(tmp_path/'bad.pt','cpu')


def test_no_warp_freezes_inter_ab(tmp_path):
    config = small(random_warp=False,adabelief=False)
    dataset = AlignedDataset(make_aligned(tmp_path/'src'),config,True,3)
    engine = MEEngine(config,'cpu')
    before = [p.detach().clone() for p in engine.network.inter_AB.parameters()]
    batch = dataset.batch()
    engine.train_step(batch,batch)
    for old,new in zip(before,engine.network.inter_AB.parameters()):
        torch.testing.assert_close(old,new,rtol=0,atol=0)


def test_msssim_anticorrelation_has_finite_gradients():
    x = torch.rand(1,3,64,64,requires_grad=True)
    y = 1-x.detach()
    loss = ms_ssim_loss(x,y).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_unknown_options_rejected():
    with pytest.raises(ValueError,match='Unsupported ME options'):
        MEConfig.from_dict({'unrecognized_training_option':.1})
