"""Compare ME PyTorch against independently executed TensorFlow fixtures."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import torch
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.losses import gaussian_blur,dssim,style_loss,ms_ssim_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference',required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(args.reference)
    ref = np.load(folder/'tf_outputs.npz')
    weights = np.load(folder/'tf_weights.npz')
    engine = MEEngine(MEConfig(resolution=64,ae_dims=32,e_dims=16,d_dims=16,d_mask_dims=16,batch_size=1),args.device)
    used = set()
    with torch.no_grad():
        for name,p in engine.network.named_parameters():
            key = re.sub(r'\.downs\.(\d+)\.',r'.downs_\1.',name).replace('.','/')+':0'
            value = weights[key]
            if p.ndim == 4:
                value = value.transpose(3,2,0,1)
            if value.shape != tuple(p.shape):
                raise AssertionError((name,value.shape,p.shape))
            p.copy_(torch.from_numpy(value).to(args.device))
            used.add(key)
    assert used == set(weights.files), 'Every TF and PyTorch parameter must match by exact name'
    x = torch.tensor(ref['input'].transpose(0,3,1,2),device=args.device,requires_grad=True)
    y = torch.tensor(ref['target'].transpose(0,3,1,2),device=args.device)
    n = engine.network
    enc = n.encoder(x)
    ab,b = n.inter_AB(enc),n.inter_B(enc)
    swap,srcm = n.decoder(torch.cat((ab,ab),1))
    dst,dstm = n.decoder(torch.cat((b,ab),1))
    ((swap-y).square().mean()+srcm.mean()).backward()
    actual = dict(encoder=enc,inter_AB=ab,inter_B=b,swap=swap,src_mask=srcm,dst=dst,dst_mask=dstm,
        input_gradient=x.grad,blur=gaussian_blur(x,2),dssim=dssim(x,y,6),style=style_loss(x,y,8),
        ms=ms_ssim_loss(x,y),ms_l1=ms_ssim_loss(x,y,True))
    errors = {}
    for name,tensor in actual.items():
        value = tensor.detach().cpu().numpy()
        if value.ndim==4:
            value = value.transpose(0,2,3,1)
        errors[name] = float(np.max(np.abs(value-ref[name])))
        np.testing.assert_allclose(value,ref[name],rtol=2e-4,atol=2e-6,err_msg=name)
    report = dict(device=args.device,matched_weight_tensors=len(used),max_abs_errors=errors)
    (folder/('comparison-'+args.device.replace(':','-')+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
