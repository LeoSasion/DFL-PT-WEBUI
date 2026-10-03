"""NCHW versions of the local TF ME losses, including encoded eye/mouth masks.

Reference: DFL-WEBUI 85fac834, Model_ME/Model.py and leras/ops/__init__old.py.
"""
import math
import torch
import torch.nn.functional as F


def gaussian_blur(x, radius):
    sigma=float(radius)
    if sigma<=0:
        return x
    size=max(3,int(4*sigma)); size += int(size%2==0)
    coords=torch.arange(size,device=x.device,dtype=x.dtype)-size//2
    g=torch.exp(-coords.square()/(2*sigma*sigma))
    kernel=g[:,None]*g[None,:];kernel=kernel/kernel.sum()
    return F.conv2d(x,kernel[None,None].repeat(x.shape[1],1,1,1),padding=size//2,groups=x.shape[1])


def _ssim_components(x,y,size):
    size=max(1,int(size))
    coords=torch.arange(size,device=x.device,dtype=x.dtype)-(size-1)/2
    logits=-.5*(coords[:,None].square()+coords[None,:].square())/(1.5**2)
    kernel=logits.flatten().softmax(0).reshape(size,size)
    kernel=kernel[None,None].repeat(x.shape[1],1,1,1)
    def reduce(z):return F.conv2d(z,kernel,groups=x.shape[1])
    a,b=reduce(x),reduce(y)
    num0=2*a*b;den0=a.square()+b.square()
    luminance=(num0+.01**2)/(den0+.01**2)
    cs=(2*reduce(x*y)-num0+.03**2)/(reduce(x.square()+y.square())-den0+.03**2)
    return (luminance*cs).mean((2,3)),cs.mean((2,3))


def dssim(x,y,filter_size):
    ssim,_=_ssim_components(x.float(),y.float(),filter_size)
    return (1-ssim)/2


def ms_ssim_loss(x,y,use_l1=False):
    resolution=x.shape[-1]
    factors=[p for i,p in enumerate((.0448,.2856,.3001,.2363,.1333)) if resolution//(2**i)>=11]
    if not factors:
        raise ValueError('MS-SSIM requires resolution >= 11')
    if sum(factors)<1:
        total=sum(factors);factors=[p/total for p in factors]
    original_x,original_y=x,y
    values=[]
    for i,power in enumerate(factors):
        ssim,cs=_ssim_components(x,y,11)
        value = ssim if i==len(factors)-1 else cs
        # Preserve TF's nonnegative value while avoiding 0 * inf in autograd.
        values.append(value.clamp_min(1e-12).pow(power)*(value>0))
        if i<len(factors)-1:
            # TF ssim_multiscale pads odd dimensions symmetrically before pooling.
            ph,pw=x.shape[-2]%2,x.shape[-1]%2
            if ph or pw:
                x=F.pad(x,(0,pw,0,ph),mode='replicate');y=F.pad(y,(0,pw,0,ph),mode='replicate')
            x=F.avg_pool2d(x,2);y=F.avg_pool2d(y,2)
    result=1-torch.stack(values).prod(0).mean(1)
    if use_l1:
        # Match local ME's MsSsim L1 branch, including its sum over scales.
        sigma=(.5,1.,2.,4.,8.)[len(factors)-1]
        coord=torch.arange(resolution,device=x.device,dtype=x.dtype)-(resolution/2-.5)
        g=torch.exp(-coord.square()/(2*sigma*sigma));kernel=g[:,None]*g[None,:];kernel/=kernel.sum()
        l1=((original_x-original_y).abs()*kernel).sum((2,3)).mean(1)*len(factors)
        result=.84*result+.16*l1
    return result


def reconstruction(x,y,kind,resolution):
    if kind=='MS-SSIM+L1':
        return 10*ms_ssim_loss(x,y,True)
    if kind=='MS-SSIM':
        value=10*ms_ssim_loss(x,y)
    elif resolution<256:
        value=10*dssim(x,y,int(resolution/11.6)).mean(1)
    else:
        value=5*dssim(x,y,int(resolution/11.6)).mean(1)+5*dssim(x,y,int(resolution/23.2)).mean(1)
    return value+10*(x-y).square().mean((1,2,3))


def priority_mask(encoded,eyes,mouth):
    both=(encoded-1).clamp(0,1);mouth_mask=(encoded-2).clamp(0,1)
    if eyes and mouth:return both
    if eyes:return (both-mouth_mask).clamp(0,1)
    if mouth:return mouth_mask
    return torch.zeros_like(encoded)


def style_loss(x,y,radius):
    x,y=gaussian_blur(x,radius),gaussian_blur(y,radius)
    xm,ym=x.mean((2,3)),y.mean((2,3))
    xs=(x.var((2,3),unbiased=False)+1e-5).sqrt()
    ys=(y.var((2,3),unbiased=False)+1e-5).sqrt()
    return ((xm-ym).square()+(xs-ys).square()).mean(1)


def blur_training_background(target, mask, resolution):
    anti = 1-mask
    numerator = gaussian_blur(target*anti,resolution/128)
    denominator = 1-gaussian_blur(mask,resolution/128)
    denominator = torch.where(denominator==0,torch.ones_like(denominator),denominator)
    return target*mask+(numerator/denominator)*anti


def training_mask(mask, resolution):
    return gaussian_blur(mask.clamp(0,1),max(1,resolution//32)).clamp(0,.5)*2


def total_variation_mse(image):
    # Per-example form of the original ME GAN background regularizer.
    dx = image[:,:,:,1:]-image[:,:,:,:-1]
    dy = image[:,:,1:,:]-image[:,:,:-1,:]
    return dx.square().sum((1,2,3))+dy.square().sum((1,2,3))


def me_losses(config,source,destination,outputs):
    _,src,srcm,srcem=source;_,dst,dstm,dstem=destination
    srcm,dstm=srcm.clamp(0,1),dstm.clamp(0,1)
    if config.blur_out_mask:
        src=blur_training_background(src,srcm,config.resolution)
        dst=blur_training_background(dst,dstm,config.resolution)
    sm=training_mask(srcm,config.resolution)
    dm=training_mask(dstm,config.resolution)
    def side(target,pred,target_mask,pred_mask,blurred,encoded):
        mask=blurred if config.masked_training else 1
        result=reconstruction(target*mask,pred*mask,config.loss_function,config.resolution)
        result=result+10*(target_mask-pred_mask).square().mean((1,2,3))
        if config.eyes_prio or config.mouth_prio:
            selected=priority_mask(encoded,config.eyes_prio,config.mouth_prio)
            result=result+300*((target-pred)*selected).abs().mean((1,2,3))
        if config.background_power:
            result=result+config.background_power*reconstruction(target,pred,config.loss_function,config.resolution)
        return result
    sl=side(src,outputs['src'],srcm,outputs['src_mask'],sm,srcem)
    dl=side(dst,outputs['dst'],dstm,outputs['dst_mask'],dm,dstem)
    if config.face_style_power:
        sl=sl+100*config.face_style_power*style_loss(outputs['swap_no_code_grad']*outputs['swap_mask'].detach(),(outputs['dst']*outputs['dst_mask']).detach(),config.resolution//8)
    if config.bg_style_power:
        anti=1-sm.detach()
        a,b=outputs['swap']*anti,dst*anti
        sl=sl+.1*config.bg_style_power*(dssim(a,b,int(config.resolution/11.6)).mean(1)+(a-b).square().mean((1,2,3)))
    return sl,dl
