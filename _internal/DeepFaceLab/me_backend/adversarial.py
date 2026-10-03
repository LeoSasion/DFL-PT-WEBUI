"""Independent ME image/latent discriminators and their training objectives.

Generator weights are never registered here. The engine owns separate optimizer
and checkpoint state, freezing these parameters while calculating generator loss.
The two GAN heads and TrueFace label direction follow the original ME backend.
"""
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

from .losses import blur_training_background, training_mask, total_variation_mse


def _patch_layers(target, max_layers=9):
    candidates = {}
    for count in range(1,max_layers+1):
        # Descending order preserves the reference architecture's tie breaker.
        for strides in product((2,1),repeat=count-1):
            layers = (2,)+strides
            field, jump = 1, 1
            for stride in layers:
                field += 2*jump
                jump *= stride
            rank = (count,-sum(layers))
            if field not in candidates or rank < candidates[field][0]:
                candidates[field] = (rank,layers)
    field = min(candidates,key=lambda value:(abs(value-target),value))
    return field,candidates[field][1]


class UNetPatchDiscriminator(nn.Module):
    """Patch-dependent U-Net with bottleneck and full-resolution logit heads."""
    def __init__(self, patch_size, in_ch=3, base_ch=16):
        super().__init__()
        self.receptive_field,self.strides = _patch_layers(patch_size)
        channels = [min(base_ch*2**i,512) for i in range(len(self.strides)+1)]
        self.in_conv = nn.Conv2d(in_ch,channels[0],1)
        self.convs = nn.ModuleList([
            _SameConv2d(channels[i],channels[i+1],3,stride=stride)
            for i,stride in enumerate(self.strides)])
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i+1]*(1 if i==len(self.strides)-1 else 2),
                               channels[i],3,stride=stride,
                               padding=1 if stride==1 else 0)
            for i,stride in reversed(list(enumerate(self.strides)))])
        self.center_out = nn.Conv2d(channels[-1],1,1)
        self.center_conv = nn.Conv2d(channels[-1],channels[-1],1)
        self.out_conv = nn.Conv2d(channels[0]*2,1,1)

    def forward(self, image):
        value = F.leaky_relu(self.in_conv(image),.2)
        skips = []
        for conv in self.convs:
            skips.append(value)
            value = F.leaky_relu(conv(value),.2)
        center = self.center_out(value)
        value = F.leaky_relu(self.center_conv(value),.2)
        for conv,skip in zip(self.upconvs,reversed(skips)):
            value = F.leaky_relu(conv(value),.2)
            # SAME downsampling rounds odd dimensions up. Remove the matching
            # final row/column on the transpose path instead of stretching it.
            value = value[:,:,:skip.shape[-2],:skip.shape[-1]]
            value = torch.cat((skip,value),1)
        return center,self.out_conv(value)


class _SameConv2d(nn.Conv2d):
    def forward(self, value):
        height,width = value.shape[-2:]
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        ph = max(0,((height+stride-1)//stride-1)*stride+kernel-height)
        pw = max(0,((width+stride-1)//stride-1)*stride+kernel-width)
        return super().forward(F.pad(value,(pw//2,pw-pw//2,ph//2,ph-ph//2)))


class CodeDiscriminator(nn.Module):
    """DF latent-domain discriminator (destination=real, source=fake)."""
    def __init__(self, in_ch, code_res, ch=256):
        super().__init__()
        layers = []
        for i in range(1+code_res//8):
            out_ch = ch*min(2**i,8)
            layers.append(_SameConv2d(in_ch,out_ch,4 if i==0 else 3,stride=2))
            in_ch = out_ch
        self.convs = nn.ModuleList(layers)
        self.out_conv = nn.Conv2d(in_ch,1,1)

    def forward(self, code):
        for conv in self.convs:
            code = F.leaky_relu(conv(code),.1)
        return self.out_conv(code)


def _bce(logits, labels):
    return F.binary_cross_entropy_with_logits(logits,labels,reduction='none').flatten(1).mean(1)


class AdversarialTraining(nn.Module):
    def __init__(self, config, network):
        super().__init__()
        self.config = config
        self.gan = (UNetPatchDiscriminator(config.gan_patch_size,base_ch=config.gan_dims)
                    if config.gan_power else None)
        self.true_face = None
        if config.true_face_power:
            if network.archi_type != 'df':
                raise ValueError('TrueFace latent discrimination requires a df architecture')
            self.true_face = CodeDiscriminator(network.inter.get_out_ch(),network.inter.get_out_res())
        self.enabled = self.gan is not None or self.true_face is not None

    def _source_images(self, outputs, source):
        target,full_mask = source[1].float(),source[2].float().clamp(0,1)
        if self.config.blur_out_mask:
            target = blur_training_background(target,full_mask,self.config.resolution)
        mask = training_mask(full_mask,self.config.resolution)
        prediction = outputs['src'].float()
        if self.config.masked_training:
            return prediction*mask,target*mask,prediction,target,mask
        return prediction,target,prediction,target,mask

    def _labels(self, logits, real):
        labels = torch.full_like(logits,float(real))
        if self.config.gan_noise:
            flips = torch.rand_like(labels) < self.config.gan_noise
            labels = torch.where(flips,1-labels,labels)
        return labels*(1-self.config.gan_smoothing)

    def generator_loss(self, outputs, source, destination):
        loss = outputs['src'].float().new_zeros(outputs['src'].shape[0])
        if self.gan is not None:
            fake,_,prediction,target,mask = self._source_images(outputs,source)
            loss = loss+self.config.gan_power*sum(
                _bce(logits,torch.ones_like(logits)) for logits in self.gan(fake))
            if self.config.masked_training:
                loss = loss+1e-6*total_variation_mse(prediction)
                loss = loss+.02*((prediction-target)*(1-mask)).square().mean((1,2,3))
        if self.true_face is not None:
            logits = self.true_face(outputs['src_code'].float())
            loss = loss+self.config.true_face_power*_bce(logits,torch.ones_like(logits))
        return loss

    def discriminator_loss(self, outputs, source, destination):
        # Detaching internally prevents an accidental second generator update.
        detached = {name:value.detach() for name,value in outputs.items()}
        loss = outputs['src'].float().new_zeros(outputs['src'].shape[0])
        if self.gan is not None:
            fake,real,_,_,_ = self._source_images(detached,source)
            for logits in self.gan(fake.detach()):
                loss = loss+_bce(logits,self._labels(logits,False))
            for logits in self.gan(real.detach()):
                loss = loss+_bce(logits,self._labels(logits,True))
        if self.true_face is not None:
            src = self.true_face(detached['src_code'].float())
            dst = self.true_face(detached['dst_code'].float())
            loss = loss+.5*(_bce(src,torch.zeros_like(src))+_bce(dst,torch.ones_like(dst)))
        return loss.mean()
