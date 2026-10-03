import torch
from torch.utils.checkpoint import checkpoint
from core.leras import nn


class MENetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        nn.initialize(nn.DeviceConfig.CPU(), data_format='NCHW')
        self.archi_type, _, options = config.archi.partition('-')
        if self.archi_type not in ('df', 'liae') or set(options) - set('udtc'):
            raise ValueError('Unsupported ME architecture')
        self.use_rg = config.use_rg
        # AMP is controlled by the engine. Parameter dtype and names stay stable
        # across CPU loading, RG, and export.
        arch = nn.DeepFakeArchi(config.resolution, use_fp16=False, opts=options)
        self.encoder = arch.Encoder(3, config.e_dims, name='encoder')
        inter_in = self.encoder.get_out_res(config.resolution)**2 * self.encoder.get_out_ch()
        if self.archi_type == 'df':
            self.inter = arch.Inter(inter_in, config.ae_dims, config.ae_dims, name='inter')
            self.decoder_src = arch.Decoder(config.ae_dims, config.d_dims, config.d_mask_dims, name='decoder_src')
            self.decoder_dst = arch.Decoder(config.ae_dims, config.d_dims, config.d_mask_dims, name='decoder_dst')
        else:
            self.inter_AB = arch.Inter(inter_in, config.ae_dims, config.ae_dims*2, name='inter_AB')
            self.inter_B = arch.Inter(inter_in, config.ae_dims, config.ae_dims*2, name='inter_B')
            self.decoder = arch.Decoder(config.ae_dims*4, config.d_dims, config.d_mask_dims, name='decoder')
        for module in self.children():
            weights = module.get_weights()
            if {id(p) for p in weights} != {id(p) for p in module.parameters()}:
                raise RuntimeError('Unregistered network parameters')
        if self.archi_type == 'liae' and not config.random_warp:
            self.inter_AB.requires_grad_(False)

    def _run(self, module, value):
        # The TF RG variant only recomputes activations; it does not alter the
        # architecture. Non-reentrant checkpointing also works with image inputs
        # that do not require gradients and with frozen LIAE intermediates.
        if (self.use_rg and self.training and torch.is_grad_enabled()
                and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export()):
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def forward(self, source, destination, face_style=False):
        if self.archi_type == 'df':
            src_code = self._run(self.inter, self._run(self.encoder, source))
            dst_code = self._run(self.inter, self._run(self.encoder, destination))
            src, src_mask = self._run(self.decoder_src, src_code)
            dst, dst_mask = self._run(self.decoder_dst, dst_code)
            swap, swap_mask = self._run(self.decoder_src, dst_code)
            swap_code, swap_decoder = dst_code, self.decoder_src
        else:
            src_ab = self._run(self.inter_AB, self._run(self.encoder, source))
            dst_code = self._run(self.encoder, destination)
            dst_ab = self._run(self.inter_AB, dst_code)
            dst_b = self._run(self.inter_B, dst_code)
            src, src_mask = self._run(self.decoder, torch.cat((src_ab, src_ab), 1))
            dst, dst_mask = self._run(self.decoder, torch.cat((dst_b, dst_ab), 1))
            swap_code, swap_decoder = torch.cat((dst_ab, dst_ab), 1), self.decoder
            swap, swap_mask = self._run(swap_decoder, swap_code)
        result = dict(src=src, src_mask=src_mask, dst=dst, dst_mask=dst_mask,
                      swap=swap, swap_mask=swap_mask)
        if self.archi_type == 'df':
            result.update(src_code=src_code, dst_code=dst_code)
        if face_style:
            result['swap_no_code_grad'] = self._run(swap_decoder, swap_code.detach())[0]
        return result

    def predict(self, destination):
        code = self.encoder(destination)
        if self.archi_type == 'df':
            code = self.inter(code)
            swap, src_mask = self.decoder_src(code)
            _, dst_mask = self.decoder_dst(code)
            return swap, src_mask, dst_mask
        ab, b = self.inter_AB(code), self.inter_B(code)
        swap, src_mask = self.decoder(torch.cat((ab, ab), 1))
        _, dst_mask = self.decoder(torch.cat((b, ab), 1))
        return swap, src_mask, dst_mask
