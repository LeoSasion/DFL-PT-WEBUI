"""TAEQI2.1 native operators and scope guards; random weights are test-only."""
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from tae._vendor import taesd
from tae.config import TAE_RESOLUTIONS, TrainConfig
from tae.vae import TAEQI21Codec, VENDOR_SHA256, TAIL_START
from tae.checkpoint import file_sha256

def native_codec(mode='frozen'):
    wrapper = TAEQI21Codec.__new__(TAEQI21Codec)
    nn.Module.__init__(wrapper)
    model = taesd.TAESD(encoder_path=None, decoder_path=None, latent_channels=64,
                       arch_variant='f16', image_channels=4)
    wrapper.encoder, wrapper.decoder = model.encoder, model.decoder
    wrapper.set_decoder_mode(mode)
    wrapper.eval()
    return wrapper

class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pinned_vendor_and_configuration(self):
        self.assertEqual(file_sha256(taesd.__file__), VENDOR_SHA256)
        for res in TAE_RESOLUTIONS:
            self.assertEqual(TrainConfig(resolution=res).validate().resolution, res)
            self.assertEqual(res % 64, 0)
        for res in (64, 128, 320, 648, 672, 800, 1024):
            with self.assertRaises(ValueError):
                TrainConfig(resolution=res).validate()
        for mode in ('frozen','tail','full'):
            TrainConfig(decoder_mode=mode).validate()
        for value in (0, -1, 2, float('nan')):
            with self.assertRaises(ValueError):
                TrainConfig(decoder_lr_multiplier=value).validate()

    def test_rgba_encoding_decoder_gradient_and_exact_training_scopes(self):
        for mode, count in (('frozen', 0), ('tail', 562960), ('full', 7644560)):
            with self.subTest(mode=mode):
                codec = native_codec(mode).train()
                image = torch.rand(1,3,32,32, requires_grad=True)
                z = codec.encode(image)
                self.assertEqual(z.shape, (1,64,2,2))
                with torch.no_grad():
                    reference = codec.encoder(torch.cat([image, torch.ones_like(image[:,:1])],1))
                torch.testing.assert_close(z, reference, rtol=0, atol=0)
                self.assertFalse(z.requires_grad)
                z.requires_grad_()
                out = codec.decode(z)
                self.assertEqual(out.shape, image.shape)
                out.square().mean().backward()
                self.assertGreater(z.grad.abs().sum().item(),0)
                self.assertIsNone(image.grad)
                self.assertEqual(sum(p.numel() for p in codec.decoder.parameters() if p.requires_grad), count)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in codec.encoder.parameters()))
                for p in codec.decoder.parameters():
                    self.assertEqual(p.grad is not None, p.requires_grad)
                codec.assert_frozen()

    def test_invalid_inputs_and_encoder_unfreeze_rejected(self):
        codec = native_codec()
        codec.train()
        self.assertFalse(codec.training)
        for shape in ((1,3,31,32), (1,4,32,32)):
            with self.assertRaises(ValueError):
                codec.encode(torch.zeros(shape))
        with self.assertRaises(ValueError):
            codec.encode(torch.full((1,3,32,32),1.1))
        next(codec.encoder.parameters()).requires_grad_(True)
        with self.assertRaises(RuntimeError):
            codec.encode(torch.zeros(1,3,32,32))

    def test_wrong_asset_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'wrong.safetensors'
            p.write_bytes(b'wrong asset')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                TAEQI21Codec(p)

if __name__ == '__main__':
    unittest.main()

