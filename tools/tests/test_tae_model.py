import unittest

import torch
from torch import nn
from torch.nn import functional as F

from tae.model import LIAEModel


from test_tae_engine import TinyFrozenVAE as ToyFrozenVAE


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(5)
        self.model = LIAEModel(ToyFrozenVAE(), inter_dims=8, hidden_dims=8, resolution=64)
        self.src = torch.rand(2, 3, 64, 64)
        self.dst = torch.rand(2, 3, 64, 64)

    def test_real_pixel_backward_updates_inter_and_keeps_vae_frozen(self):
        model = self.model.train()
        before = {key: value.clone() for key, value in model.vae.state_dict().items()}
        inter_before = model.inter_AB.compress.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        source = self.src.requires_grad_()
        src_out = model.reconstruct(source, domain="src")
        # Pixel loss alone proves the graph crosses the decoder. A mask-only
        # loss would fail to catch an accidentally detached decoder.
        F.mse_loss(src_out["image"], torch.zeros_like(self.src)).backward()
        self.assertIsNone(source.grad)
        self.assertGreater(model.inter_AB.compress.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.inter_B.compress.weight.grad)
        dst_out = model.reconstruct(self.dst, domain="dst")
        F.mse_loss(dst_out["image"], self.dst).backward()
        self.assertGreater(model.inter_B.compress.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(inter_before, model.inter_AB.compress.weight))
        for key, value in model.vae.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        self.assertTrue(all(p.grad is None for p in model.vae.parameters()))
        self.assertFalse(model.vae.training)

    def test_b_branch_changes_destination_but_not_source_or_swap(self):
        model = self.model.eval()
        with torch.no_grad():
            before = model(self.src, self.dst, include_swap=True)
            model.inter_B.expand.bias.add_(1.0)
            after = model(self.src, self.dst, include_swap=True)
        self.assertTrue(torch.equal(before["src"], after["src"]))
        self.assertTrue(torch.equal(before["swap"], after["swap"]))
        self.assertFalse(torch.equal(before["dst"], after["dst"]))

    def test_no_latent_identity_bypass(self):
        model = self.model.eval()
        with torch.no_grad():
            for module in (model.inter_AB, model.inter_B):
                for parameter in module.parameters():
                    parameter.zero_()
            first = model.predict(self.src)
            second = model.predict(self.dst)
        self.assertTrue(torch.equal(first["swap"], second["swap"]))

    def test_state_excludes_encoder_includes_decoder_and_roundtrips(self):
        state = self.model.trainable_state_dict()
        self.assertTrue(state)
        self.assertFalse(any(key.startswith("vae.encoder.") for key in state))
        other = LIAEModel(ToyFrozenVAE(), inter_dims=8, hidden_dims=8, resolution=64)
        other.load_trainable_state_dict(state)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, other.trainable_state_dict()[key]))
        bad = dict(state)
        bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):
            other.load_trainable_state_dict(bad)

    def test_multiple_sizes_masks_and_shapes(self):
        for size in (64, 128, 192):
            self.model = LIAEModel(ToyFrozenVAE(), inter_dims=8, hidden_dims=8, resolution=size)
            image = torch.rand(1, 3, size, size)
            with torch.no_grad():
                result = self.model(image, image, include_swap=True)
                prediction = self.model.predict(image)
            for name in ("src", "dst", "swap"):
                self.assertEqual(result[name].shape, (1, 3, size, size))
                self.assertEqual(result[name + "_mask"].shape, (1, 1, size, size))
                self.assertTrue(torch.all((result[name + "_mask"] >= 0) & (result[name + "_mask"] <= 1)))
            torch.testing.assert_close(prediction["swap"], result["swap"].clamp(0, 1))
            torch.testing.assert_close(prediction["swap_mask"], result["swap_mask"])
            torch.testing.assert_close(prediction["dst_mask"], result["dst_mask"])


if __name__ == "__main__":
    unittest.main()
