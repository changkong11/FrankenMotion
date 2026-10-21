"""Regression: rejected gradients must not allow indefinite empty training."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from pytorch_lightning import LightningModule
import src.prepare  # noqa
from src.model.gaussian import GaussianDiffusion


class GradientStopTests(unittest.TestCase):
    def model(self, folder):
        model = GaussianDiffusion.__new__(GaussianDiffusion)
        LightningModule.__init__(model)
        model.weight = torch.nn.Parameter(torch.tensor([1.0]))
        model.divergence_guard_max_grad_norm = 100.0
        model.divergence_guard_max_finite_spikes = 3
        model._consecutive_gradient_anomalies = 0
        model._last_training_batch_context = {'batch_size': 1}
        model.log = lambda *args, **kwargs: None
        def save(path):
            torch.save(model.state_dict(), path)
        model._trainer = SimpleNamespace(current_epoch=331, global_step=12,
            checkpoint_callback=SimpleNamespace(dirpath=folder), save_checkpoint=save)
        return model

    def test_repeated_rejections_stop_before_update(self):
        with tempfile.TemporaryDirectory() as folder:
            model = self.model(folder)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            for _ in range(2):
                model.weight.grad = torch.tensor([200.0])
                model.configure_gradient_clipping(optimizer, 1.0)
                optimizer.step()
                self.assertEqual(model.weight.item(), 1.0)
            model.weight.grad = torch.tensor([200.0])
            with self.assertRaisesRegex(RuntimeError, '3 consecutive'):
                model.configure_gradient_clipping(optimizer, 1.0)
            self.assertIsNone(model.weight.grad)
            self.assertTrue(Path(folder, 'gradient-diverged-step=12.ckpt').is_file())

    def test_healthy_gradient_resets_counter_and_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            model = self.model(folder)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            model.weight.grad = torch.tensor([200.0])
            model.configure_gradient_clipping(optimizer, 1.0)
            model.weight.grad = torch.tensor([0.5])
            model.configure_gradient_clipping(optimizer, 1.0)
            optimizer.step()
            self.assertEqual(model._consecutive_gradient_anomalies, 0)
            self.assertLess(model.weight.item(), 1.0)

    def test_nonfinite_stops_immediately(self):
        with tempfile.TemporaryDirectory() as folder:
            model = self.model(folder)
            model.weight.grad = torch.tensor([float('nan')])
            with self.assertRaisesRegex(RuntimeError, 'Gradient anomaly'):
                model.configure_gradient_clipping(None, 1.0)
            self.assertIsNone(model.weight.grad)


if __name__ == '__main__':
    unittest.main()
