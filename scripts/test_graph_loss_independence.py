import unittest
import torch
import src.prepare  # noqa: F401
from src.model.gaussian import GaussianDiffusion


class GraphLossTests(unittest.TestCase):
    def test_graph_gradients_without_state_heads(self):
        model = GaussianDiffusion.__new__(GaussianDiffusion)
        torch.nn.Module.__init__(model)
        model.coordination_fps = 20.0
        torch.manual_seed(42)
        logits = torch.randn(2, 5, 7, 7, requires_grad=True)
        mask = torch.tensor([[True]*5, [True]*3+[False]*2])
        aux = dict(relation_weights=logits.softmax(-1), relation_mask=mask,
                   relation_static_adjacency=torch.eye(7, dtype=torch.bool),
                   relation_allowed_mask=torch.ones_like(logits, dtype=torch.bool))
        motion = torch.randn(2, 5, 205) * .01
        affinity, smoothness, contact, balance = model._coordination_graph_losses(aux, motion, mask)
        self.assertIsNone(contact)
        self.assertIsNone(balance)
        self.assertTrue(torch.isfinite(affinity) and torch.isfinite(smoothness))
        (affinity + smoothness).backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertEqual(logits.grad[1, 3:].abs().sum().item(), 0)
        for name, shape in [('contact_state_logits', (2,5,2)), ('balance_state', (2,5,2))]:
            aux[name] = torch.zeros(shape, requires_grad=True)
        values = model._coordination_graph_losses(aux, motion, mask)
        self.assertIsNotNone(values[2])
        self.assertTrue(torch.isfinite(values[2]))
        masked = model._coordination_graph_losses(aux, motion, mask, sample_weight=torch.zeros(2))
        self.assertEqual(float(masked[0]), 0.0)
        self.assertEqual(float(masked[1]), 0.0)


if __name__ == '__main__':
    unittest.main()
