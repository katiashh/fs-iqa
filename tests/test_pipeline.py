import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from fs_iqa.ive import estimate_input_radius
from fs_iqa.models import DBCNNf, DBCNNprev_fc, DBCNNfc, load_state
from fs_iqa.smoothing import predict, sample_scores


class PipelineTests(unittest.TestCase):
    def test_sample_remainder_and_seed(self):
        scorer = nn.Linear(3, 1).eval()
        features = torch.ones(1, 3) * 0.5
        def sample():
            return sample_scores(features, scorer, n=103, batch_size=20,
                                 generator=torch.Generator().manual_seed(7))
        a, b = sample(), sample()
        self.assertEqual(tuple(a.shape), (103,))
        self.assertTrue(torch.equal(a, b))

    def test_clipping_matches_reference(self):
        class Record(nn.Module):
            def forward(self, x):
                self.min_val = x.min().item()
                self.max_val = x.max().item()
                return x.sum(dim=1)
        scorer = Record().eval()
        sample_scores(torch.zeros(1, 3), scorer, n=100, sigma=100)
        self.assertEqual(scorer.min_val, 0)
        self.assertEqual(scorer.max_val, 1)

    def test_score_only_and_interval_use_same_predictor(self):
        scorer = nn.Linear(3, 1).eval()
        args = (torch.ones(1, 3)*0.5, scorer)
        a = predict(*args, score_only=True, generator=torch.Generator().manual_seed(0))
        b = predict(*args, generator=torch.Generator().manual_seed(0))
        self.assertEqual(a["median"], b["median"])
        self.assertIsNone(a["lower_b"])
        self.assertEqual(b["interval_status"], "finite")

    def test_training_scorer_rejected(self):
        with self.assertRaises(ValueError):
            sample_scores(torch.ones(1, 3), nn.Linear(3, 1))

    def test_ive_linear_map(self):
        model = nn.Linear(2, 2, bias=False, dtype=torch.float64).eval()
        with torch.no_grad():
            model.weight.copy_(torch.diag(torch.tensor([2., 3.])))
        r = estimate_input_radius(model, torch.tensor([[0.2, 0.4]], dtype=torch.float64),
                                  feature_radius=0.3, n_hessian=4)
        self.assertAlmostEqual(r["jacobian_norm"], 3)
        self.assertAlmostEqual(r["curvature_proxy"], 0)
        self.assertAlmostEqual(r["epsilon_x_estimate"], 0.1)
        self.assertEqual(r["ive_status"], "empirical_proxy")

    def test_ive_zero_jacobian_abstains(self):
        model = nn.Linear(2, 2, bias=False).eval()
        with torch.no_grad():
            model.weight.zero_()
        r = estimate_input_radius(model, torch.ones(1, 2), n_hessian=2)
        self.assertEqual(r["ive_status"], "abstained")
        self.assertIsNone(r["epsilon_x_estimate"])

    def test_checkpoint_wrapper_and_dataparallel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.pth"
            torch.save({"params": {"module.weight": torch.ones(2, 2)}}, path)
            self.assertTrue(torch.equal(load_state(path)["weight"], torch.ones(2, 2)))

    def test_published_shapes_without_allocating_gigabytes(self):
        with torch.device("meta"):
            backbone, ftn, scorer = DBCNNf(), DBCNNprev_fc(), DBCNNfc()
        self.assertEqual(tuple(backbone.fc.weight.shape), (1, 65536))
        self.assertEqual(tuple(ftn.fc[0].weight.shape), (32768, 65536))
        self.assertEqual(tuple(ftn.fc[6].weight.shape), (512, 16384))
        self.assertEqual(tuple(scorer.fc[-1].weight.shape), (1, 64))
