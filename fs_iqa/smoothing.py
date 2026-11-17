"""Gaussian feature sampling; the expensive backbone is evaluated once."""

import math

import torch

from .statistics import interval_from_scores


@torch.no_grad()
def sample_scores(features, scorer, n=1000, batch_size=100, sigma=0.1,
                  clip_features=True, generator=None):
    if features.ndim != 2 or features.shape[0] != 1:
        raise ValueError("features must have shape (1, feature_dim)")
    if n < 1 or batch_size < 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("n, batch_size and sigma must be positive")
    if any(module.training for module in scorer.modules()):
        raise ValueError("scorer must be in eval mode")
    values = []
    for offset in range(0, n, batch_size):
        count = min(batch_size, n - offset)
        noise = torch.randn((count, features.shape[1]), device=features.device,
                            dtype=features.dtype, generator=generator) * sigma
        noised = features + noise
        if clip_features:
            noised = noised.clamp(0, 1)
        output = scorer(noised).reshape(-1)
        if output.numel() != count:
            raise ValueError("scorer must return one scalar per feature vector")
        if not torch.isfinite(output).all():
            raise ValueError("scorer returned nonfinite values")
        values.append(output.cpu())
    return torch.cat(values)


def predict(features, scorer, *, sigma=0.1, kappa=1.0, n=1000,
            batch_size=100, confidence=0.999, score_only=False,
            clip_features=True, generator=None):
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    scores = sample_scores(features, scorer, n, batch_size, sigma,
                           clip_features, generator).numpy()
    if score_only:
        import numpy as np
        return {"median": float(np.median(scores)), "lower_b": None,
                "upper_b": None, "interval_status": "not_requested",
                "n_samples": len(scores)}
    return interval_from_scores(scores, sigma, kappa * sigma, confidence)
