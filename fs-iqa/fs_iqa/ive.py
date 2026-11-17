"""Empirical IVE proxy from the supplied demo; NOT an input certificate.

Finite differences act on the gradient of B(x).sum(). Consequently the
curvature statistic is directional curvature of the sum of features,
not an upper bound on the Hessian operator norm of vector-valued B.
"""

import math

import torch

from .statistics import quadratic_radius


def estimate_input_radius(feature_map, image, *, feature_radius=0.1,
                          n_hessian=100, fd_step=1e-3, jacobian_threshold=1e-3,
                          generator=None):
    if image.shape[0] != 1:
        raise ValueError("IVE expects one image")
    if (n_hessian < 1 or not math.isfinite(fd_step) or fd_step <= 0
            or not math.isfinite(jacobian_threshold) or jacobian_threshold < 0
            or not math.isfinite(feature_radius) or feature_radius < 0):
        raise ValueError("invalid IVE parameters")
    if any(module.training for module in feature_map.modules()):
        raise ValueError("feature_map must be in eval mode")
    x = image.detach().requires_grad_(True)
    n_features = feature_map(x).numel()
    jacobian = torch.autograd.functional.jacobian(feature_map, x).reshape(n_features, -1)
    j_norm = torch.linalg.matrix_norm(jacobian, ord=2).item()
    del jacobian
    if not math.isfinite(j_norm) or j_norm <= jacobian_threshold:
        return {"epsilon_x_estimate": None, "jacobian_norm": j_norm,
                "curvature_proxy": None, "ive_status": "abstained"}
    eps_linear = quadratic_radius(j_norm, 0, feature_radius)
    estimates = []
    for _ in range(n_hessian):
        direction = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        direction = direction / direction.norm().clamp_min(torch.finfo(x.dtype).tiny)
        # One point uniformly distributed in the initial Euclidean ball.
        radius = torch.rand((), device=x.device, dtype=x.dtype, generator=generator)
        radius = radius.pow(1 / x.numel()) * eps_linear
        z = x.detach() + radius * direction
        zp = (z + fd_step * direction).detach().requires_grad_(True)
        zm = (z - fd_step * direction).detach().requires_grad_(True)
        gp = torch.autograd.grad(feature_map(zp).sum(), zp)[0]
        gm = torch.autograd.grad(feature_map(zm).sum(), zm)[0]
        value = ((gp - gm) / (2 * fd_step)).norm().item()
        if not math.isfinite(value):
            return {"epsilon_x_estimate": None, "jacobian_norm": j_norm,
                    "curvature_proxy": None, "ive_status": "abstained"}
        estimates.append(value)
    proxy = max(estimates)
    return {
        "epsilon_x_estimate": quadratic_radius(j_norm, proxy, feature_radius),
        "jacobian_norm": j_norm,
        "curvature_proxy": proxy,
        "ive_status": "empirical_proxy",
    }
