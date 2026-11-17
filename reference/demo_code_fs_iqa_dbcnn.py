import os
import math
import time
import random
import imageio
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from math import ceil
from typing import Optional
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from huggingface_hub import hf_hub_url

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Utility functions
# ================================================================

def clean_state_dict(state_dict):
    """Remove 'module.' prefix if checkpoint was saved from DataParallel."""
    cleaned = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned[name] = v
    return cleaned


def get_url_from_name(name: str, store_base: str = "hugging_face", base_url: str = None) -> str:
    """Map model file name to download URL."""
    if base_url is not None:
        return f"{base_url}/{name}"
    if store_base == "hugging_face":
        return hf_hub_url(repo_id="chaofengc/IQA-PyTorch-Weights", filename=name)
    if store_base == "github":
        return f"https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/{name}"
    return None


default_model_urls = {
    'csiq': get_url_from_name('DBCNN_CSIQ-8677d071.pth'),
    'tid2008': get_url_from_name('DBCNN_TID2008-4b47c5d1.pth'),
    'tid2013': get_url_from_name('DBCNN_TID2013-485d021d.pth'),
    'live': get_url_from_name('DBCNN_LIVE-97262bf4.pth'),
    'livec': get_url_from_name('DBCNN_LIVEC-83f6dad3.pth'),
    'livem': get_url_from_name('DBCNN_LIVEM-698474e3.pth'),
    'koniq': get_url_from_name('DBCNN_KonIQ10k-2de81c0a.pth'),
    'scnn': get_url_from_name('DBCNN_scnn-7ea73d75.pth'),
}


# ================================================================
# SCNN Model
# ================================================================

class SCNN(nn.Module):
    """Synthetic distortion CNN branch used in DBCNN."""

    def __init__(self, use_bn=True):
        super().__init__()
        self.num_class = 39
        self.use_bn = use_bn

        self.features = nn.Sequential(
            *self._make_layers(3, 48, 3, 1, 1),
            *self._make_layers(48, 48, 3, 2, 1),
            *self._make_layers(48, 64, 3, 1, 1),
            *self._make_layers(64, 64, 3, 2, 1),
            *self._make_layers(64, 64, 3, 1, 1),
            *self._make_layers(64, 64, 3, 2, 1),
            *self._make_layers(64, 128, 3, 1, 1),
            *self._make_layers(128, 128, 3, 1, 1),
            *self._make_layers(128, 128, 3, 2, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            *self._make_layers(128, 256, 1, 1, 0),
            *self._make_layers(256, 256, 1, 1, 0),
        )
        self.classifier = nn.Linear(256, self.num_class)

    def _make_layers(self, in_ch, out_ch, ksz, stride, pad):
        layers = [nn.Conv2d(in_ch, out_ch, ksz, stride, pad)]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(True))
        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ================================================================
# DBCNN feature backbone
# ================================================================

class DBCNNf(nn.Module):
    """Feature extraction backbone of DBCNN model."""

    def __init__(self,
                 pretrained=True,
                 pretrained_model_path=None,
                 use_bn=True,
                 default_mean=[0.485, 0.456, 0.406],
                 default_std=[0.229, 0.224, 0.225]):
        super().__init__()

        # VGG16 encoder
        self.features1 = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
        self.features1 = nn.Sequential(*self.features1[:-1])

        # SCNN branch
        scnn = SCNN(use_bn=use_bn)
        scnn_state = torch.load("data/DBCNN_scnn-7ea73d75.pth", map_location=device)
        scnn.load_state_dict(clean_state_dict(scnn_state), strict=True)
        self.features2 = scnn.features

        # Bilinear FC
        self.fc = nn.Linear(512 * 128, 1)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        # Load pretrained fc if available
        if pretrained:
            state = torch.load("data/DBCNN_KonIQ10k-2de81c0a.pth", map_location=device)['params']
            state = clean_state_dict(state)
            self.load_state_dict(state, strict=True)

    def preprocess(self, x):
        return (x - self.default_mean.to(x)) / self.default_std.to(x)

    def forward(self, x):
        x = self.preprocess(x)

        x1 = self.features1(x)
        x2 = self.features2(x)

        N, _, H, W = x1.size()
        if x2.size(2) != H or x2.size(3) != W:
            x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)

        x1 = x1.view(N, 512, -1)
        x2 = x2.view(N, 128, -1)

        # Bilinear combination
        xb = torch.bmm(x1, x2.transpose(1, 2)) / (H * W)
        xb = xb.view(N, -1)
        xb = torch.sqrt(xb + 1e-8)
        xb = F.normalize(xb)

        return xb


# ================================================================
# Scorer heads
# ================================================================

class DBCNNprev_fc(nn.Module):
    """512-d feature reducer."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512 * 128, 512 * 64),
            nn.BatchNorm1d(512 * 64),
            nn.LeakyReLU(0.01, True),

            nn.Linear(512 * 64, 512 * 32),
            nn.BatchNorm1d(512 * 32),
            nn.LeakyReLU(0.01, True),

            nn.Linear(512 * 32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01, True),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class DBCNNfc(nn.Module):
    """Final MOS predictor."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01, True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, True),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class Overall_f_model(nn.Module):
    """Full pipeline for feature extraction + 512-d mapping."""

    def __init__(self):
        super().__init__()
        self.feature_model = DBCNNf().to(device)
        self.prev_scorer_model = DBCNNprev_fc().to(device)

        prev_path = 'data/dbcnn_prevscorer_FS_01.pth'
        self.prev_scorer_model.load_state_dict(torch.load(prev_path, map_location=device))
        self.prev_scorer_model.eval()

    def forward(self, x):
        x = self.feature_model(x)
        return self.prev_scorer_model(x)


# Load full scoring model
feature_model = DBCNNf().to(device)
prev_scorer_model = DBCNNprev_fc().to(device)
scorer_model = DBCNNfc().to(device)

scorer_model.load_state_dict(torch.load('data/dbcnn_scorer_FS_01.pth', map_location=device))
prev_scorer_model.load_state_dict(torch.load('data/dbcnn_prevscorer_FS_01.pth', map_location=device))

scorer_model.eval()
prev_scorer_model.eval()


# ================================================================
# Dataset
# ================================================================

class MyCustomDataset(Dataset):
    """Koniq-10k dataset with MOS z-scores."""

    def __init__(self, path_gt, mode):
        self.mode = mode
        self.items = []

        df = pd.read_csv('data/koniq10k_scores_and_distributions.csv')
        valid_names = set(df['image_name'])

        all_imgs = sorted(os.listdir(path_gt))
        imgs = all_imgs[:8000] if mode == 'train' else all_imgs[8000:]

        for img in imgs:
            if img in valid_names:
                self.items.append(os.path.join(path_gt, img))

        random.shuffle(self.items)

        self.mos_dict = dict(zip(df['image_name'], df['MOS_zscore']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.float32)
        img = cv2.resize(img, (384, 512))
        img = img / 255.
        img = transforms.ToTensor()(img)

        mos = self.mos_dict[os.path.basename(path)]
        return img.to(device), mos


# ================================================================
# Randomized smoothing & robustness estimation
# ================================================================

def estimate_qu_ql(eps, sample_count, sigma, conf_threshold=0.99999):
    """Estimate index bounds for quantile-based confidence intervals."""
    theo_u = norm.cdf(eps / sigma)
    theo_l = norm.cdf(-eps / sigma)

    q_u_final = sample_count + 1
    q_l_final = 0

    for q in range(ceil(theo_u * sample_count), sample_count + 1):
        if norm.cdf((q - 1) - sample_count * theo_u) > conf_threshold:
            q_u_final = q
            break

    for q in range(math.floor(theo_l * sample_count), -1, -1):
        if 1 - norm.cdf((q - 1) - sample_count * theo_l) > conf_threshold:
            q_l_final = q
            break

    return q_u_final, q_l_final


def compute_median_smoothing(f, scorer, n, batch_size, q_u, q_l, sigma):
    """Apply Gaussian noise smoothing to feature representation."""
    f = f.repeat(batch_size, 1)
    scores = []

    for _ in range(n // batch_size):
        with torch.no_grad():
            noise = torch.randn_like(f) * sigma
            x_n = torch.clamp(f + noise, 0, 1)
            out = scorer(x_n)

            if len(scores) == 0:
                scores = out
            else:
                scores = torch.cat([scores, out], dim=0)

    scores_sorted = scores.sort()[0]

    return (
        [torch.median(scores).item()],
        [scores_sorted[q_l].item()],
        [scores_sorted[q_u].item()],
    )


# ================================================================
# Hessian-based robustness estimation
# ================================================================

def compute_epsilon_x_full(J, H_norm=None, sigma_f=0.25, eps=1e-12):
    """Compute input epsilon bound given Jacobian and Hessian."""
    sigma_J = torch.linalg.norm(J, ord=2).item()
    sigma_J = max(sigma_J, eps)

    if H_norm is None or H_norm < eps:
        return sigma_f / sigma_J

    L = H_norm
    return (-sigma_J + math.sqrt(sigma_J**2 + 2 * L * sigma_f)) / L


def estimate_hessian_norm_MC_fast(B, x, eps_x, N=1000, delta=1e-3, alpha=0.001):
    """Fast Monte Carlo Hessian estimation."""
    x = x.detach().requires_grad_(True)
    L_vals = []

    for _ in range(N):
        v = torch.randn_like(x)
        v /= v.norm()

        r = torch.rand(1).pow(1 / x.numel()) * eps_x
        z = (x + r * v).detach().requires_grad_(True)

        zp = (z + delta * v).detach().requires_grad_(True)
        zm = (z - delta * v).detach().requires_grad_(True)

        gp = torch.autograd.grad(B(zp).sum(), zp)[0]
        gm = torch.autograd.grad(B(zm).sum(), zm)[0]

        Hv = (gp - gm) / (2 * delta)
        L_vals.append(Hv.flatten().norm().item())

    L_emp = max(L_vals)
    volume_bound = math.log(1 / alpha) / N
    return L_emp, volume_bound


def InputVariationEstimator(B, x, sigma_f=0.25, n_hessian_samples=1000, alpha=0.001):
    """Main function computing full epsilon bound."""
    J = torch.autograd.functional.jacobian(B, x).reshape(B(x).numel(), -1)

    eps_lin = compute_epsilon_x_full(J, None, sigma_f=sigma_f)

    L_emp, vol_bound = estimate_hessian_norm_MC_fast(
        B, x, eps_lin, N=n_hessian_samples, alpha=alpha
    )

    eps_full = compute_epsilon_x_full(J, L_emp, sigma_f=sigma_f)
    return eps_full, L_emp, vol_bound


# ================================================================
# Experiment loop
# ================================================================

sigma = 0.1
epsilons = [0.005]
path_train = 'data/koniq'

for eps in epsilons:
    q_u, q_l = estimate_qu_ql(eps, sample_count=1000, sigma=sigma, conf_threshold=0.999)

    ds_val = MyCustomDataset(path_gt=path_train, mode='val')
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

    out_path = f'data/scorer_res/scorer_dbcnn_{eps}_{sigma}_norm_rem.csv'
    pd.DataFrame(columns=[
        'median', 'lower_b', 'upper_b', 'gt', 'epsilon_x', 'L_emp', 'vol_bound'
    ]).to_csv(out_path, index=False)

    full_model = Overall_f_model()

    df = pd.DataFrame()
    cnt = 0

    for y, gt in tqdm(dl_val):
        if cnt == 1000:
            break

        f = feature_model(y)
        f = prev_scorer_model(f)
        gt = gt.to(device)

        score, score_l, score_u = compute_median_smoothing(
            f, scorer_model, n=1000, batch_size=100,
            q_u=q_u, q_l=q_l, sigma=sigma
        )

        eps_x, L_emp, vol_bound = InputVariationEstimator(
            full_model, y, sigma_f=sigma, n_hessian_samples=100, alpha=0.001
        )

        df = pd.concat([df, pd.DataFrame([[
            score[0], score_l[0], score_u[0], gt.item(),
            eps_x, L_emp, vol_bound
        ]], columns=['median', 'lower_b', 'upper_b', 'gt',
                     'epsilon_x', 'L_emp', 'vol_bound'])])

        df.to_csv(out_path, index=False)
        cnt += 1