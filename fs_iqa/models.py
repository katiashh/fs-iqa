"""Checkpoint-compatible DBCNN / FTN / Scorer definitions from the supplement.

No model is constructed, no weights loaded, and no downloads started at import.
The published FTN is large: approximately 2.69 billion parameters.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


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


class DBCNNf(nn.Module):
    """Feature extraction backbone of DBCNN model."""

    def __init__(self,
                 use_bn=True,
                 default_mean=[0.485, 0.456, 0.406],
                 default_std=[0.229, 0.224, 0.225]):
        super().__init__()

        # VGG16 encoder
        self.features1 = torchvision.models.vgg16(weights=None).features
        self.features1 = nn.Sequential(*self.features1[:-1])

        # SCNN branch
        scnn = SCNN(use_bn=use_bn)
        self.features2 = scnn.features

        # Bilinear FC
        self.fc = nn.Linear(512 * 128, 1)

        self.default_mean = torch.tensor(default_mean, device="cpu").view(1, 3, 1, 1)
        self.default_std = torch.tensor(default_std, device="cpu").view(1, 3, 1, 1)

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


class FSIQA(nn.Module):
    def __init__(self, backbone, ftn, scorer):
        super().__init__()
        self.backbone = backbone
        self.ftn = ftn
        self.scorer = scorer

    def features(self, image):
        return self.ftn(self.backbone(image))

    def forward(self, image):
        return self.scorer(self.features(image))


def load_state(path):
    """Read tensor-only checkpoints, accepting common state-dict wrappers."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    for name in ("params", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and name in state:
            state = state[name]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Not a state dictionary: {path}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model(backbone_path, ftn_path, scorer_path, device="cpu"):
    """Load full DBCNN plus trained FS-IQA heads, sharing a single backbone.

    Meta initialization avoids allocating random multi-gigabyte FTN weights
    before loading the trained state. A full DBCNN checkpoint includes SCNN.
    """
    for path in (backbone_path, ftn_path, scorer_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}. See checkpoints/README.md")
    modules = []
    for cls, path in ((DBCNNf, backbone_path), (DBCNNprev_fc, ftn_path),
                      (DBCNNfc, scorer_path)):
        with torch.device("meta"):
            module = cls()
        module.load_state_dict(load_state(path), strict=True, assign=True)
        module = module.to(device).eval()
        module.requires_grad_(False)
        modules.append(module)
    return FSIQA(*modules).eval()
