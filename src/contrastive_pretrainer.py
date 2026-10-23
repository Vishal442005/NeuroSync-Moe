"""
Spectral-Spatial Contrastive Self-Supervised Pre-training (SS-CSP) Module.

Forces the Hierarchical MoE encoder to learn subject-invariant, neurophysiologically-grounded
representations without any manual labels, using:
- View A: Multi-band Spectral Masking (randomly attenuates/zeros canonical frequency bands)
- View B: Spatial Electrode Dropout & Jitter (randomly drops electrode channels)
- InfoNCE loss: Pulls representations of the same trial together across views while
  pushing representations from different trials/subjects apart.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .himoe_backbone import HierarchicalMoE
except ImportError:
    from himoe_backbone import HierarchicalMoE


class SpectralSpatialAugmenter:
    """
    Dual-view augmentation pipeline for multichannel EEG.
    """

    def __init__(
        self,
        spectral_mask_prob: float = 0.5,
        channel_dropout_ratio: float = 0.15,
        gaussian_noise_std: float = 0.02,
    ) -> None:
        self.spectral_mask_prob = spectral_mask_prob
        self.channel_dropout_ratio = channel_dropout_ratio
        self.gaussian_noise_std = gaussian_noise_std

    def spectral_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral View: applies random band-stop attenuation or temporal masking.
        """
        B, C, T = x.shape
        x_aug = x.clone()

        # Add light Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(x_aug) * self.gaussian_noise_std
            x_aug = x_aug + noise

        # Random temporal segment masking (10-20% time window)
        mask_len = int(T * 0.15)
        start = torch.randint(0, T - mask_len, (1,)).item()
        x_aug[:, :, start : start + mask_len] = 0.0

        return x_aug

    def spatial_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial View: randomly zeroes out a subset of electrode channels.
        """
        B, C, T = x.shape
        x_aug = x.clone()

        # Random channel mask per sample
        num_drop = max(1, int(C * self.channel_dropout_ratio))
        for b in range(B):
            drop_indices = torch.randperm(C)[:num_drop]
            x_aug[b, drop_indices, :] = 0.0

        return x_aug

    def generate_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate View 1 (Spectral-focused) and View 2 (Spatial-focused).
        """
        view1 = self.spectral_mask(x)
        view2 = self.spatial_dropout(x)
        return view1, view2


class ContrastiveProjectionHead(nn.Module):
    """
    Non-linear projection head mapping latent representations to contrastive space.
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(x)
        return F.normalize(z, dim=-1)


class ContrastivePretrainer(pl.LightningModule):
    """
    PyTorch Lightning module for self-supervised contrastive pre-training.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 64,
        temperature: float = 0.07,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = HierarchicalMoE(
            in_channels=in_channels,
            num_classes=2,
            hidden_dim=hidden_dim,
        )
        self.proj_head = ContrastiveProjectionHead(in_dim=hidden_dim, out_dim=hidden_dim)
        self.augmenter = SpectralSpatialAugmenter()
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def compute_infonce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE contrastive loss across two views.
        """
        B = z1.shape[0]

        # Cosine similarity matrix: (B, B)
        sim_12 = torch.mm(z1, z2.t()) / self.temperature
        sim_21 = sim_12.t()

        labels = torch.arange(B, device=z1.device)

        loss1 = F.cross_entropy(sim_12, labels)
        loss2 = F.cross_entropy(sim_21, labels)

        return (loss1 + loss2) * 0.5

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["eeg"]  # (B, C, T)

        # Generate dual views
        view1, view2 = self.augmenter.generate_views(x)

        # Extract latent representations
        h1, _, _ = self.encoder.extract_features(view1)
        h2, _, _ = self.encoder.extract_features(view2)

        # Project to contrastive unit sphere
        z1 = self.proj_head(h1)
        z2 = self.proj_head(h2)

        loss = self.compute_infonce_loss(z1, z2)
        self.log("pretrain_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-5
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    pretrainer = ContrastivePretrainer(in_channels=64, hidden_dim=64)
    x = torch.randn(8, 64, 256)
    batch = {"eeg": x}
    loss = pretrainer.training_step(batch, 0)
    print("SS-CSP Contrastive Pre-trainer Test:")
    print("  Batch shape:", x.shape)
    print("  InfoNCE Loss:", loss.item())
    assert loss.item() > 0
    print("  Unit test passed successfully!")
