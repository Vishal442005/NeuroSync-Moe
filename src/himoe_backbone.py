"""
Hierarchical Mixture of Experts (HiMoE) Backbone for EEG Alcoholism Classification.

Architecture:
1. Multi-Scale Spectral Tokenization (Delta, Theta, Alpha, Beta, Gamma)
2. Layer 1: 5 Spectral Specialists (one per canonical EEG frequency band)
3. Cross-Expert Attention Module: multi-head self-attention across spectral bands to
   model co-occurring AUD neurophysiological biomarkers (e.g., frontal theta elevation
   with parietal alpha asymmetry and P300 attenuation)
4. Layer 2: 4 Spatial Specialists:
   - Left-Hemisphere Specialist
   - Right-Hemisphere Specialist
   - Midline Specialist
   - Global Spatio-Temporal Coordinator
5. Hierarchical Gating Router with Load-Balancing Regularization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .spectral_tokenizer import DifferentiableSpectralTokenizer
except ImportError:
    from spectral_tokenizer import DifferentiableSpectralTokenizer

# Standard 64-channel montage
STANDARD_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8",
    "T9", "T10",
    "TP7", "TP8",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "IZ",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
]

# OpenNeuro 8-channel montage
DS008851_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "O1", "O2", "Fz", "Pz"]

# 7 dead channels in UCI montage: AF3, AF4, T9, T10, PO3, PO4, IZ
DEAD_CHANNEL_INDICES = [4, 6, 21, 22, 42, 44, 49]
ACTIVE_CHANNELS = [
    STANDARD_CHANNELS[i] for i in range(len(STANDARD_CHANNELS)) if i not in DEAD_CHANNEL_INDICES
]  # 57 active channels


class SpectralSpecialist(nn.Module):
    """
    Specialist sub-network dedicated to a single canonical frequency band.
    Processes (B, C, T) filtered for a specific frequency band.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        # Temporal convolution tuned for band-specific dynamics
        self.temp_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        # Spatial convolution across electrodes
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        # Pointwise projection
        self.proj = nn.Sequential(
            nn.Conv2d(32, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Band-filtered EEG of shape (B, C, T)

        Returns:
            h: Representation vector of shape (B, hidden_dim)
        """
        h = x.unsqueeze(1)  # (B, 1, C, T)
        h = self.temp_conv(h)
        h = self.spatial_conv(h)
        h = self.proj(h).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        return h


class SpatialSpecialist(nn.Module):
    """
    Regional spatial specialist processing a designated electrode sub-region.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # If empty region (fallback for reduced montage), output zero representation
        if in_channels == 0:
            self.net = None
            self.fallback = nn.Parameter(torch.zeros(hidden_dim))
            return

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 33), padding=(0, 16), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(32, hidden_dim, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 0 or x.shape[1] == 0:
            B = x.shape[0]
            return self.fallback.unsqueeze(0).expand(B, -1)
        h = x.unsqueeze(1)  # (B, 1, C, T)
        h = self.net(h).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        return h


class CrossExpertAttention(nn.Module):
    """
    Multi-Head Self-Attention across expert representations.
    Captures inter-band co-occurrence and spatial-spectral coupling.
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Expert representations of shape (B, num_experts, hidden_dim)
            return_attn: If True, returns (out, attn_weights)

        Returns:
            out: Contextualized expert representations (B, num_experts, hidden_dim)
            attn_weights: (Optional) Attention weights (B, num_heads, num_experts, num_experts)
        """
        attn_out, attn_weights = self.mha(x, x, x, need_weights=True)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        if return_attn:
            return out, attn_weights
        return out


class HierarchicalGating(nn.Module):
    """
    Dual-level routing network predicting spectral routing weights (g_spectral)
    and spatial routing weights (g_spatial).
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        # Features: channel variance (in_channels) + 5 band power ratios + laterality index (1)
        feature_dim = in_channels + 5 + 1

        self.router_spectral = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

        self.router_spatial = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
        band_power: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Raw EEG of shape (B, C, T)
            band_power: Relative power in 5 bands of shape (B, 5, C)

        Returns:
            gates_spectral: (B, 5)
            gates_spatial: (B, 4)
        """
        B, C, T = x.shape

        # Spatial variance
        var = torch.var(x, dim=-1)  # (B, C)

        # Global average band power across channels
        mean_band_power = torch.mean(band_power, dim=-1)  # (B, 5)

        # Laterality index: compare left vs right hemisphere variance
        half_c = C // 2
        var_left = torch.mean(var[:, :half_c], dim=-1, keepdim=True)
        var_right = torch.mean(var[:, half_c:], dim=-1, keepdim=True)
        lat_index = (var_left - var_right) / (var_left + var_right + 1e-6)

        features = torch.cat([var, mean_band_power, lat_index], dim=-1)

        logits_spectral = self.router_spectral(features)
        logits_spatial = self.router_spatial(features)

        gates_spectral = F.softmax(logits_spectral / self.temperature, dim=-1)
        gates_spatial = F.softmax(logits_spatial / self.temperature, dim=-1)

        return gates_spectral, gates_spatial


class HierarchicalMoE(nn.Module):
    """
    Complete NeuroSync-MoE Architecture.
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 2,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.25,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
        channel_names: Optional[List[str]] = None,
        sfreq: float = 256.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.load_balance_weight = load_balance_weight

        # 1. Multi-Scale Spectral Tokenizer
        self.tokenizer = DifferentiableSpectralTokenizer(
            in_channels=in_channels,
            sfreq=sfreq,
            learnable=True,
        )

        # Channel partitioning for Spatial Specialists
        ch_list = [c.upper() for c in (channel_names or STANDARD_CHANNELS[:in_channels])]
        ch_map = {name: idx for idx, name in enumerate(ch_list)}

        # Define canonical left, right, midline channel sets
        left_targets = ["FP1", "AF7", "AF3", "F7", "F5", "F3", "F1", "FT7", "T7", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1", "FC5", "FC3", "FC1", "C5", "C3", "C1"]
        right_targets = ["FP2", "AF8", "AF4", "F8", "F6", "F4", "F2", "FT8", "T8", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2", "FC6", "FC4", "FC2", "C6", "C4", "C2"]
        midline_targets = ["FPZ", "AFZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ", "IZ"]

        self.left_indices = [ch_map[c] for c in left_targets if c in ch_map]
        self.right_indices = [ch_map[c] for c in right_targets if c in ch_map]
        self.midline_indices = [ch_map[c] for c in midline_targets if c in ch_map]

        # 2. Layer 1: 5 Spectral Specialists (Delta, Theta, Alpha, Beta, Gamma)
        self.spectral_specialists = nn.ModuleList([
            SpectralSpecialist(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(5)
        ])

        # 3. Cross-Expert Attention Module
        self.spectral_attention = CrossExpertAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1
        )

        # 4. Layer 2: 4 Spatial Specialists
        self.spatial_left = SpatialSpecialist(in_channels=len(self.left_indices), hidden_dim=hidden_dim, dropout=dropout)
        self.spatial_right = SpatialSpecialist(in_channels=len(self.right_indices), hidden_dim=hidden_dim, dropout=dropout)
        self.spatial_midline = SpatialSpecialist(in_channels=len(self.midline_indices), hidden_dim=hidden_dim, dropout=dropout)
        self.spatial_global = SpatialSpecialist(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)

        self.spatial_specialists = nn.ModuleList([
            self.spatial_left,
            self.spatial_right,
            self.spatial_midline,
            self.spatial_global,
        ])

        # Cross-Spatial Attention
        self.spatial_attention = CrossExpertAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1
        )

        # 5. Hierarchical Gating Network
        self.gating = HierarchicalGating(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            temperature=temperature,
        )

        # Unified Fusion & Classification Head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Extract latent representations and routing gates without classification head.

        Args:
            x: (B, C, T)
            return_attn: If True, also return cross-spectral attention weights

        Returns:
            fused_latent: (B, hidden_dim)
            gates_spectral: (B, 5)
            gates_spatial: (B, 4)
            spectral_attn: (Optional) (B, heads, 5, 5)
        """
        B, C, T = x.shape

        # Step 1: Multi-scale spectral tokenization -> (B, 5, C, T)
        spectral_tokens = self.tokenizer(x)
        band_power = self.tokenizer.get_band_power(x)

        # Step 2: Layer 1 Spectral Specialists
        h_spec_list = []
        for i, specialist in enumerate(self.spectral_specialists):
            sub_band = spectral_tokens[:, i, :, :]  # (B, C, T)
            h_spec_list.append(specialist(sub_band))  # (B, hidden_dim)

        h_spectral_stack = torch.stack(h_spec_list, dim=1)  # (B, 5, hidden_dim)

        # Step 3: Cross-Expert Attention
        if return_attn:
            h_spectral_context, spectral_attn = self.spectral_attention(h_spectral_stack, return_attn=True)
        else:
            h_spectral_context = self.spectral_attention(h_spectral_stack)
            spectral_attn = None

        # Step 4: Layer 2 Spatial Specialists
        x_left = x[:, self.left_indices, :] if len(self.left_indices) > 0 else x[:, :0, :]
        x_right = x[:, self.right_indices, :] if len(self.right_indices) > 0 else x[:, :0, :]
        x_midline = x[:, self.midline_indices, :] if len(self.midline_indices) > 0 else x[:, :0, :]
        x_global = x

        h_spatial_left = self.spatial_left(x_left)
        h_spatial_right = self.spatial_right(x_right)
        h_spatial_midline = self.spatial_midline(x_midline)
        h_spatial_global = self.spatial_global(x_global)

        h_spatial_stack = torch.stack(
            [h_spatial_left, h_spatial_right, h_spatial_midline, h_spatial_global], dim=1
        )  # (B, 4, hidden_dim)

        h_spatial_context = self.spatial_attention(h_spatial_stack)  # (B, 4, hidden_dim)

        # Step 5: Hierarchical Gating
        gates_spectral, gates_spatial = self.gating(x, band_power)

        # Weighted combination
        fused_spectral = torch.sum(gates_spectral.unsqueeze(-1) * h_spectral_context, dim=1)  # (B, hidden_dim)
        fused_spatial = torch.sum(gates_spatial.unsqueeze(-1) * h_spatial_context, dim=1)  # (B, hidden_dim)

        # Combine spectral and spatial streams
        combined = torch.cat([fused_spectral, fused_spatial], dim=-1)  # (B, hidden_dim * 2)
        fused_latent = self.fusion(combined)  # (B, hidden_dim)

        if return_attn:
            return fused_latent, gates_spectral, gates_spatial, spectral_attn
        return fused_latent, gates_spectral, gates_spatial

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if not return_details:
            fused_latent, _, _ = self.extract_features(x)
            return self.classifier(fused_latent)

        fused_latent, gates_spectral, gates_spatial, spectral_attn = self.extract_features(x, return_attn=True)
        logits = self.classifier(fused_latent)

        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "latent": fused_latent,
            "features": fused_latent,
            "gates_spectral": gates_spectral,
            "gates_spatial": gates_spatial,
            "spectral_gate_weights": gates_spectral,
            "spatial_gate_weights": gates_spatial,
            "cross_spectral_attn": spectral_attn,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        label_smoothing: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss = Cross-Entropy + Spectral Load-Balance + Spatial Load-Balance.
        """
        out = self.forward(x, return_details=True)
        logits = out["logits"]
        g_spec = out["gates_spectral"]
        g_spat = out["gates_spatial"]

        # Classification loss
        cls_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        # Auxiliary load balancing loss: penalize deviation from uniform routing
        mean_spec = torch.mean(g_spec, dim=0)
        spec_balance = 5.0 * torch.sum(mean_spec ** 2)

        mean_spat = torch.mean(g_spat, dim=0)
        spat_balance = 4.0 * torch.sum(mean_spat ** 2)

        total_balance = spec_balance + spat_balance
        total_loss = cls_loss + self.load_balance_weight * total_balance

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "balance_loss": total_balance,
            "logits": logits,
            "gates_spectral": g_spec,
            "gates_spatial": g_spat,
        }


if __name__ == "__main__":
    model = HierarchicalMoE(in_channels=64, num_classes=2, hidden_dim=64)
    x = torch.randn(4, 64, 256)
    y = torch.randint(0, 2, (4,))

    out = model.compute_loss(x, y)
    print("Hierarchical MoE Test:")
    print("  Input shape:", x.shape)
    print("  Loss:", out["loss"].item())
    print("  Classification Loss:", out["cls_loss"].item())
    print("  Balance Loss:", out["balance_loss"].item())
    print("  Logits shape:", out["logits"].shape)
    print("  Spectral gates shape (B, 5):", out["gates_spectral"].shape)
    print("  Spatial gates shape (B, 4):", out["gates_spatial"].shape)

    # Test with 8-channel input (ds008851 OpenNeuro montage)
    model_8ch = HierarchicalMoE(
        in_channels=8, num_classes=2, hidden_dim=64, channel_names=DS008851_CHANNELS
    )
    x8 = torch.randn(4, 8, 256)
    out8 = model_8ch.compute_loss(x8, y)
    print("\n8-Channel Montage Test (OpenNeuro ds008851):")
    print("  Input shape:", x8.shape)
    print("  Loss:", out8["loss"].item())
    print("  All tests passed successfully!")
