"""
Differentiable Multi-Scale Spectral Tokenizer for EEG.

Decomposes raw multichannel EEG into 5 neurophysiologically-grounded canonical frequency bands:
- Delta (0.5 - 4.0 Hz): Pathological slow-wave activity
- Theta (4.0 - 8.0 Hz): Frontal theta power dysregulation in alcoholism & reward processing
- Alpha (8.0 - 13.0 Hz): Alpha asymmetry (cortical hypo-arousal and hemispheric imbalance)
- Beta (13.0 - 30.0 Hz): Cortical hyper-arousal and withdrawal dynamics
- Gamma (30.0 - 45.0 Hz): Local cortical synchronization and binding

Operates purely in PyTorch with CUDA acceleration via Depthwise 1D Convolutions initialized
with exact analytical bandpass FIR filters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

CANONICAL_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def design_fir_bandpass(
    low_cut: float,
    high_cut: float,
    sfreq: float,
    num_taps: int = 65,
) -> np.ndarray:
    """
    Design an FIR bandpass filter using Hamming window.
    """
    nyq = 0.5 * sfreq
    low = max(0.01, low_cut / nyq)
    high = min(0.99, high_cut / nyq)
    taps = signal.firwin(num_taps, [low, high], pass_zero=False, window="hamming")
    return taps.astype(np.float32)


class DifferentiableSpectralTokenizer(nn.Module):
    """
    Transforms raw EEG (B, C, T) into multi-scale spectral tokens (B, 5, C, T)
    using 5 parallel bandpass filter kernels.
    """

    def __init__(
        self,
        in_channels: int = 64,
        sfreq: float = 256.0,
        filter_length: int = 65,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.sfreq = sfreq
        self.filter_length = filter_length
        self.num_bands = len(CANONICAL_BANDS)
        self.padding = filter_length // 2

        # Design FIR filters for each canonical band
        filters_list = []
        for band_name, (low, high) in CANONICAL_BANDS.items():
            taps = design_fir_bandpass(low, high, sfreq, num_taps=filter_length)
            filters_list.append(taps)

        # Shape: (num_bands, 1, filter_length)
        filters_weight = np.stack(filters_list, axis=0)[:, np.newaxis, :]
        tensor_weight = torch.from_numpy(filters_weight).float()

        if learnable:
            self.filter_weights = nn.Parameter(tensor_weight)
        else:
            self.register_buffer("filter_weights", tensor_weight)

        # Batch normalization across spectral bands to stabilize variance
        self.spectral_bn = nn.BatchNorm2d(self.num_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decomposes input EEG into 5 frequency sub-bands.

        Args:
            x: Raw EEG tensor of shape (B, C, T)

        Returns:
            spectral_tokens: Sub-band tensor of shape (B, 5, C, T)
        """
        B, C, T = x.shape

        # Reshape to (B * C, 1, T) for 1D convolution
        x_flat = x.view(B * C, 1, T)

        # Convolve with each of the 5 band filters
        # filter_weights shape: (5, 1, filter_length)
        # filtered shape: (B * C, 5, T)
        weights = self.filter_weights.to(x.device)
        filtered = F.conv1d(
            x_flat,
            weights,
            padding=self.padding,
        )

        # Reshape to (B, C, 5, T) -> permute to (B, 5, C, T)
        filtered = filtered.view(B, C, self.num_bands, T).permute(0, 2, 1, 3)

        # Normalize per-band to handle power scaling differences (delta vs gamma)
        filtered = self.spectral_bn(filtered)

        return filtered

    def get_band_power(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute relative band power features for routing/gating.

        Args:
            x: Raw EEG tensor of shape (B, C, T)

        Returns:
            band_power: (B, 5, C) relative power in each band per channel
        """
        tokens = self.forward(x)  # (B, 5, C, T)
        # Power = mean squared amplitude over time
        power = torch.mean(tokens ** 2, dim=-1)  # (B, 5, C)
        # Relative power across bands
        total_power = torch.sum(power, dim=1, keepdim=True) + 1e-6
        rel_power = power / total_power
        return rel_power


if __name__ == "__main__":
    tokenizer = DifferentiableSpectralTokenizer(in_channels=64, sfreq=256.0)
    x = torch.randn(4, 64, 256)
    tokens = tokenizer(x)
    power = tokenizer.get_band_power(x)

    print("Tokenizer Test:")
    print("  Input shape:", x.shape)
    print("  Spectral tokens shape (B, 5, C, T):", tokens.shape)
    print("  Band power shape (B, 5, C):", power.shape)
    assert tokens.shape == (4, 5, 64, 256)
    print("  Unit test passed successfully!")
