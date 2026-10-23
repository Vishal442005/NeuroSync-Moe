"""
Publication Baseline Architectures for EEG Alcoholism Classification
Includes:
1. EEGNet (Lawhern et al. 2018): Temporal conv + Depthwise spatial conv + Separable conv.
2. ShallowFBCSPNet (Schirrmeister et al. 2017): Band-pass filterbank + spatial filter + log-energy.
3. FlatMoE: Standard single-layer MoE without hierarchical spectral-spatial decomposition (Ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm weight constraint along kernel axes."""
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """
    Standard EEGNet architecture (Lawhern et al., 2018)
    Input shape: (B, C, T) -> reshaped to (B, 1, C, T)
    """
    def __init__(
        self,
        in_channels: int = 57,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dWithConstraint(F1, F1 * D, (in_channels, 1), groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Classifier
        # Input T=256 -> pool1 (/4) -> 64 -> pool2 (/8) -> 8
        self.fc = nn.Linear(F2 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1) # (B, 1, C, T)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten & Classify
        x = x.flatten(start_dim=1)
        logits = self.fc(x)
        return logits


class ShallowFBCSPNet(nn.Module):
    """
    Shallow Filter Bank Convolutional Network (Schirrmeister et al., 2017).
    Tailored for oscillatory EEG feature decoding via temporal convolution,
    spatial filter convolution, square activation, and log average pooling.
    Input shape: (B, C, T) -> reshaped to (B, 1, C, T)
    """
    def __init__(
        self,
        in_channels: int = 57,
        num_classes: int = 2,
        time_points: int = 256,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=True)
        self.conv_spat = Conv2dWithConstraint(n_filters_time, n_filters_spat, (in_channels, 1), bias=False, max_norm=2.0)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout = nn.Dropout(p=drop_prob)

        # Compute output dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, time_points)
            out = self._forward_features(dummy)
            feat_dim = out.shape[1] * out.shape[2] * out.shape[3]

        self.fc = nn.Linear(feat_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self._forward_features(x)
        logits = self.fc(feat.flatten(start_dim=1))
        return logits


class FlatMoE(nn.Module):
    """
    Standard Flat Mixture of Experts baseline (without hierarchical decomposition).
    Uses 4 identical generic 1D CNN experts with a single softmax router.
    """
    def __init__(self, in_channels: int = 57, num_classes: int = 2, num_experts: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.num_experts = num_experts
        
        # Gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1),
        )

        # Generic 1D CNN Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            for _ in range(num_experts)
        ])

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gating
        gates = self.gate(x) # (B, num_experts)

        # Expert outputs
        expert_outputs = [expert(x) for expert in self.experts] # [(B, hidden_dim), ...]
        stacked_experts = torch.stack(expert_outputs, dim=1) # (B, num_experts, hidden_dim)

        # Weighted combination
        weights = gates.unsqueeze(-1) # (B, num_experts, 1)
        pooled = (stacked_experts * weights).sum(dim=1) # (B, hidden_dim)

        logits = self.classifier(pooled)
        return logits
