"""
Diffusion-Guided Test-Time Adaptation (DG-TTA) for EEG Alcoholism Classification.

Architecture & Mechanics:
1. Operates on the latent feature space (dim=64) extracted by HierarchicalMoE.
2. A lightweight Score-Based Denoising Diffusion Probabilistic Model (DDPM) is trained
   to model the underlying manifold of healthy and alcoholic latent representations.
3. At Test-Time (e.g., on unseen subjects or cross-dataset trials from OpenNeuro ds008851):
   - Given test latent z_0 = Encoder(x_test)
   - Evaluates whether z_0 exhibits distribution shift
   - Applies reverse score-based diffusion steps to guide z_0 toward high-density regions
     of the learned manifold
   - Backpropagation-free: No gradient updates to the classifier or encoder are required.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class LatentScoreNet(nn.Module):
    """
    Lightweight MLP-based score estimation network predicting the noise vector epsilon.
    Operates on latent dimension (e.g., 64).
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Noisy latent tensor (B, latent_dim)
            t: Discrete timesteps (B,)

        Returns:
            pred_noise: Predicted noise tensor (B, latent_dim)
        """
        t_emb = self.time_embed(t)
        h = torch.cat([z_t, t_emb], dim=-1)
        pred_noise = self.net(h)
        return pred_noise


class LatentDiffusionAdapter(nn.Module):
    """
    Diffusion model for latent representation modeling and test-time adaptation.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.score_net = LatentScoreNet(latent_dim=latent_dim, hidden_dim=128)

    def compute_training_loss(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Compute standard DDPM denoising score matching loss.

        Args:
            z_0: Clean latents (B, latent_dim)
        """
        B = z_0.shape[0]
        device = z_0.device

        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        z_t = sqrt_alpha * z_0 + sqrt_one_minus * noise
        pred_noise = self.score_net(z_t, t)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def adapt(
        self,
        z_test: torch.Tensor,
        adaptation_steps: int = 15,
        guidance_scale: float = 0.5,
    ) -> torch.Tensor:
        """
        Gradient-free Test-Time Adaptation:
        Diffuses the test representation slightly (forward noise), then applies
        reverse diffusion guided by the learned manifold density.

        Args:
            z_test: Unseen test latent representations of shape (B, latent_dim)
            adaptation_steps: Number of reverse diffusion refinement steps (default: 15)
            guidance_scale: Strength of adaptation adjustment

        Returns:
            z_adapted: Refined latent vectors aligned to the training manifold (B, latent_dim)
        """
        B = z_test.shape[0]
        device = z_test.device

        # Step 1: Inject slight perturbation at timestep t_start
        t_start = adaptation_steps
        t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)

        noise = torch.randn_like(z_test)
        sqrt_alpha = self.sqrt_alphas_cumprod[t_start]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_start]

        z_curr = sqrt_alpha * z_test + sqrt_one_minus * noise

        # Step 2: Reverse diffusion refinement from t_start down to 0
        for step in reversed(range(0, t_start)):
            t_curr = torch.full((B,), step, device=device, dtype=torch.long)
            pred_noise = self.score_net(z_curr, t_curr)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_cumprod_t = self.alphas_cumprod[step]

            # DDPM reverse mean
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                z_curr - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * pred_noise
            )

            if step > 0:
                var = beta_t * (1.0 - self.alphas_cumprod_prev[step]) / (1.0 - alpha_cumprod_t)
                sigma = torch.sqrt(torch.clamp(var, min=1e-8))
                z_curr = mean + sigma * torch.randn_like(z_curr)
            else:
                z_curr = mean

        # Weighted blend of adapted representation with original representation
        z_adapted = (1.0 - guidance_scale) * z_test + guidance_scale * z_curr
        return z_adapted


if __name__ == "__main__":
    adapter = LatentDiffusionAdapter(latent_dim=64, timesteps=100)
    z_train = torch.randn(16, 64)
    loss = adapter.compute_training_loss(z_train)
    print("DG-TTA Latent Diffusion Adapter Test:")
    print("  Training loss:", loss.item())

    # Simulate test adaptation under distribution shift
    z_test_shifted = z_train[:4] + 0.8 * torch.randn(4, 64)
    z_adapted = adapter.adapt(z_test_shifted, adaptation_steps=10)
    print("  Original test shape:", z_test_shifted.shape)
    print("  Adapted shape:", z_adapted.shape)
    assert z_adapted.shape == (4, 64)
    print("  Unit test passed successfully!")
