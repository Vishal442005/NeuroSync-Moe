#!/usr/bin/env python3
"""
Spectral-Spatial Contrastive Self-Supervised Pre-training (SS-CSP) Entrypoint.

Trains the Hierarchical MoE encoder without class labels using:
- View 1: Multi-band Spectral Masking (random attenuation of canonical EEG bands)
- View 2: Spatial Electrode Dropout and Gaussian Jitter
- InfoNCE Contrastive Loss to maximize mutual information across subject representations
"""

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from src.contrastive_pretrainer import ContrastivePretrainer
from src.himoe_backbone import HierarchicalMoE
from src.per_subject_dataset import PerSubjectDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SS-CSP-Pretrain")


def main():
    parser = argparse.ArgumentParser(description="NeuroSync-MoE SS-CSP Pretraining")
    parser.add_argument("--data_dir", type=str, default="eeg_processed", help="Path to processed UCI .pt files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for pretraining")
    parser.add_argument("--epochs", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for AdamW")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden feature dimension")
    parser.add_argument("--output_ckpt", type=str, default="checkpoints/pretrained_encoder.ckpt", help="Output checkpoint path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} | Pretraining epochs: {args.epochs}")

    # Initialize DataModule
    dm = PerSubjectDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=0)
    dm.setup("fit")

    # Initialize Base MoE Encoder
    encoder = HierarchicalMoE(
        in_channels=57,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        dropout=0.25,
    )

    # Initialize Contrastive Pretrainer
    pretrainer = ContrastivePretrainer(
        encoder=encoder,
        feat_dim=args.hidden_dim,
        proj_dim=128,
        temperature=args.temperature,
        lr=args.lr,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=Path(args.output_ckpt).parent,
        filename="pretrained_encoder_{epoch:02d}_{train_loss:.3f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )

    logger.info("Starting SS-CSP Contrastive Pretraining...")
    trainer.fit(pretrainer, dm.train_dataloader())
    logger.info(f"Pretraining complete. Checkpoint saved to: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
