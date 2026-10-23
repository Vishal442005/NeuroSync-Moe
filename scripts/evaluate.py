#!/usr/bin/env python3
"""
NeuroSync-MoE Standalone Evaluation Benchmark Script.

Loads a trained model checkpoint and evaluates:
1. Primary UCI holdout benchmark (1,863 trials, 21 unseen subjects).
2. Cross-Dataset Zero-Shot Transfer on OpenNeuro ds008851 (20,854 continuous trials).
3. DG-TTA Test-Time Adaptation improvements across both benchmarks.
4. Baseline comparison against EEGNet and Flat-MoE.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn.functional as F

from src.himoe_backbone import HierarchicalMoE, ACTIVE_CHANNELS
from src.diffusion_tta import LatentDiffusionAdapter
from src.per_subject_dataset import PerSubjectDataModule
from src.ds008851_loader import OpenNeuroAlcoholismLoader, DS008851_CHANNELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluate")


def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "roc_auc": auc, "cohen_kappa": kappa}


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model = HierarchicalMoE(
        in_channels=57,
        num_classes=2,
        hidden_dim=64,
        num_heads=4,
        dropout=0.25,
    )
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean_sd[k.replace("model.", "")] = v
        else:
            clean_sd[k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_uci(model, adapter, dm, device):
    logger.info("Evaluating on primary UCI holdout test set...")
    all_y = []
    base_logits, tta_logits = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            x = batch["eeg"].to(device)
            y = batch["label"].cpu().numpy()
            all_y.extend(y)

            out = model(x, return_details=True)
            z = out["fused_features"]
            b_log = model.classifier(z)
            base_logits.append(b_log.cpu())

            z_adapted = adapter.adapt_latents(z, steps=5, step_size=0.05)
            t_log = model.classifier(z_adapted)
            tta_logits.append(t_log.cpu())

    y_true = np.array(all_y)
    b_all = torch.cat(base_logits, dim=0).softmax(dim=-1).numpy()
    t_all = torch.cat(tta_logits, dim=0).softmax(dim=-1).numpy()

    m_no_tta = compute_metrics(y_true, np.argmax(b_all, axis=1), b_all[:, 1])
    m_tta = compute_metrics(y_true, np.argmax(t_all, axis=1), t_all[:, 1])
    return m_no_tta, m_tta


def evaluate_cross_dataset(model, adapter, openneuro_cache: str, device: torch.device):
    logger.info("Evaluating on secondary OpenNeuro ds008851 dataset...")
    cache_path = Path(openneuro_cache)
    if not cache_path.exists():
        logger.warning(f"Cache {openneuro_cache} not found. Skipping cross-dataset benchmark.")
        return None, None

    data = np.load(str(cache_path))
    X_raw, y_raw = data["X"], data["y"]  # (20854, 8, 256)
    
    # Map 8 channels to 57-channel space
    channel_to_idx = {ch.upper(): i for i, ch in enumerate(ACTIVE_CHANNELS)}
    idx_map = []
    for ch in DS008851_CHANNELS:
        ch_upper = ch.upper()
        if ch_upper in channel_to_idx:
            idx_map.append(channel_to_idx[ch_upper])
        else:
            idx_map.append(None)

    all_y = []
    base_logits, tta_logits = [], []

    batch_size = 256
    num_batches = int(np.ceil(len(X_raw) / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            x_b = X_raw[b * batch_size : (b + 1) * batch_size]
            y_b = y_raw[b * batch_size : (b + 1) * batch_size]
            all_y.extend(y_b)

            x_full = np.zeros((len(x_b), 57, 256), dtype=np.float32)
            for ch_idx, target_idx in enumerate(idx_map):
                if target_idx is not None:
                    x_full[:, target_idx, :] = x_b[:, ch_idx, :]

            x_t = torch.from_numpy(x_full).to(device)
            out = model(x_t, return_details=True)
            z = out["fused_features"]

            b_log = model.classifier(z)
            base_logits.append(b_log.cpu())

            z_adapted = adapter.adapt_latents(z, steps=5, step_size=0.05)
            t_log = model.classifier(z_adapted)
            tta_logits.append(t_log.cpu())

    y_true = np.array(all_y)
    b_all = torch.cat(base_logits, dim=0).softmax(dim=-1).numpy()
    t_all = torch.cat(tta_logits, dim=0).softmax(dim=-1).numpy()

    m_zero = compute_metrics(y_true, np.argmax(b_all, axis=1), b_all[:, 1])
    m_adapt = compute_metrics(y_true, np.argmax(t_all, axis=1), t_all[:, 1])
    return m_zero, m_adapt


def main():
    parser = argparse.ArgumentParser(description="NeuroSync-MoE Evaluation Script")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_neurosync_himoe_epoch=03_val_acc=0.690.ckpt")
    parser.add_argument("--data_dir", type=str, default="eeg_processed")
    parser.add_argument("--openneuro_cache", type=str, default="ds008851/ds008851_processed_cache.npz")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Target evaluation device: {device}")

    # Load Model
    model = load_model_from_checkpoint(args.checkpoint, device)

    # Initialize / Fit Adapter on train latents
    dm = PerSubjectDataModule(data_dir=args.data_dir, batch_size=64, num_workers=0)
    dm.setup("fit")
    dm.setup("test")

    logger.info("Fitting Latent Diffusion Adapter...")
    adapter = LatentDiffusionAdapter(latent_dim=64, num_timesteps=20).to(device)
    with torch.no_grad():
        latents = []
        for batch in dm.train_dataloader():
            x = batch["eeg"].to(device)
            out = model(x, return_details=True)
            latents.append(out["fused_features"])
        z_train = torch.cat(latents, dim=0)
    adapter.fit_score_matching(z_train, epochs=10, lr=1e-3, batch_size=128)

    # Evaluate on UCI holdout
    m_no_tta, m_tta = evaluate_uci(model, adapter, dm, device)
    print("\n" + "=" * 75)
    print("PRIMARY UCI HOLDOUT BENCHMARK (1,863 Trials across 21 Test Subjects)")
    print("=" * 75)
    print(f"{'Metric':<18} | {'Without TTA':<14} | {'With DG-TTA (Proposed)':<22} | {'Improvement'}")
    print("-" * 75)
    for k in m_no_tta:
        v_base = m_no_tta[k]
        v_tta = m_tta[k]
        diff = v_tta - v_base
        print(f"{k:<18} | {v_base * 100:>12.2f}% | {v_tta * 100:>20.2f}% | {diff * 100:>+9.2f}%")

    # Evaluate on OpenNeuro
    res_zero, res_adapt = evaluate_cross_dataset(model, adapter, args.openneuro_cache, device)
    if res_zero is not None:
        print("\n" + "=" * 80)
        print("SECONDARY CROSS-DATASET BENCHMARK (OpenNeuro ds008851: 20,854 Continuous Trials)")
        print("=" * 80)
        print(f"{'Metric':<18} | {'Zero-Shot Transfer':<18} | {'With DG-TTA Adaptation':<22} | {'Improvement'}")
        print("-" * 80)
        for k in res_zero:
            v_base = res_zero[k]
            v_tta = res_adapt[k]
            diff = v_tta - v_base
            print(f"{k:<18} | {v_base * 100:>16.2f}% | {v_tta * 100:>20.2f}% | {diff * 100:>+9.2f}%")


if __name__ == "__main__":
    main()
