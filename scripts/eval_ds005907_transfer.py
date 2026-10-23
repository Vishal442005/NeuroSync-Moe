"""
Evaluation of NeuroSync-MoE v2 on OpenNeuro ds005907 Pilot Cohort (8 Subjects, 8,379 Trials).
Applies standard per-trial z-score standardization, maps 38 shared channels into the
57-channel Active Montage, and correlates hierarchical MoE routing activations
(Frontal Theta, Parietal Alpha) against clinical AUDIT scores.
"""

import os
import sys
import json
import numpy as np
import scipy.stats as stats
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)

from himoe_backbone import HierarchicalMoE, ACTIVE_CHANNELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Running ds005907 Cross-Cohort Evaluation on {device} ===")

CACHE_FILE = Path("ds005907/ds005907_processed_cache.npz")
CKPT_FILE = Path("checkpoints_v2/best_v2_epoch=01_val_acc=0.683.ckpt")
OUT_JSON = Path("Results/ds005907_transfer_results.json")
OUT_JSON.parent.mkdir(exist_ok=True)


def evaluate_transfer():
    if not CACHE_FILE.exists():
        print(f"Error: Cache file {CACHE_FILE} not found yet!")
        return

    print(f"Loading cached dataset from {CACHE_FILE}...")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    raw_data = cache["data"]  # (N, C_shared, 256)
    labels = cache["labels"]  # (N,)
    subject_ids = cache["subject_ids"]  # (N,)
    audit_scores = cache["audit_scores"]  # (N,)
    channels = cache["channels"].tolist()  # list of str

    N, C_shared, T = raw_data.shape
    print(f"Dataset Loaded: {N} trials across {C_shared} channels, {T} time samples.")
    print(f"Class distribution: {np.bincount(labels)} (0=Control, 1=AUD)")
    unique_subs = np.unique(subject_ids)
    print(f"Unique subjects ({len(unique_subs)}): {list(unique_subs)}")

    # 1. Apply per-trial per-channel z-score standardization (consistent with UCI training)
    print("Applying per-trial z-score normalization...")
    mean = raw_data.mean(axis=-1, keepdims=True)
    std = raw_data.std(axis=-1, keepdims=True) + 1e-6
    norm_data = (raw_data - mean) / std

    # 2. Map shared channels into 57-channel UCI montage
    print(f"Aligning {C_shared} channels into standard 57-channel Active Montage...")
    mapped_data = np.zeros((N, len(ACTIVE_CHANNELS), T), dtype=np.float32)
    matched_count = 0
    for idx, ch in enumerate(channels):
        if ch in ACTIVE_CHANNELS:
            uci_idx = ACTIVE_CHANNELS.index(ch)
            mapped_data[:, uci_idx, :] = norm_data[:, idx, :]
            matched_count += 1
    print(f"Successfully mapped {matched_count}/{C_shared} channels into 57-channel space.")

    # 3. Load NeuroSync-MoE v2
    print(f"Loading checkpoint {CKPT_FILE}...")
    ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    backbone_dict = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
    if not backbone_dict:
        backbone_dict = sd

    model = HierarchicalMoE(in_channels=len(ACTIVE_CHANNELS), num_classes=2, hidden_dim=64)
    model.load_state_dict(backbone_dict, strict=True)
    model.to(device)
    model.eval()

    # 4. Batched inference
    batch_size = 128
    all_preds = []
    all_probs = []
    all_gates_spectral = []
    all_gates_spatial = []

    print("Running batched inference...")
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = torch.tensor(mapped_data[i : i + batch_size], device=device)
            out = model(batch_x, return_details=True)
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())
            all_gates_spectral.append(out["gates_spectral"].cpu().numpy())
            all_gates_spatial.append(out["gates_spatial"].cpu().numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    gates_spectral = np.concatenate(all_gates_spectral, axis=0)  # (N, 5): Delta, Theta, Alpha, Beta, Gamma
    gates_spatial = np.concatenate(all_gates_spatial, axis=0)    # (N, 4): Left, Right, Midline, Global

    # Calculate metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro")
    kappa = cohen_kappa_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(labels, preds)

    print("\n" + "=" * 60)
    print("=== OPENNEURO ds005907 TRANSFER BENCHMARK RESULTS ===")
    print("=" * 60)
    print(f"Transfer Accuracy:       {acc * 100:.2f}%")
    print(f"Macro F1-Score:          {f1 * 100:.2f}%")
    print(f"Precision:               {prec * 100:.2f}%")
    print(f"Recall:                  {rec * 100:.2f}%")
    print(f"Cohen's Kappa (kappa):   {kappa:.4f}")
    print(f"AUC-ROC:                 {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Subject-level analysis & AUDIT score correlation
    print("\n" + "=" * 60)
    print("=== SUBJECT-LEVEL AUDIT CORRELATION ANALYSIS ===")
    print("=" * 60)
    sub_audit_list = []
    sub_theta_list = []
    sub_alpha_list = []
    sub_delta_list = []
    sub_gamma_list = []
    sub_midline_list = []
    sub_acc_list = []

    print(f"{'Subject':<10} {'Group':<6} {'AUDIT':<8} {'Trials':<8} {'Acc (%)':<10} {'Theta Gate':<12} {'Alpha Gate':<12}")
    print("-" * 72)
    for sub in unique_subs:
        mask = (subject_ids == sub)
        sub_audit = float(audit_scores[mask][0])
        sub_acc = accuracy_score(labels[mask], preds[mask]) * 100
        mean_delta = float(np.mean(gates_spectral[mask, 0]))  # Delta is index 0
        mean_theta = float(np.mean(gates_spectral[mask, 1]))  # Theta is index 1
        mean_alpha = float(np.mean(gates_spectral[mask, 2]))  # Alpha is index 2
        mean_gamma = float(np.mean(gates_spectral[mask, 4]))  # Gamma is index 4
        mean_midline = float(np.mean(gates_spatial[mask, 2])) # Midline is index 2
        group = "AUD" if labels[mask][0] == 1 else "CTRL"

        sub_audit_list.append(sub_audit)
        sub_delta_list.append(mean_delta)
        sub_theta_list.append(mean_theta)
        sub_alpha_list.append(mean_alpha)
        sub_gamma_list.append(mean_gamma)
        sub_midline_list.append(mean_midline)
        sub_acc_list.append(sub_acc)

        print(f"{sub:<10} {group:<6} {sub_audit:<8.1f} {np.sum(mask):<8} {sub_acc:<10.2f} {mean_theta:<12.4f} {mean_alpha:<12.4f}")

    # Compute Pearson & Spearman correlations
    r_theta, p_theta = stats.pearsonr(sub_audit_list, sub_theta_list)
    rho_theta, sp_theta = stats.spearmanr(sub_audit_list, sub_theta_list)

    r_alpha, p_alpha = stats.pearsonr(sub_audit_list, sub_alpha_list)
    rho_alpha, sp_alpha = stats.spearmanr(sub_audit_list, sub_alpha_list)

    r_gamma, p_gamma = stats.pearsonr(sub_audit_list, sub_gamma_list)
    rho_gamma, sp_gamma = stats.spearmanr(sub_audit_list, sub_gamma_list)

    print("-" * 72)
    print(f"Correlation: Frontal Theta Gate vs. AUDIT Severity:")
    print(f"  Pearson r    = {r_theta:+.4f} (p = {p_theta:.4e})")
    print(f"  Spearman rho = {rho_theta:+.4f} (p = {sp_theta:.4e})")
    print(f"Correlation: Parietal Alpha Gate vs. AUDIT Severity:")
    print(f"  Pearson r    = {r_alpha:+.4f} (p = {p_alpha:.4e})")
    print(f"  Spearman rho = {rho_alpha:+.4f} (p = {sp_alpha:.4e})")
    print(f"Correlation: Cortical Gamma Gate vs. AUDIT Severity:")
    print(f"  Pearson r    = {r_gamma:+.4f} (p = {p_gamma:.4e})")
    print(f"  Spearman rho = {rho_gamma:+.4f} (p = {sp_gamma:.4e})")
    print("=" * 60)

    # Save results to JSON
    results = {
        "dataset": "OpenNeuro ds005907 (RL Three-Armed Bandit Task)",
        "cohort_size": int(len(unique_subs)),
        "total_trials": int(N),
        "shared_channels": int(C_shared),
        "transfer_metrics": {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "cohen_kappa": float(kappa),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
        },
        "audit_correlation": {
            "frontal_theta_pearson_r": float(r_theta),
            "frontal_theta_pearson_p": float(p_theta),
            "frontal_theta_spearman_rho": float(rho_theta),
            "frontal_theta_spearman_p": float(sp_theta),
            "parietal_alpha_pearson_r": float(r_alpha),
            "parietal_alpha_pearson_p": float(p_alpha),
            "parietal_alpha_spearman_rho": float(rho_alpha),
            "parietal_alpha_spearman_p": float(sp_alpha),
            "cortical_gamma_pearson_r": float(r_gamma),
            "cortical_gamma_pearson_p": float(p_gamma),
            "cortical_gamma_spearman_rho": float(rho_gamma),
            "cortical_gamma_spearman_p": float(sp_gamma),
        },
        "per_subject_breakdown": {
            sub: {
                "audit": sub_audit_list[idx],
                "accuracy": sub_acc_list[idx],
                "theta_gate": sub_theta_list[idx],
                "alpha_gate": sub_alpha_list[idx],
                "gamma_gate": sub_gamma_list[idx],
            }
            for idx, sub in enumerate(unique_subs)
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved evaluation metrics to {OUT_JSON}")

    # Mirror to repository
    repo_json = Path("repository/results/metrics/ds005907_transfer_results.json")
    repo_json.parent.mkdir(parents=True, exist_ok=True)
    with open(repo_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Mirrored to {repo_json}")


if __name__ == "__main__":
    evaluate_transfer()
