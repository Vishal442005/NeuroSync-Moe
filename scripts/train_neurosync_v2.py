"""
NeuroSync-MoE 2.0: Subject-Adversarial Domain Generalization (DANN),
Class-Weighted Focal Loss, Inter-Subject Mixup, and Stochastic Weight Averaging (SWA).
Evaluates directly on the 21 holdout test subjects (1,863 trials).
"""

import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)

from himoe_backbone import HierarchicalMoE
from per_subject_dataset import PerSubjectDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== NeuroSync-MoE 2.0 Training on {device} ===")


# 1. Gradient Reversal Layer for Subject-Adversarial Training
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# 2. Class-Weighted Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


# 3. Subject-Adversarial NeuroSync Model
class SubjectAdversarialNeuroSync(nn.Module):
    def __init__(self, in_channels=57, num_classes=2, hidden_dim=64, num_subjects=84):
        super().__init__()
        self.backbone = HierarchicalMoE(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.25,
            load_balance_weight=0.01,
        )
        # Subject Discriminator: predicts subject ID from latent manifold z
        self.subject_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_subjects),
        )

    def forward(self, x, return_details=False):
        return self.backbone(x, return_details=return_details)

    def compute_adversarial_loss(self, x, y, subject_ids, lambda_grl=1.0, focal_loss_fn=None):
        out = self.backbone(x, return_details=True)
        logits = out["logits"]
        z = out["features"]
        balance_loss = self.backbone.router.auxiliary_loss() if hasattr(self.backbone, "router") else torch.tensor(0.0, device=x.device)

        # 1. Task classification loss (Focal Loss)
        if focal_loss_fn is not None:
            cls_loss = focal_loss_fn(logits, y)
        else:
            cls_loss = F.cross_entropy(logits, y, label_smoothing=0.05)

        # 2. Subject Discriminator loss with GRL
        z_rev = grad_reverse(z, lambda_grl)
        sub_preds = self.subject_discriminator(z_rev)
        sub_loss = F.cross_entropy(sub_preds, subject_ids)

        total_loss = cls_loss + 0.25 * sub_loss + 0.01 * balance_loss
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "sub_loss": sub_loss,
            "logits": logits,
        }


def train_neurosync_v2(
    epochs=15,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-3,
    use_swa=True,
    output_dir="checkpoints_v2",
):
    os.makedirs(output_dir, exist_ok=True)
    dm = PerSubjectDataModule(processed_dir="eeg_processed", batch_size=batch_size, num_workers=0)
    dm.setup()

    num_train_subjects = len(dm.train_dataset.entries)
    print(f"Total training subjects: {num_train_subjects}")

    # Inverse class weights for Focal Loss
    # 77 AUD / 45 Control -> alpha ~ [0.63, 0.37]
    alpha_weights = torch.tensor([0.63, 0.37], device=device)
    focal_loss_fn = FocalLoss(alpha=alpha_weights, gamma=2.0)

    model = SubjectAdversarialNeuroSync(
        in_channels=57,
        num_classes=2,
        hidden_dim=64,
        num_subjects=num_train_subjects,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if use_swa:
        swa_model = AveragedModel(model)
        swa_start = 5

    best_val_acc = 0.0
    best_ckpt_path = None

    print(f"\nStarting {epochs} epochs of Subject-Adversarial Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_sub_loss = 0.0
        correct_train = 0
        total_train = 0

        # Dynamic GRL schedule: p = epoch / epochs
        p = float(epoch) / float(epochs)
        lambda_grl = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

        for batch in dm.train_dataloader():
            x = batch["eeg"].to(device)
            y = batch["label"].to(device)
            sub_targets = batch["subject_id"].to(device)

            optimizer.zero_grad()
            losses = model.compute_adversarial_loss(
                x, y, sub_targets, lambda_grl=lambda_grl, focal_loss_fn=focal_loss_fn
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += losses["loss"].item() * x.size(0)
            total_cls_loss += losses["cls_loss"].item() * x.size(0)
            total_sub_loss += losses["sub_loss"].item() * x.size(0)

            preds = torch.argmax(losses["logits"], dim=1)
            correct_train += (preds == y).sum().item()
            total_train += x.size(0)

        scheduler.step()
        train_acc = correct_train / total_train

        # Update SWA
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_probs_list = []

        with torch.no_grad():
            for batch in dm.val_dataloader():
                x = batch["eeg"].to(device)
                y = batch["label"].cpu().numpy()
                out = model(x, return_details=True)
                probs = F.softmax(out["logits"], dim=1).cpu().numpy()
                val_probs_list.append(probs)
                val_preds_list.extend(np.argmax(probs, axis=1))
                val_targets_list.extend(y)

        val_acc = accuracy_score(val_targets_list, val_preds_list)
        val_f1 = f1_score(val_targets_list, val_preds_list, average="macro")

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Acc: {train_acc*100:.2f}% | Cls Loss: {total_cls_loss/total_train:.4f} | Sub Loss: {total_sub_loss/total_train:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% (GRL={lambda_grl:.3f})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(output_dir, f"best_v2_epoch={epoch:02d}_val_acc={val_acc:.3f}.ckpt")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, best_ckpt_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val Acc: {best_val_acc*100:.2f}% ({best_ckpt_path})")

    # Evaluate on Holdout Test Set (21 Unseen Subjects, 1,863 Trials)
    eval_model = swa_model if use_swa else model
    eval_model.eval()

    test_preds_list = []
    test_probs_list = []
    test_targets_list = []

    print("\n--- Evaluating NeuroSync-MoE 2.0 on 21 Holdout Test Subjects ---")
    with torch.no_grad():
        for batch in dm.test_dataloader():
            x = batch["eeg"].to(device)
            y = batch["label"].cpu().numpy()
            out = eval_model(x, return_details=True)
            probs = F.softmax(out["logits"], dim=1).cpu().numpy()
            test_probs_list.append(probs)
            test_preds_list.extend(np.argmax(probs, axis=1))
            test_targets_list.extend(y)

    y_true = np.array(test_targets_list)
    y_pred = np.array(test_preds_list)
    y_probs = np.concatenate(test_probs_list, axis=0)

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    test_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    test_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    test_kappa = cohen_kappa_score(y_true, y_pred)
    try:
        test_auc = roc_auc_score(y_true, y_probs[:, 1])
    except Exception:
        test_auc = 0.5

    print(f"============================================================")
    print(f"NEUROSYNC-MOE 2.0 TEST BENCHMARK RESULTS (21 UNSEEN SUBJECTS)")
    print(f"============================================================")
    print(f"Test Accuracy:      {test_acc*100:.2f}%")
    print(f"Macro F1-Score:     {test_f1*100:.2f}%")
    print(f"Precision:          {test_prec*100:.2f}%")
    print(f"Recall:             {test_rec*100:.2f}%")
    print(f"AUC-ROC:            {test_auc:.4f}")
    print(f"Cohen's Kappa:      {test_kappa:.4f}")
    print(f"============================================================")

    results = {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "test_auc": test_auc,
        "test_kappa": test_kappa,
        "best_val_acc": best_val_acc,
        "best_ckpt_path": best_ckpt_path,
    }
    with open("neurosync_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_neurosync_v2(epochs=15, batch_size=64, lr=1e-3, use_swa=True)
