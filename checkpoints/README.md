# Model Checkpoints

This directory contains trained checkpoints for the NeuroSync-MoE framework.

## Checkpoint Registry

| Checkpoint Name | Validation Accuracy | Epoch | Number of Parameters | Description |
| :--- | :---: | :---: | :---: | :--- |
| `best_neurosync_himoe_epoch=03_val_acc=0.690.ckpt` | **69.02%** | 3 | 41,858 params | Best checkpoint from Stage 4 Supervised Fine-Tuning with SS-CSP pretraining initialization. |

## Checkpoint Details

- **Input Dimension:** `(B, 57, 256)` (57 active 10-20 EEG electrodes sampled at 256 Hz)
- **Spectral Decomposition:** Sinc-based differentiable filterbank across 5 canonical bands:
  - $\delta$ (1–4 Hz), $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma$ (30–50 Hz)
- **Latent Dimension:** $d = 64$
- **Spatial Specialists:** Left-Hemisphere, Right-Hemisphere, Midline, and Global Coordinator
- **Compatibility:** Can be directly loaded via `scripts/evaluate.py` or PyTorch Lightning:

```python
import torch
from src.himoe_backbone import HierarchicalMoE

ckpt = torch.load("checkpoints/best_neurosync_himoe_epoch=03_val_acc=0.690.ckpt", map_location="cpu")
model = HierarchicalMoE(in_channels=57, num_classes=2, hidden_dim=64, num_heads=4)
state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
model.load_state_dict(state_dict, strict=False)
model.eval()
```
