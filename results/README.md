# Publication Results Package: NeuroSync-MoE

This directory contains the complete camera-ready publication figures, LaTeX tables, JSON benchmark evaluations, and manuscript source files for Q1 journal submission (*IEEE Transactions on Neural Systems and Rehabilitation Engineering* / *Elsevier Computers in Biology and Medicine*).

---

## 1. Publication Figures

| Figure File | Phase / Stage | Description |
| :--- | :---: | :--- |
| [`fig1_neurosync_architecture_diagram.png`](figures/fig1_neurosync_architecture_diagram.png) | Stage 2 | High-level system architecture of NeuroSync-MoE (Sinc tokenizer, Hierarchical MoE, Cross-Expert Attention, DG-TTA). |
| [`fig2_cross_dataset_montage_alignment.png`](figures/fig2_cross_dataset_montage_alignment.png) | Stage 1 | Montage harmonization: 57-channel UCI 10-20 layout vs. 8-channel OpenNeuro ds008851 spatial mapping. |
| [`fig3_spectral_gating_comparison.png`](figures/fig3_spectral_gating_comparison.png) | Stage 6 | Spectral routing weight boxplots (AUD vs. Control) with Mann-Whitney $U$ test significance annotations ($p < 0.001$). |
| [`fig4_cross_expert_attention.png`](figures/fig4_cross_expert_attention.png) | Stage 6 | Multi-head cross-expert attention heatmaps showing elevated $\theta-\gamma$ cross-frequency coupling in AUD. |
| [`fig5_scalp_topography.png`](figures/fig5_scalp_topography.png) | Stage 6 | MNE 2D scalp topographic maps showing frontal hyper-activation and parietal P300 attenuation. |
| [`fig_phase1_dataset_curation_and_splits.png`](figures/fig_phase1_dataset_curation_and_splits.png) | Stage 1 | Subject-independent stratification splits (84 train, 17 val, 21 test subjects; 11,057 trials). |
| [`fig_phase1b_openneuro_cross_dataset.png`](figures/fig_phase1b_openneuro_cross_dataset.png) | Stage 1 | Secondary OpenNeuro ds008851 cross-dataset profile (60 subjects, 20,854 continuous trials, PSD comparison). |
| [`fig_phase2_architecture_and_spectral_filterbank.png`](figures/fig_phase2_architecture_and_spectral_filterbank.png) | Stage 2 | Differentiable sinc filterbank frequency responses ($\delta, \theta, \alpha, \beta, \gamma$) and parameter budgets. |
| [`fig_phase3_contrastive_pretraining_convergence.png`](figures/fig_phase3_contrastive_pretraining_convergence.png) | Stage 3 | SS-CSP InfoNCE contrastive self-supervised pre-training loss convergence curve (loss: 0.082). |
| [`fig_phase4_finetuning_curves_and_confusion_matrix.png`](figures/fig_phase4_finetuning_curves_and_confusion_matrix.png) | Stage 4 | Supervised fine-tuning loss/accuracy curves and normalized confusion matrix on holdout test subjects. |
| [`fig_phase5_diffusion_denoising_score_matching.png`](figures/fig_phase5_diffusion_denoising_score_matching.png) | Stage 5 | Latent score-based DDPM training loss curve and 2D vector field of reverse diffusion drift. |
| [`fig_phase7_radar_benchmark_comparison.png`](figures/fig_phase7_radar_benchmark_comparison.png) | Stage 7 | Multidimensional radar benchmark comparison across Accuracy, F1-Score, Interpretability, Latency, and Memory. |

---

## 2. Publication LaTeX Tables

All LaTeX tables are formatted with `booktabs` and ready for drop-in compilation in IEEEtran or Elsevier templates.

| Table File | Phase / Stage | Description |
| :--- | :---: | :--- |
| [`table_phase1_dataset_demographics_and_splits.tex`](tables/table_phase1_dataset_demographics_and_splits.tex) | Stage 1 | Primary UCI cohort demographics, clinical diagnosis, and subject-independent splits. |
| [`table_phase1b_openneuro_vs_uci_comparison.tex`](tables/table_phase1b_openneuro_vs_uci_comparison.tex) | Stage 1 | Primary (UCI) vs. Secondary (OpenNeuro ds008851) cross-paradigm specifications. |
| [`table_phase2_model_hyperparameters.tex`](tables/table_phase2_model_hyperparameters.tex) | Stage 2 | NeuroSync-MoE layer-by-layer architectural dimensions and specialist parameter counts. |
| [`table_phase3_contrastive_pretraining_hyperparameters.tex`](tables/table_phase3_contrastive_pretraining_hyperparameters.tex) | Stage 3 | SS-CSP self-supervised contrastive pretraining hyperparameters and data augmentation ratios. |
| [`table_phase4_primary_uci_benchmark.tex`](tables/table_phase4_primary_uci_benchmark.tex) | Stage 4 | Primary holdout test performance on 1,863 unseen trials (Accuracy, F1, Precision, Recall, AUC, Kappa). |
| [`table_phase5_dgt_tta_hyperparameters_and_ablation.tex`](tables/table_phase5_dgt_tta_hyperparameters_and_ablation.tex) | Stage 5 | Diffusion-guided test-time adaptation parameters (timesteps, noise schedule, guidance scale). |
| [`table_phase6_statistical_biomarker_significance.tex`](tables/table_phase6_statistical_biomarker_significance.tex) | Stage 6 | Mann-Whitney $U$ test statistics, asymptotic $p$-values, and clinical interpretation for all 5 frequency bands. |
| [`table_phase7_ablation_and_cross_dataset_transfer.tex`](tables/table_phase7_ablation_and_cross_dataset_transfer.tex) | Stage 7 | Full ablation matrix (Flat-MoE, EEGNet, No SS-CSP, No Attention, No DG-TTA, Full Proposed). |
| [`table_master_all_hyperparameters.tex`](tables/table_master_all_hyperparameters.tex) | Stages 1–7 | Comprehensive master hyperparameter reference table for all pipeline stages. |
| [`tables_summary.md`](tables_summary.md) | Stages 1–7 | Complete Markdown transcription of all tables for quick browser or documentation inspection. |

---

## 3. Quantitative Metric Logs

- [`metrics/neurosync_results.json`](metrics/neurosync_results.json): Contains exact evaluated scores for NeuroSync-MoE (Without TTA vs. With DG-TTA) on both UCI holdout and OpenNeuro ds008851 zero-shot transfer.
- [`metrics/baseline_results.json`](metrics/baseline_results.json): Contains evaluated scores for EEGNet and Flat-MoE baselines.

---
