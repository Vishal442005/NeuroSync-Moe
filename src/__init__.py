"""
NeuroSync-MoE: Hierarchical Spectral-Spatial Mixture-of-Experts with Diffusion-Guided
Test-Time Adaptation for Cross-Subject and Cross-Dataset EEG Alcoholism Detection.
"""

from .spectral_tokenizer import DifferentiableSpectralTokenizer
from .himoe_backbone import HierarchicalMoE, ACTIVE_CHANNELS, STANDARD_CHANNELS
from .contrastive_pretrainer import ContrastivePretrainer, SpectralSpatialAugmenter
from .diffusion_tta import LatentDiffusionAdapter, LatentScoreNet
from .per_subject_dataset import PerSubjectEEGDataset, PerSubjectDataModule, SubjectLRUCache, DEAD_CHANNELS
from .alcoholism_loader import AlcoholismLoader
from .ds008851_loader import OpenNeuroAlcoholismLoader, DS008851_CHANNELS
from .baselines_models import EEGNet, ShallowFBCSPNet, FlatMoE

# Aliases for convenience
SS_CSP_Pretrainer = ContrastivePretrainer
DiffusionTTA = LatentDiffusionAdapter

__all__ = [
    "DifferentiableSpectralTokenizer",
    "HierarchicalMoE",
    "ContrastivePretrainer",
    "SS_CSP_Pretrainer",
    "SpectralSpatialAugmenter",
    "LatentDiffusionAdapter",
    "LatentScoreNet",
    "DiffusionTTA",
    "PerSubjectEEGDataset",
    "PerSubjectDataModule",
    "SubjectLRUCache",
    "ACTIVE_CHANNELS",
    "DEAD_CHANNELS",
    "AlcoholismLoader",
    "STANDARD_CHANNELS",
    "OpenNeuroAlcoholismLoader",
    "DS008851_CHANNELS",
    "EEGNet",
    "ShallowFBCSPNet",
    "FlatMoE",
]
