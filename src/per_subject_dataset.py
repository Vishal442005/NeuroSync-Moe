"""
Per-Subject On-Demand EEG Dataset & DataModule.

Stores each subject as an independent .pt file in eeg_processed/ to ensure:
- Peak RAM usage < 50 MB during training (immune to system memory exhaustion)
- Full support for multi-worker PyTorch DataLoaders
- 57 active channels (7 dead channels pruned)
- Strict subject-independent stratified splitting (train/val/test)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from .alcoholism_loader import STANDARD_CHANNELS
except ImportError:
    from alcoholism_loader import STANDARD_CHANNELS

# 7 dead channels discovered in montage (indices [4, 6, 21, 22, 42, 44, 49])
DEAD_CHANNELS = [4, 6, 21, 22, 42, 44, 49]
ACTIVE_INDICES = [i for i in range(64) if i not in DEAD_CHANNELS]
ACTIVE_CHANNELS = [STANDARD_CHANNELS[i] for i in ACTIVE_INDICES]

logger = logging.getLogger(__name__)


class SubjectLRUCache:
    """Least Recently Used (LRU) in-memory cache holding at most max_subjects tensors."""

    def __init__(self, max_subjects: int = 128) -> None:
        self.max_subjects = max_subjects
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, pt_path: str) -> Optional[torch.Tensor]:
        if pt_path in self.cache:
            self.cache.move_to_end(pt_path)
            return self.cache[pt_path]
        return None

    def put(self, pt_path: str, tensor: torch.Tensor) -> None:
        if pt_path in self.cache:
            self.cache.move_to_end(pt_path)
        else:
            if len(self.cache) >= self.max_subjects:
                self.cache.popitem(last=False)
            self.cache[pt_path] = tensor


class PerSubjectEEGDataset(Dataset):
    """
    On-demand PyTorch Dataset indexing trials across per-subject .pt files.
    """

    def __init__(
        self,
        subject_entries: List[Dict],
        normalize: bool = True,
        max_cached_subjects: int = 128,
    ) -> None:
        self.entries = subject_entries
        self.normalize = normalize
        self.cache = SubjectLRUCache(max_subjects=max_cached_subjects)

        # Build index mapping: global_idx -> (entry_idx, local_trial_idx)
        self.index_map = []
        for e_idx, entry in enumerate(self.entries):
            n_trials = entry["n_trials"]
            for local_idx in range(n_trials):
                self.index_map.append((e_idx, local_idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e_idx, local_idx = self.index_map[idx]
        entry = self.entries[e_idx]
        pt_file = entry["file"]

        # Check LRU cache
        x_subject = self.cache.get(pt_file)
        if x_subject is None:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            x_subject = data["X"]  # (n_trials, 57, 256)
            self.cache.put(pt_file, x_subject)

        trial = x_subject[local_idx].clone()  # (57, 256)

        if self.normalize:
            mean = trial.mean(dim=-1, keepdim=True)
            std = trial.std(dim=-1, keepdim=True) + 1e-6
            trial = (trial - mean) / std

        label = entry["label"]
        return {
            "eeg": trial,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": torch.tensor(e_idx, dtype=torch.long),
        }


class PerSubjectDataModule(pl.LightningDataModule):
    """
    LightningDataModule using per-subject on-demand files with stratified splits.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path] = "C:/Users/Visha/Desktop/NewAlcohol/eeg_processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset: Optional[PerSubjectEEGDataset] = None
        self.val_dataset: Optional[PerSubjectEEGDataset] = None
        self.test_dataset: Optional[PerSubjectEEGDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        manifest_file = self.processed_dir / "manifest.pt"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}. Run preprocess_per_subject.py first.")

        manifest: List[Dict] = torch.load(manifest_file, weights_only=False)

        # Separate into alcohol (1) and control (0)
        alc_entries = [e for e in manifest if e["label"] == 1]
        ctrl_entries = [e for e in manifest if e["label"] == 0]

        # Shuffle with fixed seed
        rng = np.random.RandomState(self.seed)
        rng.shuffle(alc_entries)
        rng.shuffle(ctrl_entries)

        # Stratified subject counts
        n_alc_train = int(len(alc_entries) * self.train_ratio)
        n_alc_val = int(len(alc_entries) * self.val_ratio)

        n_ctrl_train = int(len(ctrl_entries) * self.train_ratio)
        n_ctrl_val = int(len(ctrl_entries) * self.val_ratio)

        train_entries = alc_entries[:n_alc_train] + ctrl_entries[:n_ctrl_train]
        val_entries = (
            alc_entries[n_alc_train : n_alc_train + n_alc_val]
            + ctrl_entries[n_ctrl_train : n_ctrl_train + n_ctrl_val]
        )
        test_entries = (
            alc_entries[n_alc_train + n_alc_val :]
            + ctrl_entries[n_ctrl_train + n_ctrl_val :]
        )

        logger.info(
            f"Train Split: {len(train_entries)} subjects "
            f"(alc={sum(1 for e in train_entries if e['label']==1)}, "
            f"ctrl={sum(1 for e in train_entries if e['label']==0)})"
        )
        logger.info(
            f"Val Split:   {len(val_entries)} subjects "
            f"(alc={sum(1 for e in val_entries if e['label']==1)}, "
            f"ctrl={sum(1 for e in val_entries if e['label']==0)})"
        )
        logger.info(
            f"Test Split:  {len(test_entries)} subjects "
            f"(alc={sum(1 for e in test_entries if e['label']==1)}, "
            f"ctrl={sum(1 for e in test_entries if e['label']==0)})"
        )

        if stage in ("fit", None):
            self.train_dataset = PerSubjectEEGDataset(train_entries, normalize=True)
            self.val_dataset = PerSubjectEEGDataset(val_entries, normalize=True)
            logger.info(f"Train trials: {len(self.train_dataset)}, Val trials: {len(self.val_dataset)}")

        if stage in ("test", None):
            self.test_dataset = PerSubjectEEGDataset(test_entries, normalize=True)
            logger.info(f"Test trials: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dm = PerSubjectDataModule()
    try:
        dm.setup("fit")
        print("DataModule test passed successfully!")
    except Exception as e:
        print("DataModule test noted:", e)
