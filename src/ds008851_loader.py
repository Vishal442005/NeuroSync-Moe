"""
OpenNeuro ds008851 Alcohol Dependence vs Healthy Control EEG Dataset Loader.

Dataset summary:
- 60 subjects (30 Alcohol_Dependent, 30 Healthy_Control)
- 3 sessions per subject = 180 runs total (~116 seconds continuous per run)
- 8 channels: ['Fp1', 'Fp2', 'F3', 'F4', 'O1', 'O2', 'Fz', 'Pz'] at 250 Hz
- Channels map directly to spatial experts:
    Left: ['Fp1', 'F3', 'O1']
    Right: ['Fp2', 'F4', 'O2']
    Midline: ['Fz', 'Pz']
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DS008851_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "O1", "O2", "Fz", "Pz"]

class OpenNeuroAlcoholismLoader:
    """
    Loader for OpenNeuro ds008851 EEG dataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "C:/Users/Visha/Desktop/NewAlcohol/ds008851",
        sfreq: float = 256.0,
        epoch_duration: float = 1.0,
        cache_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_sfreq = sfreq
        self.epoch_duration = epoch_duration
        self.epoch_samples = int(self.epoch_duration * self.target_sfreq)
        self.cache_file = Path(cache_file) if cache_file else self.data_dir / "ds008851_processed_cache.npz"

        if not self.data_dir.exists():
            raise FileNotFoundError(f"ds008851 directory not found: {self.data_dir}")

    def load_participants(self) -> Dict[str, int]:
        """Read participants.tsv and return {sub-id: 0 (control) or 1 (alcohol)}."""
        participants_file = self.data_dir / "participants.tsv"
        sub_labels = {}
        with open(participants_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sub_id = parts[0].strip()
                group = parts[1].strip()
                # 1 = Alcohol_Dependent, 0 = Healthy_Control
                label = 1 if "alcohol" in group.lower() else 0
                sub_labels[sub_id] = label
        return sub_labels

    def load_subject_epochs(
        self, sub_id: str, label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all session EDF files for a subject, filter, resample to target sfreq,
        and segment into 1.0-second non-overlapping epochs.
        """
        sub_dir = self.data_dir / sub_id
        edf_files = list(sub_dir.glob("**/*_eeg.edf"))
        
        all_epochs = []
        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                # Ensure channel order matches DS008851_CHANNELS
                raw.pick(DS008851_CHANNELS)
                raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
                if raw.info["sfreq"] != self.target_sfreq:
                    raw.resample(self.target_sfreq, verbose=False)
                
                data = raw.get_data()  # (8, n_samples)
                n_samples = data.shape[1]
                n_epochs = n_samples // self.epoch_samples
                
                if n_epochs > 0:
                    truncated = data[:, : n_epochs * self.epoch_samples]
                    epochs = truncated.reshape(8, n_epochs, self.epoch_samples).transpose(1, 0, 2)
                    all_epochs.append(epochs)
            except Exception as e:
                logger.warning(f"Error parsing {edf_path}: {e}")

        if not all_epochs:
            return np.empty((0, 8, self.epoch_samples), dtype=np.float32), np.empty((0,), dtype=np.int64)

        X = np.concatenate(all_epochs, axis=0).astype(np.float32)
        y = np.full((len(X),), label, dtype=np.int64)
        return X, y

    def load_dataset(self) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Load full ds008851 dataset across all 60 subjects.
        Caches to .npz for instantaneous subsequent loads.
        """
        if self.cache_file.exists():
            logger.info(f"Loading cached ds008851 from {self.cache_file}...")
            cached = np.load(self.cache_file, allow_pickle=True)
            return {
                "X": cached["X"],
                "y": cached["y"],
                "subject_idx": cached["subject_idx"],
                "subject_ids": list(cached["subject_ids"]),
            }

        sub_labels = self.load_participants()
        logger.info(f"Found {len(sub_labels)} participants in ds008851.")

        X_list = []
        y_list = []
        sub_list = []
        valid_subs = []

        for idx, (sub_id, label) in enumerate(sorted(sub_labels.items())):
            X_sub, y_sub = self.load_subject_epochs(sub_id, label)
            if len(X_sub) > 0:
                X_list.append(X_sub)
                y_list.append(y_sub)
                sub_list.append(np.full((len(y_sub),), idx, dtype=np.int64))
                valid_subs.append(sub_id)
                logger.info(f"Loaded {len(X_sub)} epochs for {sub_id} (label={label})")

        if not X_list:
            raise RuntimeError("No valid trials loaded from ds008851.")

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        sub_all = np.concatenate(sub_list, axis=0)

        logger.info(f"Saving ds008851 cache to {self.cache_file}...")
        np.savez_compressed(
            self.cache_file,
            X=X_all,
            y=y_all,
            subject_idx=sub_all,
            subject_ids=np.array(valid_subs, dtype=object),
        )

        return {
            "X": X_all,
            "y": y_all,
            "subject_idx": sub_all,
            "subject_ids": valid_subs,
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = OpenNeuroAlcoholismLoader()
    data = loader.load_dataset()
    print("ds008851 Loaded:")
    print("  X shape:", data["X"].shape)
    print("  y shape:", data["y"].shape)
    print("  Class distribution:", np.bincount(data["y"]))
    print("  Subjects:", len(data["subject_ids"]))
