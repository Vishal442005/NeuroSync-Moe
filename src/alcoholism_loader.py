"""
UCI EEG Alcoholism Dataset Loader.

Self-contained dataset loader adapted for the Rest-Diff-BCI + MoE-BFT framework.
Handles alcoholic vs control subject classification from the UCI EEG database.
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Standard 64-channel names (same as PhysioNet)
STANDARD_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8",
    "T9", "T10",
    "TP7", "TP8",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "IZ",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
]


class AlcoholismLoader:
    """
    Loader for the UCI EEG Alcoholism dataset.

    Data parameters:
        - 122 subjects (alcoholic: co2a*, control: co2c*)
        - 64 channels, 256 Hz sampling rate
        - Trial duration: 1 second (256 samples)
        - Conditions: S1 (single stimulus), S2m (matching), S2n (non-matching)
        - Each subject: 120 trials
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "C:/Users/Visha/Desktop/NewAlcohol/eeg_full",
        sfreq: float = 256.0,
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        trial_duration: float = 1.0,
        resting_duration: float = 30.0,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize Alcoholism loader.

        Args:
            data_dir: Path to directory containing subject tar.gz files
            sfreq: Sampling rate (UCI EEG native is 256.0 Hz)
            l_freq: Low cutoff frequency for bandpass filter in Hz
            h_freq: High cutoff frequency for bandpass filter in Hz
            trial_duration: Length of trials in seconds (default 1.0s = 256 samples)
            resting_duration: Duration of resting-state baseline window in seconds (default 30.0s)
            cache_dir: Optional directory to cache preprocessed NumPy arrays
        """
        self.data_dir = Path(data_dir)
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.trial_duration = trial_duration
        self.resting_duration = resting_duration
        self.trial_samples = int(self.trial_duration * self.sfreq)
        self.resting_samples = int(self.resting_duration * self.sfreq)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Alcoholism data directory not found: {self.data_dir}")

    def get_subject_folder(self, subject_id: str) -> Path:
        """Return path to subject directory (e.g. eeg_full/co2a0000364)."""
        # Extract from tar.gz if not already extracted
        subject_tar = self.data_dir / f"{subject_id}.tar.gz"
        subject_folder = self.data_dir / subject_id

        if not subject_folder.exists():
            if subject_tar.exists():
                import tarfile
                logger.info(f"Extracting {subject_tar}...")
                with tarfile.open(subject_tar, 'r:gz') as tar:
                    tar.extractall(self.data_dir)
            else:
                logger.warning(f"Neither folder nor tar.gz found for {subject_id}")

        return subject_folder

    def _parse_rd_file(self, rd_file_path: Path) -> np.ndarray:
        """
        Parse a .rd.gz file containing EEG trial data.

        Format:
            # header lines
            # 120 trials, 64 chans, 416 samples 368 post_stim samples
            # 3.906000 msecs uV
            # S1 obj , trial X
            # FP1 chan 0
            trial_idx channel_name sample_idx voltage

        Each .rd file contains data for ONE trial across all 64 channels.

        Returns:
            EEG array of shape (1, 64, n_samples) - single trial
        """
        if not rd_file_path.exists():
            raise FileNotFoundError(f"RD file not found: {rd_file_path}")

        # Read gzipped file
        with gzip.open(rd_file_path, 'rt') as f:
            lines = f.readlines()

        # Channel name to index mapping
        channel_name_to_idx = {name: idx for idx, name in enumerate(STANDARD_CHANNELS)}

        # Initialize data array: 64 channels x unknown samples
        data_dict = {}  # {channel_idx: [samples]}

        current_channel_idx = None

        for line in lines:
            line = line.strip()

            if line.startswith('#'):
                # Check for channel header
                if 'chan' in line.lower():
                    parts = line.split()
                    if len(parts) >= 4:
                        channel_name = parts[1].upper()
                        if channel_name in channel_name_to_idx:
                            current_channel_idx = channel_name_to_idx[channel_name]
                            data_dict[current_channel_idx] = []
                continue

            # Parse data line: trial_idx channel_name sample_idx voltage
            parts = line.split()
            if len(parts) >= 4:
                try:
                    trial = int(parts[0])
                    channel_name = parts[1].upper()
                    sample = int(parts[2])
                    voltage = float(parts[3])

                    if channel_name in channel_name_to_idx:
                        channel_idx = channel_name_to_idx[channel_name]
                        if channel_idx not in data_dict:
                            data_dict[channel_idx] = []
                        data_dict[channel_idx].append(voltage)
                except (ValueError, IndexError):
                    continue

        if not data_dict:
            raise ValueError(f"No data found in {rd_file_path}")

        # Determine max samples
        max_sample = max(len(v) for v in data_dict.values())

        # Create array for single trial: (1, 64, n_samples)
        eeg_data = np.zeros((1, 64, max_sample), dtype=np.float32)

        for channel_idx, samples in data_dict.items():
            if channel_idx < 64:
                eeg_data[0, channel_idx, :len(samples)] = samples

        return eeg_data

    def load_subject_trials(
        self,
        subject_id: str,
        conditions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load trials for a given subject.

        Args:
            subject_id: Subject identifier (e.g., "co2a0000364" for alcoholic, "co2c0000337" for control)
            conditions: List of conditions to load (e.g., ["S1", "S2m", "S2n"]). Default: all available.

        Returns:
            X: np.ndarray of shape (N_trials, 64, trial_samples)
            y: np.ndarray of shape (N_trials,) with labels (0=control, 1=alcoholic)
        """
        subject_folder = self.get_subject_folder(subject_id)

        if not subject_folder.exists():
            logger.warning(f"Subject folder not found: {subject_folder}")
            return np.empty((0, 64, self.trial_samples), dtype=np.float32), np.empty((0,), dtype=np.int64)

        # Determine label from subject ID (co2a* and co3a* are alcoholic; co2c* and co3c* are control)
        is_alcoholic = subject_id.startswith(('co2a', 'co3a'))
        label = 1 if is_alcoholic else 0

        # Find all .rd.gz files
        rd_files = list(subject_folder.glob("*.rd.*.gz"))

        if not rd_files:
            logger.warning(f"No .rd.gz files found for subject {subject_id}")
            return np.empty((0, 64, self.trial_samples), dtype=np.float32), np.empty((0,), dtype=np.int64)

        all_trials = []

        for rd_file in rd_files:
            try:
                eeg_data = self._parse_rd_file(rd_file)  # (1, 64, n_samples) - single trial

                # Resample or truncate to target samples
                n_trials, n_chans, n_samples = eeg_data.shape

                if n_samples != self.trial_samples:
                    # Simple truncation or padding (can add resampling later)
                    if n_samples > self.trial_samples:
                        eeg_data = eeg_data[:, :, :self.trial_samples]
                    else:
                        pad_width = self.trial_samples - n_samples
                        eeg_data = np.pad(eeg_data, ((0, 0), (0, 0), (0, pad_width)), mode='edge')

                all_trials.append(eeg_data)

            except Exception as e:
                logger.error(f"Error parsing {rd_file}: {e}")

        if not all_trials:
            return np.empty((0, 64, self.trial_samples), dtype=np.float32), np.empty((0,), dtype=np.int64)

        X = np.concatenate(all_trials, axis=0).astype(np.float32)
        y = np.full((len(X),), label, dtype=np.int64)

        return X, y

    def load_dataset(
        self,
        subject_ids: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Load dataset across multiple subjects.

        Args:
            subject_ids: List of subject IDs to load. If None, loads all available subjects.
            max_subjects: Maximum number of subjects to load (for testing).

        Returns:
            Dict containing:
                - 'X': np.ndarray of shape (Total_trials, 64, T)
                - 'y': np.ndarray of shape (Total_trials,)
                - 'subject_idx': np.ndarray of shape (Total_trials,)
                - 'subject_ids': List of subject ID strings
        """
        # Check if zero-RAM memory-mapped dataset exists
        x_mmap_path = self.data_dir / "uci_eeg_X.npy"
        y_path = self.data_dir / "uci_eeg_y.npy"
        sub_path = self.data_dir / "uci_eeg_sub.npy"
        sub_ids_path = self.data_dir / "uci_eeg_sub_ids.npy"

        if subject_ids is None and max_subjects is None and x_mmap_path.exists() and y_path.exists():
            logger.info(f"Loading memory-mapped dataset from {x_mmap_path} (0 MB physical RAM overhead)...")
            X_mmap = np.load(str(x_mmap_path), mmap_mode="r")
            y_arr = np.load(str(y_path))
            sub_arr = np.load(str(sub_path))
            sub_ids = list(np.load(str(sub_ids_path)))
            return {
                "X": X_mmap,
                "y": y_arr,
                "subject_idx": sub_arr,
                "subject_ids": sub_ids,
            }

        # Find all available subject tar.gz files (co2* and co3*)
        if subject_ids is None:
            tar_files = sorted(self.data_dir.glob("co*.tar.gz"))
            subject_ids = [f.name.replace('.tar.gz', '') for f in tar_files]

        if max_subjects:
            subject_ids = subject_ids[:max_subjects]

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        sub_list: List[np.ndarray] = []
        valid_subject_ids = []

        for idx, sub_id in enumerate(subject_ids):
            try:
                X_sub, y_sub = self.load_subject_trials(sub_id)
                if len(X_sub) > 0:
                    X_list.append(X_sub)
                    y_list.append(y_sub)
                    sub_list.append(np.full((len(y_sub),), idx, dtype=np.int64))
                    valid_subject_ids.append(sub_id)
                    logger.info(f"Loaded {len(X_sub)} trials from {sub_id}")
            except Exception as e:
                logger.error(f"Failed to load subject {sub_id}: {e}")

        if not X_list:
            raise RuntimeError("No valid trials loaded for specified subjects.")

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        sub_all = np.concatenate(sub_list, axis=0)

        # Save to disk cache if full dataset was loaded
        if max_subjects is None and len(valid_subject_ids) >= 100:
            logger.info(f"Saving dataset cache to {cache_file}...")
            try:
                np.savez_compressed(
                    cache_file,
                    X=X_all,
                    y=y_all,
                    subject_idx=sub_all,
                    subject_ids=np.array(valid_subject_ids, dtype=object),
                )
                logger.info(f"Cache successfully saved ({cache_file.stat().st_size / (1024*1024):.2f} MB).")
            except Exception as e:
                logger.warning(f"Could not save dataset cache: {e}")

        return {
            "X": X_all,
            "y": y_all,
            "subject_idx": sub_all,
            "subject_ids": valid_subject_ids,
        }


if __name__ == "__main__":
    # Test the loader
    loader = AlcoholismLoader()
    data = loader.load_dataset(max_subjects=2)
    print(f"Loaded dataset:")
    print(f"  X shape: {data['X'].shape}")
    print(f"  y shape: {data['y'].shape}")
    print(f"  subject_idx shape: {data['subject_idx'].shape}")
    print(f"  Subject IDs: {data['subject_ids']}")
    print(f"  Class distribution: {np.bincount(data['y'])}")
