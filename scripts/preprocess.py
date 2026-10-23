#!/usr/bin/env python3
"""
Preprocessing Entrypoint for NeuroSync-MoE.

Handles:
1. Extraction of raw UCI EEG subject archives (.tar.gz).
2. Pruning of 7 unrecorded/dead channels ([4, 6, 21, 22, 42, 44, 49]) leaving 57 active electrodes.
3. Converting each subject into an on-demand, LRU-cacheable .pt file.
4. Parsing and caching OpenNeuro ds008851 continuous EEG trials with montage harmonization.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.alcoholism_loader import AlcoholismLoader
from src.per_subject_dataset import DEAD_CHANNELS, ACTIVE_INDICES, ACTIVE_CHANNELS
from src.ds008851_loader import OpenNeuroAlcoholismLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Preprocess")


def preprocess_uci(raw_dir: Path, output_dir: Path):
    logger.info(f"--- Preprocessing Primary UCI Dataset ---")
    logger.info(f"Raw source directory: {raw_dir}")
    logger.info(f"Processed target directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        logger.warning(f"Raw directory {raw_dir} not found. Checking if processed files already exist...")
        existing_pt = list(output_dir.glob("*.pt"))
        if len(existing_pt) > 0:
            logger.info(f"Found {len(existing_pt)} existing preprocessed subject files in {output_dir}.")
            return
        else:
            raise FileNotFoundError(f"Neither raw source {raw_dir} nor preprocessed files in {output_dir} exist.")

    loader = AlcoholismLoader(data_dir=raw_dir)
    tar_files = sorted(raw_dir.glob("co*.tar.gz"))
    subject_ids = [f.name.replace(".tar.gz", "") for f in tar_files]
    logger.info(f"Discovered {len(subject_ids)} subjects to process.")

    total_trials = 0
    for idx, sub_id in enumerate(subject_ids, 1):
        pt_file = output_dir / f"{sub_id}.pt"
        if pt_file.exists():
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            total_trials += data["X"].shape[0]
            continue

        try:
            X, y, sub_list = loader.load_subject(sub_id)
            if X is None or len(X) == 0:
                continue

            # Prune dead channels: (N, 64, 256) -> (N, 57, 256)
            X_clean = X[:, ACTIVE_INDICES, :]

            # Z-score normalization per trial
            mean = X_clean.mean(axis=-1, keepdims=True)
            std = X_clean.std(axis=-1, keepdims=True) + 1e-6
            X_norm = (X_clean - mean) / std

            torch.save(
                {
                    "X": torch.from_numpy(X_norm).float(),
                    "y": torch.tensor(y[0], dtype=torch.long),
                    "sub_id": sub_id,
                    "n_trials": len(y),
                    "active_channels": ACTIVE_CHANNELS,
                },
                pt_file,
            )
            total_trials += len(y)
            if idx % 20 == 0 or idx == len(subject_ids):
                logger.info(f"[{idx}/{len(subject_ids)}] Processed subject {sub_id} ({len(y)} trials)")
        except Exception as e:
            logger.error(f"Error processing subject {sub_id}: {e}")

    logger.info(f"UCI Dataset preprocessing complete: {len(subject_ids)} subjects, {total_trials} total trials.")


def preprocess_openneuro(raw_dir: Path, cache_file: Path):
    logger.info(f"\n--- Preprocessing Secondary OpenNeuro ds008851 Dataset ---")
    logger.info(f"Source directory: {raw_dir}")
    logger.info(f"Cache file: {cache_file}")

    if cache_file.exists():
        logger.info(f"Preprocessed OpenNeuro cache already exists at {cache_file}.")
        return

    if not raw_dir.exists():
        logger.warning(f"OpenNeuro directory {raw_dir} not found. Skipping.")
        return

    loader = OpenNeuroAlcoholismLoader(dataset_root=raw_dir, cache_file=cache_file)
    loader.build_cache()
    logger.info(f"OpenNeuro caching complete: {len(loader)} trials cached at {cache_file}.")


def main():
    parser = argparse.ArgumentParser(description="NeuroSync-MoE Preprocessing Script")
    parser.add_argument("--uci_raw", type=str, default="eeg_full", help="Path to raw UCI dataset")
    parser.add_argument("--uci_processed", type=str, default="eeg_processed", help="Path to output processed UCI directory")
    parser.add_argument("--openneuro_raw", type=str, default="ds008851", help="Path to raw OpenNeuro ds008851 dataset")
    parser.add_argument("--openneuro_cache", type=str, default="ds008851/ds008851_processed_cache.npz", help="Path to OpenNeuro cache file")
    args = parser.parse_args()

    preprocess_uci(Path(args.uci_raw), Path(args.uci_processed))
    preprocess_openneuro(Path(args.openneuro_raw), Path(args.openneuro_cache))


if __name__ == "__main__":
    main()
