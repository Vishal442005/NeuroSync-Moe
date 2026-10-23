"""
Robust Streaming and Caching of Balanced 8-Subject Pilot Cohort from OpenNeuro ds005907.
4 AUD patients (AUDIT 7-27) + 4 Healthy Controls (AUDIT 1-2).
Streams via ThreadPoolExecutor, extracts shared 10-20 channels with UCI, downsamples to 256 Hz,
filters 1-50 Hz, segments into 1.0s epochs, and saves persistent ds005907_processed_cache.npz.
"""

import os
import sys
import time
import logging
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import mne

# Suppress verbose mne output
mne.set_log_level("ERROR")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StreamDS005907")

from himoe_backbone import ACTIVE_CHANNELS

OUT_DIR = Path("ds005907")
OUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = OUT_DIR / "temp_raw"
TEMP_DIR.mkdir(exist_ok=True)

TARGET_SUBJECTS = [
    # 4 Controls (AUDIT 1-2)
    {"id": "sub-001", "group": "CTRL", "label": 0, "audit": 1.0, "age": 43},
    {"id": "sub-002", "group": "CTRL", "label": 0, "audit": 2.0, "age": 50},
    {"id": "sub-003", "group": "CTRL", "label": 0, "audit": 1.0, "age": 22},
    {"id": "sub-004", "group": "CTRL", "label": 0, "audit": 1.0, "age": 18},
    # 4 AUD patients (AUDIT 7-27)
    {"id": "sub-007", "group": "AUD", "label": 1, "audit": 7.0, "age": 44},
    {"id": "sub-008", "group": "AUD", "label": 1, "audit": 27.0, "age": 38},
    {"id": "sub-009", "group": "AUD", "label": 1, "audit": 11.0, "age": 32},
    {"id": "sub-010", "group": "AUD", "label": 1, "audit": 9.0, "age": 54},
]

URL_BASE = "https://s3.amazonaws.com/openneuro.org/ds005907"

# Standard 38 channels shared across all 8 subjects and UCI
COMMON_38_CHANNELS = [
    'AFZ', 'AF8', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
    'TP7', 'TP8', 'CP3', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'P4',
    'P6', 'P8', 'PO7', 'POZ', 'PO8', 'O1', 'OZ', 'O2', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC6', 'C5', 'C3', 'C2', 'C4', 'C6'
]


def download_with_retry(url: str, out_path: Path, max_retries: int = 5) -> bool:
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp, open(out_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break
                    f.write(chunk)
            if out_path.stat().st_size > 1000000:
                return True
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed for {url}: {e}")
            time.sleep(3)
    return False


def process_subject(sub_meta: dict, target_sfreq: float = 256.0):
    sub_id = sub_meta["id"]
    group = sub_meta["group"]
    audit = sub_meta["audit"]
    label = sub_meta["label"]

    sub_cache = OUT_DIR / f"{sub_id}_epochs.npz"
    if sub_cache.exists():
        logger.info(f"Loaded existing cache for {sub_id} from {sub_cache}")
        loaded = np.load(sub_cache)
        return {
            "sub_id": sub_id,
            "data": loaded["data"],
            "channels": loaded["channels"].tolist(),
            "label": label,
            "audit": audit,
        }

    set_filename = f"{sub_id}_task-ThreeArmedBandit_eeg.set"
    set_url = f"{URL_BASE}/{sub_id}/eeg/{set_filename}"
    local_path = TEMP_DIR / set_filename

    logger.info(f"Downloading {sub_id} ({group}, AUDIT={audit})...")
    t0 = time.time()
    ok = download_with_retry(set_url, local_path)
    if not ok:
        logger.error(f"Failed to download {sub_id}")
        return None

    dl_time = time.time() - t0
    logger.info(f"Downloaded {sub_id} ({local_path.stat().st_size / (1024*1024):.1f} MB in {dl_time:.1f}s). Processing MNE raw...")

    try:
        raw = mne.io.read_raw_eeglab(str(local_path), preload=True, verbose=False)
        # Uppercase channel names for standardized mapping
        raw_ch_map = {c: c.upper() for c in raw.ch_names}
        raw.rename_channels(raw_ch_map)

        # Pick channels present in COMMON_38_CHANNELS
        avail_ch = [c for c in COMMON_38_CHANNELS if c in raw.ch_names]
        if len(avail_ch) < 35:
            logger.warning(f"Only {len(avail_ch)} common channels in {sub_id}. Available: {raw.ch_names}")
            return None

        raw.pick(avail_ch)

        if raw.info["sfreq"] != target_sfreq:
            raw.resample(target_sfreq, verbose=False)

        raw.filter(1.0, 50.0, verbose=False)

        data = raw.get_data()  # (C, T)
        epoch_len = int(target_sfreq)  # 256 samples = 1.0 sec
        n_epochs = data.shape[1] // epoch_len
        epochs = data[:, :n_epochs * epoch_len].reshape(len(avail_ch), n_epochs, epoch_len)
        epochs = np.transpose(epochs, (1, 0, 2))  # (N, C, 256)

        # Save per-subject cache
        np.savez_compressed(
            sub_cache,
            data=epochs,
            channels=np.array(avail_ch),
            label=label,
            audit=audit,
        )
        logger.info(f"Successfully processed {sub_id}: {epochs.shape[0]} epochs, {len(avail_ch)} channels. Saved {sub_cache}")

        # Clean raw file
        if local_path.exists():
            local_path.unlink()

        return {
            "sub_id": sub_id,
            "data": epochs,
            "channels": avail_ch,
            "label": label,
            "audit": audit,
        }
    except Exception as e:
        logger.error(f"Error processing {sub_id}: {e}")
        if local_path.exists():
            local_path.unlink()
        return None


def main():
    logger.info(f"Starting parallel ingestion of {len(TARGET_SUBJECTS)} subjects from ds005907...")
    start_time = time.time()

    results = []
    # Use 3 concurrent workers to stream cleanly
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_sub = {executor.submit(process_subject, s): s["id"] for s in TARGET_SUBJECTS}
        for future in as_completed(future_to_sub):
            sub_id = future_to_sub[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                logger.error(f"Exception processing {sub_id}: {e}")

    if not results:
        logger.error("No subjects successfully processed!")
        return

    # Sort results to keep consistent subject ordering
    results.sort(key=lambda x: x["sub_id"])

    # Determine exact common channel set across all successfully processed subjects
    common_ch_sets = [set(r["channels"]) for r in results]
    final_channels = [c for c in COMMON_38_CHANNELS if all(c in ch_set for ch_set in common_ch_sets)]
    logger.info(f"Final standardized channel montage across {len(results)} subjects: {len(final_channels)} channels")
    logger.info(f"Channels: {final_channels}")

    all_aligned_data = []
    all_labels = []
    all_subs = []
    all_audit = []

    for r in results:
        sub_ch = r["channels"]
        ch_idx = [sub_ch.index(c) for c in final_channels]
        aligned_epochs = r["data"][:, ch_idx, :]  # (N, C_final, 256)
        all_aligned_data.append(aligned_epochs)
        
        n_sub_trials = aligned_epochs.shape[0]
        all_labels.extend([r["label"]] * n_sub_trials)
        all_subs.extend([r["sub_id"]] * n_sub_trials)
        all_audit.extend([r["audit"]] * n_sub_trials)
        logger.info(f"  {r['sub_id']}: {n_sub_trials} trials aligned to {len(final_channels)} channels.")

    final_data_array = np.concatenate(all_aligned_data, axis=0)  # (Total_N, C_final, 256)
    final_labels_array = np.array(all_labels, dtype=np.int64)
    final_subs_array = np.array(all_subs)
    final_audit_array = np.array(all_audit, dtype=np.float32)
    final_channels_array = np.array(final_channels)

    out_cache = OUT_DIR / "ds005907_processed_cache.npz"
    np.savez_compressed(
        out_cache,
        data=final_data_array,
        labels=final_labels_array,
        subject_ids=final_subs_array,
        audit_scores=final_audit_array,
        channels=final_channels_array,
    )

    logger.info("=" * 60)
    logger.info(f"SUCCESS! Master cache saved: {out_cache}")
    logger.info(f"Total Trials: {final_data_array.shape[0]}")
    logger.info(f"Shape: {final_data_array.shape} (Trials x Channels x Time)")
    logger.info(f"Cache File Size: {out_cache.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"Total Pipeline Runtime: {time.time() - start_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
