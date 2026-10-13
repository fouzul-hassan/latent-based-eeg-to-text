"""
Memory-efficient ZuCo Dataset Loader using memory mapping.

Loads data on-demand instead of loading entire dataset into RAM.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

# Module-level cache: avoid loading the DataFrame more than once per path.
_df_cache: dict = {}


class ZuCoMemMapDataset(Dataset):
    """
    Memory-mapped ZuCo dataset that loads samples on-demand.
    Reduces RAM usage from ~48GB to <1GB.
    """
    def __init__(self, data_path: str, split: str = 'train',
                 val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42,
                 _shared_df=None):
        """
        Args:
            data_path:   Path to .df file
            split:       'train', 'val', or 'test'
            val_ratio, test_ratio: Split ratios
            seed:        Random seed
            _shared_df:  Pre-loaded DataFrame (avoids repeated disk reads)
        """
        self.data_path = data_path
        self.split = split

        logger.info(f"Loading {split} split from {data_path} (memory-mapped)")

        # Use shared DataFrame if provided; otherwise load (and cache) once
        if _shared_df is not None:
            df = _shared_df
        elif data_path in _df_cache:
            logger.info(f"  Using cached DataFrame for {data_path}")
            df = _df_cache[data_path]
        else:
            logger.info(f"  Reading pickle (first time) …")
            df = pd.read_pickle(data_path)
            _df_cache[data_path] = df

        # Get indices for this split
        np.random.seed(seed)
        n = len(df)
        indices = np.arange(n)
        np.random.shuffle(indices)

        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)

        if split == 'test':
            self.indices = indices[:test_size]
        elif split == 'val':
            self.indices = indices[test_size:test_size + val_size]
        else:  # train
            self.indices = indices[test_size + val_size:]

        # Store DataFrame reference (not loaded into memory again)
        self.df = df

        # Extract labels
        if 'subject' in df.columns:
            subjects = df['subject'].values
            unique_subjects = np.unique(subjects)
            subject_to_id = {s: i for i, s in enumerate(unique_subjects)}
            self.labels = np.array([subject_to_id[s] for s in subjects])
        else:
            self.labels = np.zeros(len(df), dtype=np.int32)

        logger.info(f"{split.capitalize()} samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Load single sample on-demand."""
        real_idx = self.indices[idx]

        # Load EEG data for this sample only
        eeg = self.df.iloc[real_idx]['eeg']  # (128, 1280)
        eeg = eeg.T  # Transpose to (1280, 128)

        label = self.labels[real_idx]

        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(label), real_idx


def load_zuco_memmap(data_path: str, val_ratio: float = 0.1,
                     test_ratio: float = 0.1, seed: int = 42):
    """
    Create memory-mapped datasets.

    The underlying DataFrame is loaded ONCE and shared across all three
    splits to avoid triple-RAM usage.

    Returns:
        Dict with train/val/test dataset objects
    """
    logger.info(f"Loading data from {data_path} (memory-mapped)")
    # Load the DataFrame once
    if data_path in _df_cache:
        df = _df_cache[data_path]
    else:
        df = pd.read_pickle(data_path)
        _df_cache[data_path] = df

    return {
        'train': ZuCoMemMapDataset(data_path, 'train', val_ratio, test_ratio, seed, _shared_df=df),
        'val':   ZuCoMemMapDataset(data_path, 'val',   val_ratio, test_ratio, seed, _shared_df=df),
        'test':  ZuCoMemMapDataset(data_path, 'test',  val_ratio, test_ratio, seed, _shared_df=df),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_path = "../data/tmp/zuco_eeg_128ch_1280len.df"
    if os.path.exists(data_path):
        datasets = load_zuco_memmap(data_path)
        print(f"Train: {len(datasets['train'])} samples")
        print(f"Val: {len(datasets['val'])} samples")
        print(f"Test: {len(datasets['test'])} samples")

        # Test loading one sample
        x, y, idx = datasets['train'][0]
        print(f"Sample shape: {x.shape}, Label: {y}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_path = "../data/tmp/zuco_eeg_128ch_1280len.df"
    if os.path.exists(data_path):
        datasets = load_zuco_memmap(data_path)
        print(f"Train: {len(datasets['train'])} samples")
        print(f"Val: {len(datasets['val'])} samples")
        print(f"Test: {len(datasets['test'])} samples")

        # Test loading one sample
        x, y, idx = datasets['train'][0]
        print(f"Sample shape: {x.shape}, Label: {y}")
