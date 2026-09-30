#!/usr/bin/env python3
"""
Dataset adapter for CSDI official code.

Provides OurPhysioDataset that mimics Physio_Dataset from CSDI's
dataset_physio.py, but reads from our preprocessed npy arrays instead
of raw PhysioNet txt files.

The CSDI model expects DataLoader batches with keys:
  observed_data  : (B, T, D) float32
  observed_mask  : (B, T, D) float32  — 1 = originally observed
  gt_mask        : (B, T, D) float32  — 1 = visible to model (observed - held-out)
  timepoints     : (B, T) float32
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OurPhysioDataset(Dataset):
    """
    Wraps our (X, M_orig, eval_mask) arrays into CSDI's expected format.

    Parameters
    ----------
    X : (N, T, D) float32  — observed values, 0 where M_orig==0
    M_orig : (N, T, D) float32  — original observation mask
    eval_mask : (N, T, D) float32  — held-out positions (1 = held out for eval)
    """

    def __init__(self, X: np.ndarray, M_orig: np.ndarray,
                 eval_mask: np.ndarray):
        N, T, D = X.shape
        self.T = T

        self.observed_data = (X * M_orig).astype(np.float32)
        self.observed_mask = M_orig.astype(np.float32)
        self.gt_mask = (M_orig * (1 - eval_mask)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.observed_data)

    def __getitem__(self, idx: int) -> dict:
        return {
            "observed_data": self.observed_data[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": np.arange(self.T, dtype=np.float32),
        }


def get_our_dataloader(X: np.ndarray, M_orig: np.ndarray,
                       eval_mask: np.ndarray, batch_size: int = 16,
                       shuffle: bool = False) -> DataLoader:
    """Create a DataLoader from our arrays for CSDI."""
    dataset = OurPhysioDataset(X, M_orig, eval_mask)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
