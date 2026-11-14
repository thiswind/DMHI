#!/usr/bin/env python3
"""Vendored PSW-I official classes for E6 (strict-official re-run).

Provenance: cloned from https://github.com/FMLYD/PSW-I (ICLR 2025,
"Optimal Transport for Time Series Imputation") on 2026-11-14.

- ``OTImputationIni`` is copied VERBATIM from ``benchmark.py`` (the class
  defined in the ``__main__`` block, lines ~236-424). This is the FULL
  official model: FFT spectral distance + MVA initializer + uot_mm
  transport + internal validation early-stop. It is stronger than the
  bare ``model.py::OTImputation`` used by the old ``run_psw_i.py``.
- ``MVAImputation`` is copied from ``model_others.py``.

Only deliberate modification: the module-global ``DEVICE`` of benchmark.py
is replaced by an explicit ``device`` constructor argument (default cpu),
because we run E6 on CPU while the shared GPUs are occupied. Everything
else — including the early-stop quirk (``break`` before the lr-halving
lines, which are unreachable in the original) — is kept byte-equivalent so
the re-run is defensible as "strictly following the official
implementation and configuration".
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import TransformerMixin
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pygrinder import mcar


class MVAImputation(TransformerMixin):
    """Moving-average initializer (official model_others.py, verbatim)."""

    def __init__(self, window_length, window_type, min_periods=1):
        super().__init__()
        self.window_length = window_length
        self.window_type = window_type
        self.min_periods = min_periods

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs):
        X = pd.DataFrame(X)
        X_roll = X.rolling(window=self.window_length, min_periods=self.min_periods,
                           center=True, win_type=self.window_type).mean()
        X.fillna(X_roll, inplace=True)
        X = X.to_numpy()
        return X


class OTImputationIni(TransformerMixin):
    """Sinkhorn imputation — official benchmark.py class, verbatim.

    Args:
        eps: float, default=0.01
            Sinkhorn regularization parameter.
        lr : float, default = 0.01
            Learning rate.
        opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
            Optimizer class to use for fitting.
        n_epochs : int, default=15
            Number of gradient updates for each model within a cycle.
        batch_size : int, default=256
            Size of the batches on which the sinkhorn divergence is evaluated.
        n_pairs : int, default=10
            Number of batch pairs used per gradient update.
        noise : float, default = 0.1
            Noise used for the missing values initialization.
        scaling: float, default=0.9
            Scaling parameter in Sinkhorn iterations
    """

    def __init__(
        self,
        lr: float = 1e-2,
        opt=torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-2,
        reg_sk: float = 1,
        numItermax: int = 1000,
        stopThr=1e-9,
        normalize=0,
        initializer=None,
        replace=False,
        seq_length=16,
        distance=1,
        ot_type='sinkhorn',
        reg_m=10,
        dropout=0.1,
        device='cpu',  # MODIFIED: was module-global DEVICE in benchmark.py
    ):
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.reg_sk = reg_sk
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize
        self.initializer = initializer
        self.replace = replace
        self.seq_length = seq_length
        self.distance = distance
        self.ot_type = ot_type
        self.dropout = nn.Dropout(p=dropout)
        self.reg_m = reg_m
        self.device = device  # MODIFIED
        # ADDED (does not alter math): per-epoch curves for R3-2b evidence.
        self.history_ = []

    def fit_transform(self, X: pd.DataFrame, X_all, *args, **kwargs):
        DEVICE = self.device  # MODIFIED: bound from constructor
        all_mask = np.isnan(X)
        val = X
        train = mcar(X, p=0.1)
        mask = np.isnan(train)
        val_mask = np.isnan(train) ^ (np.isnan(X))
        maes = []
        mses = []
        mres = []
        val_maes = []
        val_mses = []
        val_mres = []
        tick = 0
        if self.initializer is not None:
            imps = self.initializer.fit_transform(train)

            while np.any(np.isnan(imps)):
                temp_mask = np.isnan(imps)
                imps[temp_mask] = self.initializer.fit_transform(imps)[temp_mask]
            imps = imps
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + imps)[mask]
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        train = torch.tensor(train).to(DEVICE)
        train = train.clone()
        val = torch.tensor(val).to(DEVICE)
        val = val.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        all_mask = torch.tensor(all_mask).to(DEVICE)
        val_mask = torch.tensor(val_mask).to(DEVICE)

        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        import ot  # POT

        for i in range(self.n_epochs):
            X_filled = train.detach().clone().double()
            X_filled[mask.bool()] = imps.double()
            loss = 0
            tick += 1
            for _ in range(self.n_pairs):
                optimizer.zero_grad()
                coeff = 1
                idx1 = np.random.choice((n - self.seq_length) // coeff, self.batch_size, replace=self.replace) * coeff
                idx2 = np.random.choice((n - self.seq_length) // coeff, self.batch_size, replace=self.replace) * coeff

                X1 = torch.stack([(X_filled[idx:idx + self.seq_length]) for idx in idx1], dim=0)
                X2 = torch.stack([X_filled[idx:idx + self.seq_length] for idx in idx2], dim=0)
                if self.distance == 'time':
                    M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
                elif self.distance == 'fft':
                    X1 = torch.fft.rfft(X1.transpose(1, 2))
                    X2 = torch.fft.rfft(X2.transpose(1, 2))
                    M = (self.dropout(((X1.flatten(1)[:, None, :]) - X2.flatten(1)[None, :, :]).abs())).sum(-1)
                elif self.distance == 'fft_mag':
                    X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
                    X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
                    M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
                elif self.distance == 'fft_mag_abs':
                    X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
                    X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
                    M = torch.norm(X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :], p=1, dim=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), device=M.device) / self.batch_size, \
                    torch.ones((self.batch_size,), device=M.device) / self.batch_size
                if self.ot_type == 'sinkhorn':
                    pi = ot.sinkhorn(a, b, M, reg=self.reg_sk, max_iter=self.numItermax, tol_rel=self.stopThr).detach()
                elif self.ot_type == 'emd':
                    pi = ot.emd2(a, b, M, numItermax=self.numItermax).detach()
                elif self.ot_type == 'uot':
                    pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=self.reg_sk, stopThr=self.stopThr,
                                                           numItermax=self.numItermax, reg_m=self.reg_m).detach()
                elif self.ot_type == 'uot_mm':
                    pi = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=self.reg_m, c=None, reg=0, div='kl', G0=None,
                                                     numItermax=self.numItermax, stopThr=self.stopThr).detach()
                loss = loss + (pi * M).sum() / self.n_pairs

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    # Catch numerical errors/overflows (should not happen)
                    break

            loss.backward()
            optimizer.step()
            print(loss.item())
            X_filled = train.detach().clone().double()
            X_filled[mask.bool()] = imps
            X_imputed = X_filled.detach().cpu().numpy()

            mae = calc_mae(X_imputed, np.nan_to_num(X_all), all_mask.detach().cpu().numpy())
            mse = calc_mse(X_imputed, np.nan_to_num(X_all), all_mask.detach().cpu().numpy())
            mre = calc_mre(X_imputed, np.nan_to_num(X_all), all_mask.detach().cpu().numpy())

            maes.append(mae)
            mses.append(mse)
            mres.append(mre)
            val_mae = calc_mae(X_imputed, np.nan_to_num(X_all), val_mask.detach().cpu().numpy())
            val_mse = calc_mse(X_imputed, np.nan_to_num(X_all), val_mask.detach().cpu().numpy())
            val_mre = calc_mre(X_imputed, np.nan_to_num(X_all), val_mask.detach().cpu().numpy())
            print("tick:", tick, "val_MAE:", round(val_mae, 5), "val_MSE:", round(val_mse, 5),
                  "test_MAE:", round(mae, 5), "test_MSE:", round(mse, 5))
            if i > 0 and (val_mae < min(val_maes)):
                tick = 0

            val_maes.append(val_mae)
            val_mses.append(val_mse)
            val_mres.append(val_mre)
            self.history_.append({  # ADDED (recording only)
                "epoch": i, "loss": float(loss.item()),
                "val_mae": float(val_mae), "val_mse": float(val_mse),
                "test_mae": float(mae), "test_mse": float(mse),
            })
            if tick > 10:
                break
                self.lr = self.lr / 2
                optimizer = self.opt([imps], lr=self.lr)

        temp = []
        for i in range(len(val_maes)):
            temp.append(val_maes[i] / max(val_maes) + val_mses[i] / max(val_mses))

        loc = temp.index(min(temp))
        return X_filled.detach().cpu().numpy(), maes[loc], mses[loc], mres[loc]
