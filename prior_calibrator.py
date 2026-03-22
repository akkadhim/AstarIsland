"""
Learned calibrator for static prior + direct model prior.

This module trains a small neural calibrator that sees:
- static prior probabilities
- direct model probabilities
- their difference
- local terrain/context features

and predicts a better per-cell class distribution. It is optional at inference
time and falls back cleanly if no fresh checkpoint is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from fast_sim import T_EMPTY, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS, T_PORT, T_RUIN, T_SETTLE

TRAIN_DIR = Path("/workspace/checkpoints/training_data")
MODEL_PATH = Path("/workspace/checkpoints/prior_calibrator.pt")
TERRAIN_CODES = np.array([T_EMPTY, T_SETTLE, T_PORT, T_RUIN, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS], dtype=np.int16)

_CACHE = None
_CACHE_MTIME = None


class PriorCalibrator(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _training_mtime() -> Optional[float]:
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None
    return max(fp.stat().st_mtime for fp in files)


def _terrain_one_hot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(grid.shape + (len(TERRAIN_CODES),), dtype=np.float32)
    for i, code in enumerate(TERRAIN_CODES):
        out[..., i] = (grid == code).astype(np.float32)
    return out


def build_calibration_features(initial_grid: np.ndarray,
                               settlements: Optional[list],
                               static_prior: np.ndarray,
                               direct_prior: np.ndarray) -> np.ndarray:
    H, W = initial_grid.shape
    terrain = _terrain_one_hot(initial_grid)
    sett_mask = np.zeros((H, W), dtype=np.float32)
    if settlements:
        for s in settlements:
            if s.get("alive", True):
                y, x = int(s["y"]), int(s["x"])
                if 0 <= y < H and 0 <= x < W:
                    sett_mask[y, x] = 1.0

    dist_to_sett = distance_transform_edt(1 - sett_mask.astype(np.uint8)).astype(np.float32)
    dist_to_ocean = distance_transform_edt((initial_grid != T_OCEAN).astype(np.uint8)).astype(np.float32)
    coast = (dist_to_ocean <= 1.5).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    x_norm = xx.astype(np.float32) / max(W - 1, 1)
    y_norm = yy.astype(np.float32) / max(H - 1, 1)

    feats = np.concatenate([
        static_prior.astype(np.float32),
        direct_prior.astype(np.float32),
        (direct_prior - static_prior).astype(np.float32),
        terrain,
        coast[..., None],
        np.clip(dist_to_sett / 12.0, 0.0, 1.0)[..., None],
        np.clip(dist_to_ocean / 8.0, 0.0, 1.0)[..., None],
        x_norm[..., None],
        y_norm[..., None],
    ], axis=-1)
    return feats.reshape(H * W, -1).astype(np.float32)


def _load_checkpoint() -> Optional[dict]:
    global _CACHE, _CACHE_MTIME
    tm = _training_mtime()
    if tm is None or not MODEL_PATH.exists():
        return None
    if _CACHE is not None and _CACHE_MTIME == tm:
        return _CACHE
    try:
        payload = torch.load(MODEL_PATH, map_location="cpu")
        if payload.get("training_mtime") != tm:
            return None
        _CACHE = payload
        _CACHE_MTIME = tm
        return payload
    except Exception:
        return None


def apply_prior_calibrator(initial_grid: np.ndarray,
                           settlements: Optional[list],
                           static_prior: np.ndarray,
                           direct_prior: np.ndarray) -> Optional[np.ndarray]:
    payload = _load_checkpoint()
    if payload is None:
        return None

    feats = build_calibration_features(initial_grid, settlements, static_prior, direct_prior)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    std = np.asarray(payload["feature_std"], dtype=np.float32)
    feats = (feats - mean) / std

    model = PriorCalibrator(in_dim=feats.shape[1], hidden_dim=int(payload.get("hidden_dim", 128)))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(feats)
        probs = torch.softmax(model(x), dim=-1).cpu().numpy().astype(np.float32)

    H, W = initial_grid.shape
    return probs.reshape(H, W, 6)


def train_prior_calibrator(epochs: int = 80, batch_size: int = 4096, lr: float = 2e-3,
                           hidden_dim: int = 128, device: Optional[str] = None) -> Optional[dict]:
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None

    # Local imports avoid circular imports at module import time.
    from prediction import build_static_prior
    from direct_model import build_direct_prediction

    all_x = []
    all_y = []
    all_w = []
    all_rounds = []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        grid = d["initial_grid"].astype(np.int16)
        gt = d["ground_truth"].astype(np.float32)
        settlements = list(d["settlements"])
        static = build_static_prior(grid, settlements)
        direct = build_direct_prediction(grid, settlements)
        feats = build_calibration_features(grid, settlements, static, direct)
        y = gt.reshape(-1, 6).astype(np.float32)
        ent = -(np.clip(y, 1e-9, 1.0) * np.log(np.clip(y, 1e-9, 1.0))).sum(axis=1).astype(np.float32)
        w = (0.10 + ent).astype(np.float32)
        round_num = int(fp.stem.split("_")[0][1:])

        all_x.append(feats)
        all_y.append(y)
        all_w.append(w)
        all_rounds.append(np.full(feats.shape[0], round_num, dtype=np.int16))

    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    Wt = np.concatenate(all_w, axis=0)
    R = np.concatenate(all_rounds, axis=0)

    val_mask = (R % 5) == 0
    if val_mask.sum() < 5000:
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[::10] = True
    train_mask = ~val_mask

    feat_mean = X[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    feat_std = X[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-4)
    X = (X - feat_mean) / feat_std

    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.device_count()-1}"
        else:
            device = "cpu"
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    model = PriorCalibrator(in_dim=X.shape[1], hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_train = torch.from_numpy(X[train_mask]).to(dev)
    y_train = torch.from_numpy(Y[train_mask]).to(dev)
    w_train = torch.from_numpy(Wt[train_mask]).to(dev)
    x_val = torch.from_numpy(X[val_mask]).to(dev)
    y_val = torch.from_numpy(Y[val_mask]).to(dev)
    w_val = torch.from_numpy(Wt[val_mask]).to(dev)

    best_state = None
    best_val = float("inf")
    patience = 10
    stale = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=dev)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            wb = w_train[idx]
            logits = model(xb)
            logp = torch.log_softmax(logits, dim=-1)
            ce = -(yb * logp).sum(dim=-1)
            loss = (ce * wb).sum() / wb.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            logp = torch.log_softmax(logits, dim=-1)
            ce = -(y_val * logp).sum(dim=-1)
            val_loss = float((ce * w_val).sum().item() / w_val.sum().clamp(min=1e-6).item())

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    payload = {
        "training_mtime": _training_mtime(),
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "hidden_dim": hidden_dim,
        "state_dict": best_state,
        "val_loss": best_val,
        "train_examples": int(train_mask.sum()),
        "val_examples": int(val_mask.sum()),
        "device": str(dev),
    }
    torch.save(payload, MODEL_PATH)
    return payload
