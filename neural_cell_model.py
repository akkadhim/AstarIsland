"""
Neural per-cell predictor trained on completed rounds.

This is a larger learned model than the bucket-based direct predictor:
- input: engineered local + global map features per cell
- target: soft 6-class ground-truth distribution per cell
- loss: entropy-weighted soft cross-entropy

The model is optional at inference time. If the checkpoint is missing or stale,
the rest of the system falls back to the safer statistical priors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import convolve, distance_transform_edt

from fast_sim import T_EMPTY, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS, T_PORT, T_RUIN, T_SETTLE

TRAIN_DIR = Path("/workspace/checkpoints/training_data")
MODEL_PATH = Path("/workspace/checkpoints/neural_cell_model.pt")

TERRAIN_CODES = np.array([T_EMPTY, T_SETTLE, T_PORT, T_RUIN, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS], dtype=np.int16)

_CACHE = None
_CACHE_MTIME = None


class CellMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 160):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 6),
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


def _count_bin_float(arr: np.ndarray, max_value: float) -> np.ndarray:
    return np.clip(arr.astype(np.float32) / max_value, 0.0, 1.0)


def build_feature_matrix(initial_grid: np.ndarray, settlements: Optional[list]) -> np.ndarray:
    H, W = initial_grid.shape
    sett_mask = np.zeros((H, W), dtype=np.uint8)
    if settlements:
        for s in settlements:
            if s.get("alive", True):
                y, x = int(s["y"]), int(s["x"])
                if 0 <= y < H and 0 <= x < W:
                    sett_mask[y, x] = 1

    terrain_oh = _terrain_one_hot(initial_grid)
    dist_to_sett = distance_transform_edt(1 - sett_mask).astype(np.float32)
    dist_to_ocean = distance_transform_edt((initial_grid != T_OCEAN).astype(np.uint8)).astype(np.float32)
    coast = (dist_to_ocean <= 1.5).astype(np.float32)

    k3 = np.ones((3, 3), dtype=np.int16)
    k5 = np.ones((5, 5), dtype=np.int16)
    k3[1, 1] = 0
    k5[2, 2] = 0

    forest3 = convolve((initial_grid == T_FOREST).astype(np.int16), k3, mode="constant", cval=0)
    ocean3 = convolve((initial_grid == T_OCEAN).astype(np.int16), k3, mode="constant", cval=0)
    mountain3 = convolve((initial_grid == T_MOUNTAIN).astype(np.int16), k3, mode="constant", cval=0)
    plains3 = convolve((initial_grid == T_PLAINS).astype(np.int16), k3, mode="constant", cval=0)
    settle3 = convolve(sett_mask.astype(np.int16), k3, mode="constant", cval=0)

    forest5 = convolve((initial_grid == T_FOREST).astype(np.int16), k5, mode="constant", cval=0)
    ocean5 = convolve((initial_grid == T_OCEAN).astype(np.int16), k5, mode="constant", cval=0)
    mountain5 = convolve((initial_grid == T_MOUNTAIN).astype(np.int16), k5, mode="constant", cval=0)
    plains5 = convolve((initial_grid == T_PLAINS).astype(np.int16), k5, mode="constant", cval=0)
    settle5 = convolve(sett_mask.astype(np.int16), k5, mode="constant", cval=0)

    yy, xx = np.mgrid[0:H, 0:W]
    x_norm = xx.astype(np.float32) / max(W - 1, 1)
    y_norm = yy.astype(np.float32) / max(H - 1, 1)

    cell_count = float(H * W)
    global_feats = np.array([
        float((initial_grid == T_FOREST).sum()) / cell_count,
        float((initial_grid == T_OCEAN).sum()) / cell_count,
        float((initial_grid == T_MOUNTAIN).sum()) / cell_count,
        float((initial_grid == T_PLAINS).sum()) / cell_count,
        float(coast.sum()) / cell_count,
        float(sett_mask.sum()) / cell_count,
    ], dtype=np.float32)
    global_map = np.broadcast_to(global_feats, (H, W, len(global_feats))).astype(np.float32)

    features = np.concatenate([
        terrain_oh,
        x_norm[..., None],
        y_norm[..., None],
        coast[..., None],
        np.clip(dist_to_sett / 12.0, 0.0, 1.0)[..., None],
        np.exp(-dist_to_sett / 3.0)[..., None].astype(np.float32),
        np.clip(dist_to_ocean / 8.0, 0.0, 1.0)[..., None],
        _count_bin_float(forest3, 8.0)[..., None],
        _count_bin_float(ocean3, 8.0)[..., None],
        _count_bin_float(mountain3, 8.0)[..., None],
        _count_bin_float(plains3, 8.0)[..., None],
        _count_bin_float(settle3, 8.0)[..., None],
        _count_bin_float(forest5, 24.0)[..., None],
        _count_bin_float(ocean5, 24.0)[..., None],
        _count_bin_float(mountain5, 24.0)[..., None],
        _count_bin_float(plains5, 24.0)[..., None],
        _count_bin_float(settle5, 24.0)[..., None],
        global_map,
    ], axis=-1)
    return features.reshape(H * W, -1).astype(np.float32)


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


def build_neural_prediction(initial_grid: np.ndarray, settlements: Optional[list] = None) -> Optional[np.ndarray]:
    payload = _load_checkpoint()
    if payload is None:
        return None

    features = build_feature_matrix(initial_grid, settlements)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    std = np.asarray(payload["feature_std"], dtype=np.float32)
    feats = (features - mean) / std

    model = CellMLP(in_dim=feats.shape[1], hidden_dim=int(payload.get("hidden_dim", 160)))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(feats)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

    H, W = initial_grid.shape
    return probs.reshape(H, W, 6)


def train_neural_cell_model(epochs: int = 60, batch_size: int = 4096, lr: float = 2e-3,
                            hidden_dim: int = 160, device: Optional[str] = None) -> Optional[dict]:
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None

    all_x = []
    all_y = []
    all_w = []
    all_rounds = []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        grid = d["initial_grid"].astype(np.int16)
        gt = d["ground_truth"].astype(np.float32)
        settlements = list(d["settlements"])
        x = build_feature_matrix(grid, settlements)
        y = gt.reshape(-1, 6).astype(np.float32)
        ent = -(np.clip(y, 1e-9, 1.0) * np.log(np.clip(y, 1e-9, 1.0))).sum(axis=1).astype(np.float32)
        w = (0.10 + ent).astype(np.float32)
        round_num = int(fp.stem.split("_")[0][1:])

        all_x.append(x)
        all_y.append(y)
        all_w.append(w)
        all_rounds.append(np.full(x.shape[0], round_num, dtype=np.int16))

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
    model = CellMLP(in_dim=X.shape[1], hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_train = torch.from_numpy(X[train_mask]).to(dev)
    y_train = torch.from_numpy(Y[train_mask]).to(dev)
    w_train = torch.from_numpy(Wt[train_mask]).to(dev)
    x_val = torch.from_numpy(X[val_mask]).to(dev)
    y_val = torch.from_numpy(Y[val_mask]).to(dev)
    w_val = torch.from_numpy(Wt[val_mask]).to(dev)

    best_state = None
    best_val = float("inf")
    patience = 8
    stale = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=dev)
        epoch_loss = 0.0
        total_w = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            wb = w_train[idx]
            logits = model(xb)
            logp = torch.log_softmax(logits, dim=-1)
            loss_vec = -(yb * logp).sum(dim=-1)
            loss = (loss_vec * wb).sum() / wb.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float((loss_vec * wb).sum().item())
            total_w += float(wb.sum().item())

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            logp = torch.log_softmax(logits, dim=-1)
            val_vec = -(y_val * logp).sum(dim=-1)
            val_loss = float((val_vec * w_val).sum().item() / w_val.sum().clamp(min=1e-6).item())

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


if __name__ == "__main__":
    info = train_neural_cell_model()
    if info is None:
        print("No training data found")
    else:
        print(json.dumps({
            "val_loss": info["val_loss"],
            "train_examples": info["train_examples"],
            "val_examples": info["val_examples"],
        }, indent=2))
