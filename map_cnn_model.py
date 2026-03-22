"""
Map-level CNN prior for Astar Island.

Unlike the cell MLP, this model sees the whole 40x40 map at once and can learn
larger spatial structure such as mountain chains, coast bands, and settlement
frontiers. It is optional at inference time and used only when a fresh
checkpoint is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from fast_sim import T_EMPTY, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS, T_PORT, T_RUIN, T_SETTLE

TRAIN_DIR = Path("/workspace/checkpoints/training_data")
MODEL_PATH = Path("/workspace/checkpoints/map_cnn_model.pt")
TERRAIN_CODES = np.array([T_EMPTY, T_SETTLE, T_PORT, T_RUIN, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS], dtype=np.int16)

_CACHE = None
_CACHE_MTIME = None


class ResidualDilatedBlock(torch.nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            torch.nn.GroupNorm(8, channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            torch.nn.GroupNorm(8, channels),
        )
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MapCNN(torch.nn.Module):
    def __init__(self, in_channels: int, width: int = 64):
        super().__init__()
        dilations = [1, 2, 4, 8, 4, 2, 1]
        layers = [
            torch.nn.Conv2d(in_channels, width, 3, padding=1),
            torch.nn.GroupNorm(8, width),
            torch.nn.GELU(),
        ]
        layers.extend(ResidualDilatedBlock(width, d) for d in dilations)
        layers.append(torch.nn.Conv2d(width, 6, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _training_mtime() -> Optional[float]:
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None
    return max(fp.stat().st_mtime for fp in files)


def build_input_tensor(initial_grid: np.ndarray, settlements: Optional[list]) -> np.ndarray:
    H, W = initial_grid.shape
    terrain = np.zeros((len(TERRAIN_CODES), H, W), dtype=np.float32)
    for i, code in enumerate(TERRAIN_CODES):
        terrain[i] = (initial_grid == code).astype(np.float32)

    sett_mask = np.zeros((H, W), dtype=np.float32)
    port_mask = np.zeros((H, W), dtype=np.float32)
    if settlements:
        for s in settlements:
            if s.get("alive", True):
                y, x = int(s["y"]), int(s["x"])
                if 0 <= y < H and 0 <= x < W:
                    sett_mask[y, x] = 1.0
                    if s.get("has_port", False):
                        port_mask[y, x] = 1.0

    dist_to_sett = distance_transform_edt(1 - sett_mask.astype(np.uint8)).astype(np.float32)
    dist_to_ocean = distance_transform_edt((initial_grid != T_OCEAN).astype(np.uint8)).astype(np.float32)
    coast = (dist_to_ocean <= 1.5).astype(np.float32)
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
    global_maps = np.broadcast_to(global_feats[:, None, None], (len(global_feats), H, W)).astype(np.float32)

    return np.concatenate([
        terrain,
        sett_mask[None],
        port_mask[None],
        coast[None],
        np.clip(dist_to_sett / 12.0, 0.0, 1.0)[None],
        np.clip(dist_to_ocean / 8.0, 0.0, 1.0)[None],
        x_norm[None],
        y_norm[None],
        global_maps,
    ], axis=0).astype(np.float32)


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


def build_map_prediction(initial_grid: np.ndarray, settlements: Optional[list] = None) -> Optional[np.ndarray]:
    payload = _load_checkpoint()
    if payload is None:
        return None

    x = build_input_tensor(initial_grid, settlements)[None]
    model = MapCNN(in_channels=x.shape[1], width=int(payload.get("width", 64)))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(x)
        probs = torch.softmax(model(xt), dim=1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return probs


def _augment(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1]); w = torch.flip(w, dims=[-1])
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2]); w = torch.flip(w, dims=[-2])
    if torch.rand(()) < 0.5:
        x = x.transpose(-1, -2); y = y.transpose(-1, -2); w = w.transpose(-1, -2)
    return x, y, w


def train_map_cnn_model(epochs: int = 120, batch_size: int = 8, lr: float = 1e-3,
                        width: int = 64, device: Optional[str] = None) -> Optional[dict]:
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None

    xs = []
    ys = []
    ws = []
    rounds = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        grid = d["initial_grid"].astype(np.int16)
        gt = d["ground_truth"].astype(np.float32)
        settlements = list(d["settlements"])
        x = build_input_tensor(grid, settlements)
        y = gt.transpose(2, 0, 1).astype(np.float32)
        w = -(np.clip(gt, 1e-9, 1.0) * np.log(np.clip(gt, 1e-9, 1.0))).sum(axis=-1).astype(np.float32)
        w = 0.10 + w
        xs.append(x)
        ys.append(y)
        ws.append(w[None])
        rounds.append(int(fp.stem.split("_")[0][1:]))

    X = torch.from_numpy(np.stack(xs, axis=0))
    Y = torch.from_numpy(np.stack(ys, axis=0))
    Wt = torch.from_numpy(np.stack(ws, axis=0))
    R = np.array(rounds)

    val_mask = (R % 5) == 0
    if val_mask.sum() < 8:
        val_mask = np.zeros(len(R), dtype=bool)
        val_mask[::5] = True
    train_mask = ~val_mask

    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.device_count()-1}"
        else:
            device = "cpu"
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    model = MapCNN(in_channels=X.shape[1], width=width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_train = X[train_mask].to(dev)
    y_train = Y[train_mask].to(dev)
    w_train = Wt[train_mask].to(dev)
    x_val = X[val_mask].to(dev)
    y_val = Y[val_mask].to(dev)
    w_val = Wt[val_mask].to(dev)

    best_state = None
    best_val = float("inf")
    stale = 0
    patience = 12

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=dev)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            wb = w_train[idx]
            xb, yb, wb = _augment(xb, yb, wb)
            logits = model(xb)
            logp = torch.log_softmax(logits, dim=1)
            ce = -(yb * logp).sum(dim=1)
            loss = (ce * wb.squeeze(1)).sum() / wb.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            logp = torch.log_softmax(logits, dim=1)
            ce = -(y_val * logp).sum(dim=1)
            val_loss = float((ce * w_val.squeeze(1)).sum().item() / w_val.sum().clamp(min=1e-6).item())

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
        "state_dict": best_state,
        "width": width,
        "val_loss": best_val,
        "train_examples": int(train_mask.sum()),
        "val_examples": int(val_mask.sum()),
        "device": str(dev),
    }
    torch.save(payload, MODEL_PATH)
    return payload


if __name__ == "__main__":
    info = train_map_cnn_model()
    if info is None:
        print("No training data found")
    else:
        print({
            "val_loss": info["val_loss"],
            "train_examples": info["train_examples"],
            "val_examples": info["val_examples"],
            "device": info["device"],
        })
