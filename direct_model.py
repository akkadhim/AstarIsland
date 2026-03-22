"""
Direct learned predictor built from finished rounds.

This is a non-parametric cell model:
- Extract local features from the initial state
- Aggregate historical outcome distributions into buckets
- Back off to simpler buckets when a fine bucket is sparse
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

from fast_sim import T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS, T_PORT, T_SETTLE
from neural_cell_model import build_neural_prediction

TRAIN_DIR = Path("/workspace/checkpoints/training_data")
MODEL_CACHE = Path("/workspace/checkpoints/direct_model_stats.json")

_MODEL_STATS = None
_MODEL_MTIME = None


def _distance_bin(dist: np.ndarray) -> np.ndarray:
    bins = np.array([1.0, 2.0, 3.0, 5.0, 8.0], dtype=np.float32)
    return np.digitize(dist, bins, right=False).astype(np.int8)


def _count_bin(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 4).astype(np.int8)


def _feature_maps(initial_grid: np.ndarray, settlements: Optional[list]) -> Dict[str, np.ndarray]:
    H, W = initial_grid.shape
    sett_mask = np.zeros((H, W), dtype=np.uint8)
    if settlements:
        for s in settlements:
            if s.get("alive", True):
                y, x = int(s["y"]), int(s["x"])
                if 0 <= y < H and 0 <= x < W:
                    sett_mask[y, x] = 1

    kernel = np.ones((3, 3), dtype=np.int16)
    coast = (distance_transform_edt((initial_grid != T_OCEAN).astype(np.uint8)) <= 1.5).astype(np.int8)
    dist_bin = _distance_bin(distance_transform_edt(1 - sett_mask))
    settle_near = _count_bin(convolve(sett_mask.astype(np.int16), kernel, mode="constant", cval=0))
    forest_near = _count_bin(convolve((initial_grid == T_FOREST).astype(np.int16), kernel, mode="constant", cval=0))
    ocean_near = _count_bin(convolve((initial_grid == T_OCEAN).astype(np.int16), kernel, mode="constant", cval=0))
    mountain_near = _count_bin(convolve((initial_grid == T_MOUNTAIN).astype(np.int16), kernel, mode="constant", cval=0))
    plains_near = _count_bin(convolve((initial_grid == T_PLAINS).astype(np.int16), kernel, mode="constant", cval=0))
    settle_count_bin = min(int(sett_mask.sum() // 5), 5)

    return {
        "coast": coast,
        "dist_bin": dist_bin,
        "settle_near": settle_near,
        "forest_near": forest_near,
        "ocean_near": ocean_near,
        "mountain_near": mountain_near,
        "plains_near": plains_near,
        "settle_count_bin": np.full((H, W), settle_count_bin, dtype=np.int8),
    }


def _bucket_keys(initial_grid: np.ndarray, features: Dict[str, np.ndarray], y: int, x: int) -> List[str]:
    t = int(initial_grid[y, x])
    db = int(features["dist_bin"][y, x])
    c = int(features["coast"][y, x])
    sn = int(features["settle_near"][y, x])
    fn = int(features["forest_near"][y, x])
    on = int(features["ocean_near"][y, x])
    mn = int(features["mountain_near"][y, x])
    pn = int(features["plains_near"][y, x])
    sc = int(features["settle_count_bin"][y, x])
    return [
        f"{t}:{db}:{c}:{sn}:{fn}:{on}:{mn}:{pn}:{sc}",
        f"{t}:{db}:{c}:{sn}:{fn}:{on}:{sc}",
        f"{t}:{db}:{c}:{sn}:{fn}:{sc}",
        f"{t}:{db}:{c}:{sn}:{sc}",
        f"{t}:{db}:{c}:{sc}",
        f"{t}:{db}:{c}",
        f"{t}:{db}",
        f"{t}",
    ]


def _compute_stats() -> dict:
    terrain_stats = {}
    bucket_stats = {}
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        raise FileNotFoundError("No training data found for direct model")

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        grid = d["initial_grid"].astype(np.int16)
        gt = d["ground_truth"].astype(np.float32)
        settlements = list(d["settlements"])
        H, W = grid.shape
        feats = _feature_maps(grid, settlements)

        for y in range(H):
            for x in range(W):
                t = int(grid[y, x])
                terrain_key = str(t)
                terrain_entry = terrain_stats.setdefault(terrain_key, {"sum": np.zeros(6), "count": 0})
                terrain_entry["sum"] += gt[y, x]
                terrain_entry["count"] += 1

                key = _bucket_keys(grid, feats, y, x)[0]
                entry = bucket_stats.setdefault(key, {"sum": np.zeros(6), "count": 0})
                entry["sum"] += gt[y, x]
                entry["count"] += 1

    return {
        "terrain": {
            k: {"mean": (v["sum"] / v["count"]).tolist(), "count": v["count"]}
            for k, v in terrain_stats.items()
        },
        "bucket": {
            k: {"mean": (v["sum"] / v["count"]).tolist(), "count": v["count"]}
            for k, v in bucket_stats.items()
        },
    }


def _load_stats() -> Optional[dict]:
    global _MODEL_STATS, _MODEL_MTIME
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None

    latest_mtime = max(fp.stat().st_mtime for fp in files)
    if _MODEL_STATS is not None and _MODEL_MTIME == latest_mtime:
        return _MODEL_STATS

    if MODEL_CACHE.exists():
        try:
            cached = json.loads(MODEL_CACHE.read_text())
            if cached.get("training_mtime") == latest_mtime:
                _MODEL_STATS = cached["stats"]
                _MODEL_MTIME = latest_mtime
                return _MODEL_STATS
        except Exception:
            pass

    stats = _compute_stats()
    MODEL_CACHE.write_text(json.dumps({"training_mtime": latest_mtime, "stats": stats}))
    _MODEL_STATS = stats
    _MODEL_MTIME = latest_mtime
    return stats


def build_direct_prediction(initial_grid: np.ndarray, settlements: Optional[list] = None) -> Optional[np.ndarray]:
    stats = _load_stats()
    if stats is None:
        return None

    H, W = initial_grid.shape
    out = np.zeros((H, W, 6), dtype=np.float32)
    feats = _feature_maps(initial_grid, settlements)
    bucket_stats = stats["bucket"]
    terrain_stats = stats["terrain"]

    for y in range(H):
        for x in range(W):
            vec = None
            for key in _bucket_keys(initial_grid, feats, y, x):
                entry = bucket_stats.get(key)
                if entry and entry["count"] >= 25:
                    vec = np.array(entry["mean"], dtype=np.float32)
                    break
            if vec is None:
                entry = terrain_stats.get(str(int(initial_grid[y, x])))
                if entry:
                    vec = np.array(entry["mean"], dtype=np.float32)
                else:
                    vec = np.full(6, 1 / 6, dtype=np.float32)
            out[y, x] = vec

    neural = build_neural_prediction(initial_grid, settlements)
    if neural is not None:
        out = 0.55 * neural + 0.45 * out
        out /= out.sum(axis=-1, keepdims=True)
    return out.astype(np.float32)
