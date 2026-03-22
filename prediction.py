"""
Build final prediction tensors and submit.

Key ideas:
1. Learn a strong empirical prior from completed rounds
2. Blend simulation with that prior conservatively
3. Let direct observations override the model more aggressively
4. Always apply a minimum floor to avoid KL blowups
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional
from scipy.ndimage import distance_transform_edt
from fast_sim import SimParams, T_MOUNTAIN, T_OCEAN, T_PLAINS, T_FOREST, T_SETTLE, T_PORT
from multi_gpu import run_sequential_multi_gpu as run_multi_gpu
from direct_model import build_direct_prediction
from prior_calibrator import apply_prior_calibrator

# Minimum probability floor to prevent KL divergence blowup
# Lower floor = better score (less KL penalty on confident correct predictions)
# 0.001 gains ~4-6pts vs 0.01 across all tested rounds
MIN_PROB = 0.001
TRAIN_DIR = Path("/workspace/checkpoints/training_data")
PRIOR_CACHE = Path("/workspace/checkpoints/empirical_prior_stats.json")

_PRIOR_STATS = None
_PRIOR_MTIME = None


def _default_static_prior(initial_grid: np.ndarray, settlements: list = None) -> np.ndarray:
    """Hard-coded fallback prior used when no training cache is available."""
    H, W = initial_grid.shape
    probs = np.zeros((H, W, 6), dtype=np.float32)

    g = initial_grid
    probs[:] = [0.82, 0.09, 0.01, 0.01, 0.05, 0.02]

    mask = (g == T_OCEAN)
    probs[mask] = [0.97, 0.01, 0.005, 0.005, 0.005, 0.005]

    mask = (g == T_MOUNTAIN)
    probs[mask] = [0.005, 0.005, 0.005, 0.005, 0.005, 0.975]

    mask = (g == T_FOREST)
    probs[mask] = [0.09, 0.15, 0.01, 0.016, 0.73, 0.004]

    mask = (g == T_PLAINS)
    probs[mask] = [0.80, 0.14, 0.01, 0.015, 0.039, 0.005]

    mask = (g == T_SETTLE)
    probs[mask] = [0.45, 0.31, 0.01, 0.030, 0.21, 0.004]

    mask = (g == T_PORT)
    probs[mask] = [0.49, 0.10, 0.16, 0.021, 0.236, 0.003]

    if settlements:
        sett_mask = np.zeros((H, W), dtype=np.float32)
        for sv in settlements:
            if sv.get("alive", True):
                sy, sx = int(sv["y"]), int(sv["x"])
                if 0 <= sy < H and 0 <= sx < W:
                    sett_mask[sy, sx] = 1.0

        dist_to_sett = distance_transform_edt(1 - sett_mask)

        plains_nearby = (g == T_PLAINS) & (dist_to_sett <= 6)
        if plains_nearby.any():
            boost = np.clip(1.0 - dist_to_sett / 6.0, 0, 1)[plains_nearby]
            probs[plains_nearby, 0] -= boost * 0.12
            probs[plains_nearby, 1] += boost * 0.10
            probs[plains_nearby] = np.clip(probs[plains_nearby], 0.005, None)
            probs[plains_nearby] /= probs[plains_nearby].sum(axis=1, keepdims=True)

        forest_nearby = (g == T_FOREST) & (dist_to_sett <= 5)
        if forest_nearby.any():
            boost = np.clip(1.0 - dist_to_sett / 5.0, 0, 1)[forest_nearby]
            probs[forest_nearby, 0] -= boost * 0.05
            probs[forest_nearby, 1] += boost * 0.05
            probs[forest_nearby] = np.clip(probs[forest_nearby], 0.005, None)
            probs[forest_nearby] /= probs[forest_nearby].sum(axis=1, keepdims=True)

    return probs


def _prior_distance_bin(dist: np.ndarray) -> np.ndarray:
    bins = np.array([1.0, 2.0, 3.0, 5.0, 8.0], dtype=np.float32)
    return np.digitize(dist, bins, right=False).astype(np.int8)


def _compute_empirical_prior_stats() -> dict:
    terrain_stats = {}
    bucket_stats = {}
    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        raise FileNotFoundError("No training data found for empirical prior")

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        grid = d["initial_grid"].astype(np.int16)
        gt = d["ground_truth"].astype(np.float32)
        settlements = list(d["settlements"])
        H, W = grid.shape

        sett_mask = np.zeros((H, W), dtype=np.uint8)
        for s in settlements:
            if s.get("alive", True):
                y, x = int(s["y"]), int(s["x"])
                if 0 <= y < H and 0 <= x < W:
                    sett_mask[y, x] = 1

        dist = distance_transform_edt(1 - sett_mask)
        dist_bin = _prior_distance_bin(dist)
        coast = (distance_transform_edt((grid != T_OCEAN).astype(np.uint8)) <= 1.5).astype(np.int8)

        for y in range(H):
            for x in range(W):
                t = int(grid[y, x])
                terrain_key = str(t)
                terrain_entry = terrain_stats.setdefault(terrain_key, {"sum": np.zeros(6), "count": 0})
                terrain_entry["sum"] += gt[y, x]
                terrain_entry["count"] += 1

                bucket_key = f"{t}:{int(dist_bin[y, x])}:{int(coast[y, x])}"
                bucket_entry = bucket_stats.setdefault(bucket_key, {"sum": np.zeros(6), "count": 0})
                bucket_entry["sum"] += gt[y, x]
                bucket_entry["count"] += 1

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


def _load_empirical_prior_stats() -> Optional[dict]:
    global _PRIOR_STATS, _PRIOR_MTIME
    if not TRAIN_DIR.exists():
        return None

    files = sorted(TRAIN_DIR.glob("r*_s*.npz"))
    if not files:
        return None

    latest_mtime = max(fp.stat().st_mtime for fp in files)
    if _PRIOR_STATS is not None and _PRIOR_MTIME == latest_mtime:
        return _PRIOR_STATS

    if PRIOR_CACHE.exists():
        try:
            cached = json.loads(PRIOR_CACHE.read_text())
            if cached.get("training_mtime") == latest_mtime:
                _PRIOR_STATS = cached["stats"]
                _PRIOR_MTIME = latest_mtime
                return _PRIOR_STATS
        except Exception:
            pass

    stats = _compute_empirical_prior_stats()
    PRIOR_CACHE.write_text(json.dumps({"training_mtime": latest_mtime, "stats": stats}))
    _PRIOR_STATS = stats
    _PRIOR_MTIME = latest_mtime
    return stats


def build_static_prior(initial_grid: np.ndarray, settlements: list = None) -> np.ndarray:
    """
    Build a strong prior from historical training data when available.
    Falls back to a hand-tuned prior if no training cache exists yet.
    """
    stats = _load_empirical_prior_stats()
    if stats is None:
        return _default_static_prior(initial_grid, settlements)

    H, W = initial_grid.shape
    probs = np.zeros((H, W, 6), dtype=np.float32)

    sett_mask = np.zeros((H, W), dtype=np.uint8)
    if settlements:
        for s in settlements:
            if s.get("alive", True):
                y, x = int(s["y"]), int(s["x"])
                if 0 <= y < H and 0 <= x < W:
                    sett_mask[y, x] = 1

    dist = distance_transform_edt(1 - sett_mask)
    dist_bin = _prior_distance_bin(dist)
    coast = (distance_transform_edt((initial_grid != T_OCEAN).astype(np.uint8)) <= 1.5).astype(np.int8)

    terrain_stats = stats["terrain"]
    bucket_stats = stats["bucket"]
    fallback = _default_static_prior(initial_grid, settlements)

    for y in range(H):
        for x in range(W):
            t = int(initial_grid[y, x])
            db = int(dist_bin[y, x])
            c = int(coast[y, x])
            candidates = [
                f"{t}:{db}:{c}",
                f"{t}:{db}:0",
                f"{t}:0:{c}",
                f"{t}:0:0",
            ]
            vec = None
            for key in candidates:
                entry = bucket_stats.get(key)
                if entry and entry["count"] >= 50:
                    vec = np.array(entry["mean"], dtype=np.float32)
                    break
            if vec is None:
                terrain_entry = terrain_stats.get(str(t))
                if terrain_entry:
                    vec = np.array(terrain_entry["mean"], dtype=np.float32)
                else:
                    vec = fallback[y, x]
            probs[y, x] = vec

    return probs.astype(np.float32)


def choose_sim_weight(sim_probs: np.ndarray,
                      static_prior: np.ndarray,
                      obs_probs: Optional[np.ndarray] = None,
                      obs_weights: Optional[np.ndarray] = None,
                      base_sim_weight: float = 0.24) -> float:
    """
    Choose simulation weight adaptively.
    When observations available: trust whichever source (sim or prior) is closer to obs.
    R16: prior over-predicted settlement → sim was better → higher sim weight needed.
    R17: sim under-predicted settlement → prior was better → lower sim weight needed.
    """
    sim_mean = sim_probs.mean(axis=(0, 1))
    prior_mean = static_prior.mean(axis=(0, 1))
    drift = (
        1.0 * abs(float(sim_mean[1] - prior_mean[1])) +
        1.5 * abs(float(sim_mean[2] - prior_mean[2])) +
        1.5 * abs(float(sim_mean[3] - prior_mean[3])) +
        0.7 * abs(float(sim_mean[4] - prior_mean[4]))
    )
    weight = base_sim_weight - 0.90 * max(0.0, drift - 0.03)

    if obs_probs is None or obs_weights is None or not np.any(obs_weights > 0):
        return float(np.clip(weight, 0.08, 0.34))

    mask = obs_weights > 0
    obs_civ = float((obs_probs[mask, 1] + obs_probs[mask, 2] + obs_probs[mask, 3]).mean())
    sim_civ = float((sim_probs[mask, 1] + sim_probs[mask, 2] + sim_probs[mask, 3]).mean())
    prior_civ = float((static_prior[mask, 1] + static_prior[mask, 2] + static_prior[mask, 3]).mean())

    sim_mismatch = abs(sim_civ - obs_civ)
    prior_mismatch = abs(prior_civ - obs_civ)
    if sim_mismatch + prior_mismatch < 1e-6:
        return base_sim_weight

    # rel=0 means sim perfect, rel=1 means prior perfect
    rel = sim_mismatch / (sim_mismatch + prior_mismatch)
    # When sim is closer (rel<0.5): boost weight up to 0.50
    # When prior is closer (rel>0.5): reduce weight down to 0.05
    weight += 0.20 - 0.40 * rel
    return float(np.clip(weight, 0.05, 0.50))


def stabilize_rare_classes(probs: np.ndarray,
                           static_prior: np.ndarray,
                           obs_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pull rare classes back toward the learned prior unless observations support them.
    This reduces damage when the simulator invents too many ports/ruins.
    """
    out = probs.copy()
    if obs_weights is None:
        damp = np.full(probs.shape[:2] + (1,), 0.35, dtype=np.float32)
    else:
        damp = (0.35 + 0.45 * np.clip(obs_weights / 3.0, 0.0, 1.0))[:, :, None].astype(np.float32)
    out[:, :, 2:4] = damp * out[:, :, 2:4] + (1.0 - damp) * static_prior[:, :, 2:4]
    out /= out.sum(axis=-1, keepdims=True)
    return out.astype(np.float32)


def build_learned_prior(initial_grid: np.ndarray, settlements: list = None) -> np.ndarray:
    """
    Ensemble the empirical terrain prior with a feature-based direct predictor.
    """
    static_prior = build_static_prior(initial_grid, settlements)
    direct_prior = build_direct_prediction(initial_grid, settlements)
    if direct_prior is None:
        return static_prior

    calibrated = apply_prior_calibrator(initial_grid, settlements, static_prior, direct_prior)
    if calibrated is not None:
        return calibrated.astype(np.float32)

    # Fallback blend if no calibrator checkpoint is available.
    learned = 0.60 * direct_prior + 0.40 * static_prior
    return learned.astype(np.float32)


def _global_rescale(probs: np.ndarray, obs_probs: np.ndarray,
                    obs_weights: np.ndarray, initial_grid: np.ndarray) -> np.ndarray:
    """
    Rescale global class fractions to match observed fractions.
    Strength adapts based on how many dynamic cells we observed —
    more observations = higher confidence = stronger rescaling.
    """
    static_mask = ((initial_grid == T_MOUNTAIN) | (initial_grid == T_OCEAN))
    dynamic = ~static_mask

    if not np.any(obs_weights > 0) or dynamic.sum() == 0:
        return probs

    obs_mask = (obs_weights > 0) & dynamic
    n_obs = obs_mask.sum()
    n_dynamic = dynamic.sum()
    if n_obs < 20:
        return probs

    # Adaptive strength: more observed cells = higher confidence
    # 20 cells → 0.25, 200 cells → 0.65, 500+ cells → 0.80
    coverage = n_obs / max(n_dynamic, 1)
    strength = float(np.clip(0.25 + 0.70 * coverage, 0.25, 0.85))

    obs_fracs = obs_probs[obs_mask].mean(axis=0)
    pred_fracs = probs[dynamic].mean(axis=0)

    scale = np.ones(6, dtype=np.float32)
    for c in range(6):
        if pred_fracs[c] > 0.003:
            raw = obs_fracs[c] / pred_fracs[c]
            scale[c] = 1.0 + strength * (np.clip(raw, 0.05, 8.0) - 1.0)

    out = probs.copy()
    out[dynamic] = out[dynamic] * scale[None, :]
    out[dynamic] /= out[dynamic].sum(axis=-1, keepdims=True)
    return out.astype(np.float32)


def blend_predictions(sim_probs: np.ndarray,
                      obs_probs: Optional[np.ndarray],
                      obs_weights: Optional[np.ndarray],
                      static_prior: np.ndarray,
                      initial_grid: np.ndarray,
                      sim_weight: Optional[float] = None) -> np.ndarray:
    """
    Combine simulation probs, observed probs, and static prior.

    sim_probs: H×W×6 from Monte Carlo
    obs_probs: H×W×6 from direct observation (or None)
    obs_weights: H×W observation counts (or None)
    static_prior: H×W×6 initial state prior
    initial_grid: H×W terrain codes
    sim_weight: weight for simulation vs prior

    Returns H×W×6 final probability tensor.
    """
    # Static cells (ocean/mountain): use prior directly
    static_mask = ((initial_grid == T_MOUNTAIN) | (initial_grid == T_OCEAN))
    static_mask3 = static_mask[:, :, None]

    if sim_weight is None:
        sim_weight = choose_sim_weight(sim_probs, static_prior, obs_probs, obs_weights)

    # Vectorized base blend: sim + prior
    prior_weight = 1.0 - sim_weight
    base_blend = sim_weight * sim_probs + prior_weight * static_prior
    base_blend = stabilize_rare_classes(base_blend, static_prior, obs_weights)

    # Override static cells with prior
    final = np.where(static_mask3, static_prior, base_blend)

    # Incorporate observations: use global rescaling only.
    # Per-cell obs blending hurts with noisy single-sample observations because
    # it trusts individual stochastic outcomes too much. Global rescaling is
    # more robust — it shifts the overall class distribution to match observed
    # fractions without corrupting per-cell spatial patterns from the prior.
    if obs_probs is not None and obs_weights is not None:
        final = _global_rescale(final, obs_probs, obs_weights, initial_grid)

    return final.astype(np.float32)


def apply_floor_and_normalize(probs: np.ndarray, floor: float = MIN_PROB) -> np.ndarray:
    """Apply minimum probability floor and renormalize."""
    probs = np.maximum(probs, floor)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs.astype(np.float32)


def validate_prediction(prediction: np.ndarray, H: int, W: int) -> bool:
    """Validate prediction format."""
    assert prediction.shape == (H, W, 6), f"Expected ({H},{W},6) got {prediction.shape}"
    sums = prediction.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=0.011), f"Probs don't sum to 1.0 (max diff: {np.abs(sums-1.0).max():.4f})"
    assert (prediction >= 0).all(), "Negative probabilities!"
    return True


def build_prediction_for_seed(
    seed_idx: int,
    initial_state: dict,
    params: SimParams,
    obs_probs: Optional[np.ndarray],
    obs_weights: Optional[np.ndarray],
    total_sims: int = 50000,
    W: int = 40, H: int = 40,
    verbose: bool = True
) -> np.ndarray:
    """
    Build final H×W×6 prediction for one seed.

    Uses Monte Carlo simulation + observation blending.
    """
    import torch

    grid_np = np.array(initial_state["grid"])
    settlements = initial_state.get("settlements", [])

    # Learned prior from historical rounds and local context
    static_prior = build_learned_prior(grid_np, settlements)

    if verbose:
        print(f"  Running {total_sims} Monte Carlo sims for seed {seed_idx}...")

    sim_probs = run_multi_gpu(grid_np, settlements, params,
                               total_sims=total_sims, n_years=50, verbose=verbose)

    # Build obs_probs from counts if provided
    final = blend_predictions(
        sim_probs=sim_probs,
        obs_probs=obs_probs,
        obs_weights=obs_weights,
        static_prior=static_prior,
        initial_grid=grid_np,
        sim_weight=None,
    )

    # Apply floor and normalize
    final = apply_floor_and_normalize(final, floor=MIN_PROB)
    validate_prediction(final, H, W)

    if verbose:
        # Print class distribution
        argmax = final.argmax(axis=-1)
        unique, counts = np.unique(argmax, return_counts=True)
        class_names = ['Empty', 'Settlement', 'Port', 'Ruin', 'Forest', 'Mountain']
        print(f"  Prediction class distribution:")
        for u, c in zip(unique, counts):
            print(f"    {class_names[u]}: {c} cells ({100*c/(H*W):.1f}%)")

    return final


def submit_prediction(client, round_id: str, seed_idx: int,
                      prediction: np.ndarray) -> dict:
    """Submit prediction to API."""
    pred_list = prediction.tolist()
    result = client.submit(round_id, seed_idx, pred_list)
    print(f"  Submitted seed {seed_idx}: {result.get('status', 'unknown')}")
    return result
