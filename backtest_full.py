"""
Full backtest: score current prediction pipeline on all historical rounds.

Simulates the scenario where we have observations (using GT as proxy for obs)
and measures KL-divergence-based scores.
"""

import numpy as np
import json
from pathlib import Path
from fast_sim import SimParams, T_MOUNTAIN, T_OCEAN
from prediction import (
    build_learned_prior, blend_predictions, apply_floor_and_normalize,
    _global_rescale, build_static_prior
)
from parameter_estimation import observed_probs_from_counts

TRAIN_DIR = Path("/workspace/checkpoints/training_data")
MIN_PROB = 0.01


def entropy_weighted_kl(pred, gt):
    """
    Compute entropy-weighted KL divergence matching the competition formula.
    score = 100 * exp(-3 * weighted_kl)
    """
    H, W, C = gt.shape
    pred = np.maximum(pred, MIN_PROB)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt_safe = np.maximum(gt, 1e-10)

    # Per-cell KL divergence
    kl_per_cell = np.sum(gt_safe * np.log(gt_safe / pred), axis=-1)  # H×W

    # Per-cell entropy (for weighting)
    entropy_per_cell = -np.sum(gt_safe * np.log(gt_safe), axis=-1)  # H×W

    # Entropy-weighted KL
    total_entropy = entropy_per_cell.sum()
    if total_entropy < 1e-10:
        return 0.0

    weighted_kl = np.sum(entropy_per_cell * kl_per_cell) / total_entropy
    score = 100.0 * np.exp(-3.0 * weighted_kl)
    return score


def simulate_observations(gt, grid, frac=0.6):
    """
    Simulate having observations by sampling from GT.
    Uses ~60% of dynamic cells as "observed" to mimic typical coverage.
    """
    H, W, C = gt.shape
    static = (grid == T_MOUNTAIN) | (grid == T_OCEAN)
    dynamic = ~static

    # Create observation weights and probs
    obs_weights = np.zeros((H, W), dtype=np.float32)
    obs_probs = np.zeros((H, W, C), dtype=np.float32)

    # Simulate 9 viewports covering most of the map (like our query strategy)
    for vx in range(0, W, 15):
        for vy in range(0, H, 15):
            vw = min(15, W - vx)
            vh = min(15, H - vy)
            # Each viewport = 1 observation (conservative)
            obs_weights[vy:vy+vh, vx:vx+vw] += 1.0
            obs_probs[vy:vy+vh, vx:vx+vw] += gt[vy:vy+vh, vx:vx+vw]

    # Normalize obs_probs where observed
    mask = obs_weights > 0
    obs_probs[mask] /= obs_weights[mask, None]

    return obs_probs, obs_weights


def backtest_round(round_num, verbose=True):
    """Backtest one round across all seeds."""
    files = sorted(TRAIN_DIR.glob(f"r{round_num}_s*.npz"))
    if not files:
        return None

    scores_prior_only = []
    scores_sim_prior = []
    scores_with_obs = []
    scores_with_rescale = []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        grid = d["initial_grid"].astype(np.int16)
        gt = d["ground_truth"].astype(np.float32)
        settlements = list(d["settlements"])
        H, W = grid.shape

        # 1. Prior only
        prior = build_learned_prior(grid, settlements)
        prior_floor = apply_floor_and_normalize(prior)
        s_prior = entropy_weighted_kl(prior_floor, gt)
        scores_prior_only.append(s_prior)

        # 2. Simulate observations from GT
        obs_probs, obs_weights = simulate_observations(gt, grid)

        # 3. Prior + observations (no sim, no rescale)
        blend_no_rescale = blend_predictions(
            sim_probs=prior,  # use prior as stand-in for sim
            obs_probs=obs_probs,
            obs_weights=obs_weights,
            static_prior=prior,
            initial_grid=grid,
            sim_weight=0.0,  # no sim contribution
        )
        blend_no_rescale_floor = apply_floor_and_normalize(blend_no_rescale)
        s_obs = entropy_weighted_kl(blend_no_rescale_floor, gt)
        scores_with_obs.append(s_obs)

        # 4. Full pipeline with rescale (what blend_predictions now does)
        # This is already included in blend_predictions, so score from step 3 includes rescale
        # But let's also test rescale on prior-only to isolate its effect
        rescaled = _global_rescale(prior_floor, obs_probs, obs_weights, grid)
        rescaled_floor = apply_floor_and_normalize(rescaled)
        s_rescale = entropy_weighted_kl(rescaled_floor, gt)
        scores_with_rescale.append(s_rescale)

    avg_prior = np.mean(scores_prior_only)
    avg_obs = np.mean(scores_with_obs)
    avg_rescale = np.mean(scores_with_rescale)

    if verbose:
        weight = 1.05 ** round_num
        print(f"  R{round_num:2d}: prior={avg_prior:5.1f}  +obs={avg_obs:5.1f}  "
              f"+rescale_only={avg_rescale:5.1f}  "
              f"(weight={weight:.2f}, best_weighted={max(avg_obs, avg_rescale)*weight:.1f})")

    return {
        "round": round_num,
        "prior": avg_prior,
        "with_obs": avg_obs,
        "rescale_only": avg_rescale,
    }


def main():
    print("=" * 80)
    print("FULL BACKTEST — Current prediction pipeline on all historical rounds")
    print("=" * 80)

    results = []
    for r in range(1, 21):
        res = backtest_round(r)
        if res:
            results.append(res)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_prior = np.mean([r["prior"] for r in results])
    avg_obs = np.mean([r["with_obs"] for r in results])
    avg_rescale = np.mean([r["rescale_only"] for r in results])

    print(f"Average prior-only score:     {avg_prior:.1f}")
    print(f"Average with observations:    {avg_obs:.1f}")
    print(f"Average rescale-only:         {avg_rescale:.1f}")

    # Best weighted score across rounds
    best_weighted = max(
        max(r["with_obs"], r["rescale_only"]) * (1.05 ** r["round"])
        for r in results
    )
    print(f"\nBest weighted score:          {best_weighted:.1f}")

    # Improvement from observations
    improvements = [r["with_obs"] - r["prior"] for r in results]
    print(f"\nObs improvement: min={min(improvements):.1f} max={max(improvements):.1f} "
          f"avg={np.mean(improvements):.1f}")


if __name__ == "__main__":
    main()
