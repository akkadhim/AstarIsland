"""
Parameter estimation: fit hidden SimParams from observations.

Key insight: ALL 5 seeds share the same hidden parameters.
We can pool observations from all seeds to get better estimates.

Approach:
1. Use observations to build empirical per-cell class distributions
2. Run local simulations with candidate params
3. Minimize KL divergence between simulated and observed distributions
4. Use CMA-ES (or scipy minimize) for optimization

Since we don't know exact initial stats (pop, food, wealth, defense),
we treat them as random with priors and marginalize over them.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import List, Dict, Tuple, Optional
import time

from fast_sim import SimParams, FastViking


# Parameter bounds for optimization
PARAM_BOUNDS = {
    "food_per_forest": (0.1, 1.0),
    "food_base": (0.02, 0.3),
    "pop_growth_rate": (0.02, 0.2),
    "pop_maintenance": (0.05, 0.5),
    "expansion_min_pop": (1.5, 6.0),
    "expansion_prob": (0.05, 0.4),
    "port_prob": (0.05, 0.3),
    "raid_prob_normal": (0.02, 0.2),
    "raid_prob_desperate": (0.1, 0.6),
    "raid_desperate_threshold": (0.1, 0.6),
    "winter_mean": (0.2, 1.0),
    "winter_std": (0.1, 0.4),
    "winter_severe_prob": (0.05, 0.3),
    "forest_reclaim_prob": (0.01, 0.15),
    "rebuild_prob": (0.05, 0.25),
    "trade_food_gain": (0.05, 0.4),
    "trade_wealth_gain": (0.05, 0.5),
}

import dataclasses as _dc
# Only optimize keys that exist in SimParams AND have bounds defined
PARAM_KEYS = [k for k in PARAM_BOUNDS.keys()
              if k in {f.name for f in _dc.fields(SimParams())}]


def params_to_vector(p: SimParams) -> np.ndarray:
    return np.array([getattr(p, k) for k in PARAM_KEYS])


def vector_to_params(v: np.ndarray) -> SimParams:
    p = SimParams()
    for i, k in enumerate(PARAM_KEYS):
        setattr(p, k, float(v[i]))
    return p


def get_bounds() -> List[Tuple[float, float]]:
    return [PARAM_BOUNDS[k] for k in PARAM_KEYS]


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """KL(p || q) with smoothing."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def observed_probs_from_counts(obs_counts: np.ndarray, obs_n: np.ndarray,
                                prior: float = 0.01) -> np.ndarray:
    """
    Convert observation counts to probability estimates (Dirichlet posterior).
    obs_counts: H×W×6
    obs_n: H×W
    Returns H×W×6 probability array.

    Prior of 0.05 (not 0.5): with 1 observation, this gives the observed class
    ~77% probability instead of ~37%. Important because observations are single
    stochastic samples and we don't want the prior to overwhelm the signal.
    """
    H, W = obs_n.shape
    probs = np.zeros((H, W, 6), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            n = obs_n[y, x]
            if n > 0:
                # Dirichlet posterior with weak uniform prior
                alpha = obs_counts[y, x] + prior
                probs[y, x] = alpha / alpha.sum()
            else:
                probs[y, x] = 1.0 / 6  # Uniform prior for unobserved cells
    return probs


def fit_parameters(initial_states: List[Dict],
                   observations_per_seed: List[List[Dict]],
                   W: int = 40, H: int = 40,
                   n_eval_sims: int = 500,
                   max_iter: int = 50,
                   verbose: bool = True) -> SimParams:
    """
    Estimate hidden parameters from observations using differential evolution.

    initial_states: one per seed (grid + settlements)
    observations_per_seed: list of observation results per seed
    n_eval_sims: simulations per parameter evaluation (more = better estimate but slower)
    """
    from strategy import build_observation_map

    # Build empirical distributions from observations for each seed
    print("Building empirical observation maps...")
    seed_obs_probs = []
    seed_obs_weights = []  # Which cells have observations

    for seed_idx, obs_list in enumerate(observations_per_seed):
        if not obs_list:
            continue
        obs_counts, obs_n = build_observation_map(obs_list, W, H)
        obs_probs = observed_probs_from_counts(obs_counts, obs_n)
        # Weight by number of observations at each cell
        weights = np.minimum(obs_n, 5.0)  # Cap at 5 for stability
        seed_obs_probs.append((seed_idx, obs_probs, weights))
        seed_obs_weights.append(weights.sum())

    if not seed_obs_probs:
        print("No observations to fit — using default params")
        return SimParams()

    # Extract initial grids
    initial_grids = []
    initial_settlements_list = []
    for state in initial_states:
        grid = np.array(state["grid"])
        settlements = state.get("settlements", [])
        initial_grids.append(grid)
        initial_settlements_list.append(settlements)

    call_count = [0]
    best_loss = [float('inf')]
    best_params = [None]

    def objective(v: np.ndarray) -> float:
        call_count[0] += 1
        params = vector_to_params(v)

        total_loss = 0.0
        total_weight = 0.0

        for seed_idx, obs_probs, weights in seed_obs_probs:
            grid = initial_grids[seed_idx]
            settlements = initial_settlements_list[seed_idx]

            # Run simulations with candidate params
            import torch
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            sim = FastViking(
                initial_grid=grid,
                initial_settlements=settlements,
                params=params,
                batch_size=n_eval_sims,
                device=dev,
            )
            try:
                sim_probs = sim.run_and_aggregate(n_years=50)
            except Exception as e:
                return 1e6  # Penalize invalid params

            # Compute weighted KL divergence for observed cells
            for y in range(H):
                for x in range(W):
                    w = weights[y, x]
                    if w < 0.1:
                        continue
                    kl = compute_kl_divergence(obs_probs[y, x], sim_probs[y, x])
                    total_loss += w * kl
                    total_weight += w

        loss = total_loss / max(total_weight, 1.0)

        if loss < best_loss[0]:
            best_loss[0] = loss
            best_params[0] = v.copy()
            if verbose:
                print(f"  Iter {call_count[0]}: loss={loss:.4f}")

        return loss

    bounds = get_bounds()
    x0 = params_to_vector(SimParams())

    print(f"Starting parameter optimization (max_iter={max_iter}, n_eval_sims={n_eval_sims})...")
    t0 = time.time()

    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max_iter,
            popsize=8,
            tol=0.01,
            seed=42,
            x0=x0,
            workers=1,  # Single-threaded to avoid GPU conflicts
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            disp=verbose,
        )
        final_params = vector_to_params(result.x)
    except Exception as e:
        print(f"Optimization failed: {e}, using best found so far")
        if best_params[0] is not None:
            final_params = vector_to_params(best_params[0])
        else:
            final_params = SimParams()

    elapsed = time.time() - t0
    print(f"Optimization complete in {elapsed:.1f}s. Best loss: {best_loss[0]:.4f}")
    return final_params


def quick_parameter_estimate(initial_states: List[Dict],
                              observations_per_seed: List[List[Dict]],
                              W: int = 40, H: int = 40,
                              n_eval_sims: int = 200) -> SimParams:
    """
    Fast parameter estimation using Nelder-Mead (fewer evaluations).
    Good for quick first pass.
    """
    from strategy import build_observation_map

    seed_obs_probs = []
    for seed_idx, obs_list in enumerate(observations_per_seed):
        if not obs_list:
            continue
        obs_counts, obs_n = build_observation_map(obs_list, W, H)
        obs_probs = observed_probs_from_counts(obs_counts, obs_n)
        weights = np.minimum(obs_n, 3.0)
        if weights.sum() > 0:
            seed_obs_probs.append((seed_idx, obs_probs, weights))

    if not seed_obs_probs:
        return SimParams()

    initial_grids = [np.array(s["grid"]) for s in initial_states]
    initial_settlements_list = [s.get("settlements", []) for s in initial_states]

    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def fast_objective(v: np.ndarray) -> float:
        params = vector_to_params(v)
        total_loss = 0.0
        total_weight = 0.0
        for seed_idx, obs_probs, weights in seed_obs_probs[:2]:  # Use first 2 seeds
            grid = initial_grids[seed_idx]
            settlements = initial_settlements_list[seed_idx]
            sim = FastViking(grid, settlements, params, n_eval_sims, dev)
            try:
                sim_probs = sim.run_and_aggregate(50)
            except:
                return 1e6
            # Sample random observed cells for speed
            observed = np.argwhere(weights > 0.1)
            if len(observed) == 0:
                continue
            sample = observed[np.random.choice(len(observed), min(50, len(observed)), replace=False)]
            for yx in sample:
                y, x = yx
                w = weights[y, x]
                kl = compute_kl_divergence(obs_probs[y, x], sim_probs[y, x])
                total_loss += w * kl
                total_weight += w
        return total_loss / max(total_weight, 1.0)

    x0 = params_to_vector(SimParams())
    bounds = get_bounds()

    result = minimize(
        fast_objective, x0, method='Nelder-Mead',
        options={'maxiter': 30, 'xatol': 0.05, 'fatol': 0.01}
    )
    return vector_to_params(np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds]))
