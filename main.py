"""
Astar Island — Main competition solver.

Usage:
    python main.py --token YOUR_JWT_TOKEN [--skip-queries] [--skip-param-fit] [--dry-run]

Steps:
1. Get active round + initial states
2. Execute strategic queries to observe the world
3. Estimate hidden parameters from observations
4. Run massive Monte Carlo simulations on 8×H100
5. Build probability predictions and submit
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path

from api_client import AstarClient
from strategy import plan_queries, build_observation_map
from fast_sim import SimParams
from parameter_estimation import quick_parameter_estimate, fit_parameters
from prediction import (build_prediction_for_seed, submit_prediction,
                        apply_floor_and_normalize)
from parameter_estimation import observed_probs_from_counts


def save_checkpoint(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_queries(client: AstarClient, round_id: str, initial_states: list,
                queries: list, verbose: bool = True) -> dict:
    """
    Execute all planned queries and return observations grouped by seed.
    Returns: {seed_idx: [observation_result, ...]}
    """
    observations = {i: [] for i in range(len(initial_states))}
    total = len(queries)

    print(f"\nExecuting {total} queries...")
    for i, q in enumerate(queries):
        seed_idx = q["seed_index"]
        if verbose:
            print(f"  Query {i+1}/{total}: seed={seed_idx} "
                  f"x={q['viewport_x']},y={q['viewport_y']} "
                  f"{q['viewport_w']}×{q['viewport_h']}")

        result = client.simulate(
            round_id=round_id,
            seed_index=seed_idx,
            viewport_x=q["viewport_x"],
            viewport_y=q["viewport_y"],
            viewport_w=q["viewport_w"],
            viewport_h=q["viewport_h"],
        )

        if result:
            observations[seed_idx].append(result)
            budget_left = result.get("queries_max", 50) - result.get("queries_used", 0)
            if verbose and (i + 1) % 5 == 0:
                print(f"  Budget remaining: {budget_left}")

        # Respect rate limit (5 req/sec)
        time.sleep(0.25)

    return observations


def main():
    parser = argparse.ArgumentParser(description="Astar Island Solver")
    parser.add_argument("--token", required=True, help="JWT access token")
    parser.add_argument("--skip-queries", action="store_true",
                        help="Skip queries (load from checkpoint)")
    parser.add_argument("--skip-param-fit", action="store_true",
                        help="Skip parameter fitting (use defaults)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build predictions but don't submit")
    parser.add_argument("--total-sims", type=int, default=30000,
                        help="Monte Carlo sims per seed (default: 30000 across 3 GPUs)")
    parser.add_argument("--n-eval-sims", type=int, default=300,
                        help="Sims per parameter evaluation (default: 300)")
    parser.add_argument("--checkpoint-dir", default="/workspace/checkpoints",
                        help="Directory for checkpoints")
    args = parser.parse_args()

    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    ckpt_obs = f"{args.checkpoint_dir}/observations.json"
    ckpt_params = f"{args.checkpoint_dir}/params.json"

    client = AstarClient(args.token)

    # ── Step 1: Get active round ──────────────────────────────────────────
    print("\n=== Step 1: Get Active Round ===")
    active = client.get_active_round()
    if not active:
        print("No active round found!")
        rounds = client.get_rounds()
        print(f"Available rounds: {[(r['round_number'], r['status']) for r in rounds]}")
        return

    round_id = active["id"]
    W = active["map_width"]
    H = active["map_height"]
    print(f"Active round #{active['round_number']} | {W}×{H} map | closes: {active['closes_at']}")

    # ── Step 2: Get round details ─────────────────────────────────────────
    print("\n=== Step 2: Get Round Details ===")
    detail = client.get_round_detail(round_id)
    initial_states = detail["initial_states"]
    n_seeds = detail["seeds_count"]
    print(f"Seeds: {n_seeds}")
    for i, state in enumerate(initial_states):
        grid = np.array(state["grid"])
        n_settlements = len([s for s in state.get("settlements", []) if s.get("alive", True)])
        unique, counts = np.unique(grid, return_counts=True)
        print(f"  Seed {i}: {n_settlements} settlements, terrain: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Check budget
    try:
        budget_info = client.get_budget()
        queries_used = budget_info.get("queries_used", 0)
        queries_max = budget_info.get("queries_max", 50)
        budget_remaining = queries_max - queries_used
        print(f"Budget: {queries_used}/{queries_max} used, {budget_remaining} remaining")
    except Exception as e:
        print(f"Could not get budget: {e}")
        budget_remaining = 50

    # ── Step 3: Execute queries ───────────────────────────────────────────
    print("\n=== Step 3: Execute Queries ===")
    observations = None

    if not args.skip_queries and budget_remaining > 0:
        queries = plan_queries(initial_states, budget=budget_remaining, W=W, H=H)
        observations_by_seed = run_queries(client, round_id, initial_states, queries)

        # Save checkpoint
        checkpoint_data = {
            "round_id": round_id,
            "W": W, "H": H,
            "observations_by_seed": {str(k): v for k, v in observations_by_seed.items()}
        }
        save_checkpoint(checkpoint_data, ckpt_obs)
        observations = observations_by_seed
    else:
        # Try loading from checkpoint
        ckpt = load_checkpoint(ckpt_obs)
        if ckpt and ckpt.get("round_id") == round_id:
            print("Loading observations from checkpoint...")
            observations = {int(k): v for k, v in ckpt["observations_by_seed"].items()}
            total_obs = sum(len(v) for v in observations.values())
            print(f"  Loaded {total_obs} observations")
        else:
            print("No checkpoint found, proceeding without observations")
            observations = {i: [] for i in range(n_seeds)}

    # ── Step 4: Parameter estimation ─────────────────────────────────────
    print("\n=== Step 4: Parameter Estimation ===")

    if not args.skip_param_fit and observations:
        # Check if we have enough observations
        total_obs = sum(len(v) for v in observations.values())
        if total_obs >= 3:
            obs_per_seed = [observations.get(i, []) for i in range(n_seeds)]
            print(f"Fitting parameters from {total_obs} total observations...")

            params = quick_parameter_estimate(
                initial_states=initial_states,
                observations_per_seed=obs_per_seed,
                W=W, H=H,
                n_eval_sims=args.n_eval_sims,
            )

            # Save params
            params_dict = {k: getattr(params, k) for k in vars(SimParams()).__class__.__annotations__
                          if hasattr(SimParams(), k)}
            # Simpler: save all float attrs
            import dataclasses
            params_dict = dataclasses.asdict(params)
            save_checkpoint(params_dict, ckpt_params)
            print("  Parameters estimated and saved.")
        else:
            print("  Not enough observations for fitting, using defaults")
            params = SimParams()
    else:
        # Try loading saved params
        ckpt = load_checkpoint(ckpt_params)
        if ckpt:
            print("Loading parameters from checkpoint...")
            params = SimParams(**ckpt)
        else:
            print("Using default parameters")
            params = SimParams()

    print(f"  Key params: food_per_forest={params.food_per_forest:.3f}, "
          f"winter_mean={params.winter_mean:.3f}, "
          f"expansion_prob={params.expansion_prob:.3f}")

    # ── Step 5: Build and submit predictions ─────────────────────────────
    print(f"\n=== Step 5: Build Predictions ({args.total_sims} sims/seed on GPU) ===")

    for seed_idx in range(n_seeds):
        print(f"\nSeed {seed_idx}/{n_seeds-1}:")
        initial_state = initial_states[seed_idx]
        grid_np = np.array(initial_state["grid"])

        # Build observation probability maps for this seed
        seed_obs_list = observations.get(seed_idx, []) if observations else []
        obs_probs = None
        obs_weights = None

        if seed_obs_list:
            obs_counts, obs_n = build_observation_map(seed_obs_list, W, H)
            obs_probs = observed_probs_from_counts(obs_counts, obs_n)
            obs_weights = obs_n
            total_obs_cells = (obs_n > 0).sum()
            print(f"  Using {len(seed_obs_list)} observations covering {total_obs_cells} cells")

        # Build prediction
        prediction = build_prediction_for_seed(
            seed_idx=seed_idx,
            initial_state=initial_state,
            params=params,
            obs_probs=obs_probs,
            obs_weights=obs_weights,
            total_sims=args.total_sims,
            W=W, H=H,
            verbose=True,
        )

        # Save prediction as checkpoint
        pred_path = f"{args.checkpoint_dir}/prediction_seed{seed_idx}.npy"
        np.save(pred_path, prediction)
        print(f"  Saved prediction: {pred_path}")

        # Submit
        if not args.dry_run:
            result = submit_prediction(client, round_id, seed_idx, prediction)
        else:
            print(f"  [DRY RUN] Would submit seed {seed_idx}")

    print("\n=== Done! ===")
    if not args.dry_run:
        print("All 5 seeds submitted. Check leaderboard at app.ainm.no")

        # Show my rounds info
        try:
            my_rounds = client.get_my_rounds()
            for r in my_rounds:
                if r.get("id") == round_id:
                    print(f"Seeds submitted: {r.get('seeds_submitted', 0)}/{n_seeds}")
                    if r.get("round_score") is not None:
                        print(f"Round score: {r['round_score']:.1f}")
        except Exception as e:
            print(f"Could not fetch my-rounds: {e}")


if __name__ == "__main__":
    main()
