"""
Late overwrite worker for an active round.

Waits until close to the round, then rebuilds predictions with the latest params
and a larger simulation budget, and overwrites the team's prior submissions.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from api_client import AstarClient
from fast_sim import SimParams
from multi_gpu import run_sequential_multi_gpu
from prediction import build_learned_prior, blend_predictions, apply_floor_and_normalize, validate_prediction
from parameter_estimation import observed_probs_from_counts
from watcher import TOKEN

CHECKPOINT_DIR = Path("/workspace/checkpoints")
PARAMS_FILE = CHECKPOINT_DIR / "best_params.json"
LOG_FILE = Path("/workspace/late_submit_worker.log")
FINAL_SIM_BUDGET = 250000
FINAL_BUFFER_SECS = 15 * 60


def log(msg: str):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_params() -> SimParams:
    if PARAMS_FILE.exists():
        return SimParams(**json.loads(PARAMS_FILE.read_text()))
    return SimParams()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: late_submit_worker.py <round_id> <round_number> <closes_at>")

    round_id = sys.argv[1]
    round_number = int(sys.argv[2])
    closes_at = sys.argv[3]

    client = AstarClient(TOKEN)
    target = parse_time(closes_at).timestamp() - FINAL_BUFFER_SECS
    log(f"Late worker armed for round #{round_number}, target overwrite at {datetime.fromtimestamp(target, tz=timezone.utc).isoformat()}")

    while time.time() < target:
        active = client.get_active_round()
        if not active or active["id"] != round_id:
            log("Round is no longer active; exiting late worker")
            return
        time.sleep(min(30, max(1, target - time.time())))

    active = client.get_active_round()
    if not active or active["id"] != round_id:
        log("Round ended before late overwrite window")
        return

    detail = client.get_round_detail(round_id)
    initial_states = detail["initial_states"]
    W, H = active["map_width"], active["map_height"]
    params = load_params()
    # Load saved observations from watcher if available
    obs_data = None
    obs_file = CHECKPOINT_DIR / "observations.json"
    if obs_file.exists():
        try:
            obs_data = json.loads(obs_file.read_text())
            if obs_data.get("round_id") != round_id:
                log("Observations are from a different round, ignoring")
                obs_data = None
            else:
                log(f"Loaded observations for {len(obs_data.get('observations_by_seed', {}))} seeds")
        except Exception as e:
            log(f"Failed to load observations: {e}")

    log(f"Starting late overwrite for round #{round_number} with sims={FINAL_SIM_BUDGET} and winter={params.winter_mean:.3f}")

    for si, state in enumerate(initial_states):
        try:
            grid_np = np.array(state["grid"])
            sett = state.get("settlements", [])
            sim_probs = run_sequential_multi_gpu(
                grid_np, sett, params,
                total_sims=FINAL_SIM_BUDGET, n_years=50, verbose=False,
            )
            # Use observations if available
            obs_probs, obs_weights = None, None
            if obs_data:
                from strategy import build_observation_map
                obs_list = obs_data["observations_by_seed"].get(str(si), [])
                if obs_list:
                    obs_counts, obs_n = build_observation_map(obs_list, W, H)
                    obs_probs, obs_weights = observed_probs_from_counts(obs_counts, obs_n), obs_n
                    log(f"  seed {si}: using {len(obs_list)} observations ({int(np.sum(obs_weights > 0))} cells)")
            final = blend_predictions(
                sim_probs, obs_probs, obs_weights,
                build_learned_prior(grid_np, sett), grid_np,
            )
            final = apply_floor_and_normalize(final)
            validate_prediction(final, H, W)
            np.save(CHECKPOINT_DIR / f"pred_r{round_number}_s{si}_late.npy", final)
            resp = client.submit(round_id, si, final.tolist())
            log(f"Late overwrite seed {si} -> {resp.get('status')}")
        except Exception as e:
            log(f"Error processing late overwrite for seed {si}: {e}")

    log(f"Late overwrite complete for round #{round_number}")


if __name__ == "__main__":
    main()
