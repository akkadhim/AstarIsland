"""
Training script: fetch GT from all completed rounds, run comprehensive calibration.
Run this between rounds to improve params for the next round.
"""

import json, time, dataclasses, traceback
import numpy as np
from pathlib import Path
import requests
import torch
from scipy.optimize import differential_evolution

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyZjM5NTgwYy05YjRkLTQ3ODEtOGM0Yy04YmI4Y2Y3Y2IwOWIiLCJlbWFpbCI6ImFobWVkLmsua2FkaGltQHVpYS5ubyIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTExMDU0fQ.WnaB9Tkh4IC8-oA8WmiB8M1gmvlX-iv1kZYu48Ef-qg"

CKPT_DIR = Path("/workspace/checkpoints")
TRAIN_DIR = CKPT_DIR / "training_data"
TRAIN_DIR.mkdir(exist_ok=True)
PARAMS_FILE = CKPT_DIR / "best_params.json"

from fast_sim import SimParams, FastViking
import dataclasses as dc


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_training_data(session):
    """Fetch GT from all completed rounds where we submitted."""
    my_rounds = session.get("https://api.ainm.no/astar-island/my-rounds").json()

    all_data = []  # list of (round_num, seed_idx, initial_grid, initial_settlements, ground_truth)

    for r in my_rounds:
        if r["status"] != "completed" or not r.get("seeds_submitted", 0):
            continue

        round_id = r["id"]
        round_num = r["round_number"]
        log(f"Fetching GT for round #{round_num}...")

        # Get initial states
        detail = session.get(f"https://api.ainm.no/astar-island/rounds/{round_id}").json()
        initial_states = detail["initial_states"]
        n_seeds = detail["seeds_count"]

        for si in range(n_seeds):
            cache_file = TRAIN_DIR / f"r{round_num}_s{si}.npz"
            if cache_file.exists():
                d = np.load(cache_file, allow_pickle=True)
                all_data.append({
                    "round": round_num, "seed": si,
                    "initial_grid": d["initial_grid"],
                    "settlements": d["settlements"].tolist(),
                    "ground_truth": d["ground_truth"],
                })
                log(f"  Loaded cached GT r{round_num} s{si}")
                continue

            resp = session.get(f"https://api.ainm.no/astar-island/analysis/{round_id}/{si}")
            if resp.status_code != 200:
                log(f"  GT not available for r{round_num} s{si}: {resp.status_code}")
                continue

            data = resp.json()
            gt = np.array(data["ground_truth"], dtype=np.float32)
            ig = np.array(data.get("initial_grid") or initial_states[si]["grid"], dtype=np.int32)
            sett = initial_states[si].get("settlements", [])

            np.savez(cache_file,
                     initial_grid=ig,
                     settlements=np.array(sett, dtype=object),
                     ground_truth=gt)

            all_data.append({
                "round": round_num, "seed": si,
                "initial_grid": ig,
                "settlements": sett,
                "ground_truth": gt,
            })
            log(f"  Fetched GT r{round_num} s{si}, score={data.get('score'):.2f}")

    return all_data


def calibrate_on_all_data(all_data, current_params: SimParams) -> SimParams:
    """Run differential evolution on all available GT data."""
    if not all_data:
        log("No training data available!")
        return current_params

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    log(f"Building calibration dataset from {len(all_data)} seed-round pairs...")

    # Vectorized entropy computation
    cells = []  # (data_idx, y, x, gt_dist, entropy)
    for di, d in enumerate(all_data):
        gt = d["ground_truth"].astype(np.float32)  # H×W×6
        p_clip = np.clip(gt, 1e-9, 1)
        p_clip /= p_clip.sum(axis=-1, keepdims=True)
        ent = -(p_clip * np.log(p_clip)).sum(axis=-1)  # H×W
        high = np.where(ent > 0.15)
        for y, x in zip(high[0], high[1]):
            cells.append((di, int(y), int(x), gt[y, x].copy(), float(ent[y, x])))

    log(f"Total high-entropy cells: {len(cells)}")
    if len(cells) < 20:
        log("Too few cells — skipping calibration")
        return current_params

    # Sample at most 400 cells weighted by entropy
    if len(cells) > 400:
        weights = np.array([c[4] for c in cells])
        weights /= weights.sum()
        idx = np.random.choice(len(cells), 400, replace=False, p=weights)
        cells = [cells[i] for i in idx]

    # Focus on 8 key parameters that most affect settlement distribution
    KEYS = ["food_per_forest", "food_base", "pop_growth_rate", "pop_maintenance",
            "expansion_min_pop", "expansion_min_food", "expansion_prob", "winter_mean"]
    BOUNDS = [(0.10, 0.90), (0.05, 0.30), (0.02, 0.20), (0.05, 0.30),
              (1.0, 4.0),   (0.10, 0.50), (0.10, 0.70), (0.10, 0.70)]

    x0 = np.clip([getattr(current_params, k) for k in KEYS],
                 [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

    best = [float("inf")]; best_v = [x0.copy()]
    iter_count = [0]
    t_start = time.time()

    def obj(v):
        iter_count[0] += 1
        p = dc.replace(current_params, **{k: float(v[i]) for i, k in enumerate(KEYS)})
        by_data = {}
        for di, y, x, gt, w in cells:
            by_data.setdefault(di, []).append((y, x, gt, w))

        loss = weight = 0.0
        for di, cell_list in by_data.items():
            d = all_data[di]
            grid_np = d["initial_grid"].astype(np.int32)
            sett = d["settlements"]
            try:
                sim = FastViking(grid_np, sett, p, batch_size=200, device=dev)
                sp = sim.run_and_aggregate(50)
            except Exception:
                return 1e6
            for y, x, gt, w in cell_list:
                sq = np.clip(sp[y, x], 1e-7, 1); sq /= sq.sum()
                gq = np.clip(gt, 1e-7, 1); gq /= gq.sum()
                loss += w * float((gq * np.log(gq / sq)).sum())
                weight += w

        l = loss / max(weight, 1e-9)
        if l < best[0]:
            best[0] = l; best_v[0] = v.copy()
            elapsed = time.time() - t_start
            log(f"  iter {iter_count[0]} ({elapsed:.0f}s): KL={l:.4f} "
                f"exp_prob={v[6]:.3f} pop_growth={v[2]:.3f} "
                f"winter={v[7]:.3f} exp_min_pop={v[4]:.2f}")
        return l

    log("Running differential evolution (maxiter=30, popsize=5, 8 params)...")
    try:
        result = differential_evolution(
            obj, BOUNDS, x0=x0, seed=42,
            maxiter=30, popsize=5,
            tol=0.002, mutation=(0.5, 1.0), recombination=0.7,
            polish=True, disp=False, workers=1)
        v = result.x
    except Exception as e:
        log(f"DE failed: {e}, using best so far")
        v = best_v[0]

    new_p = dc.replace(current_params, **{k: float(v[i]) for i, k in enumerate(KEYS)})
    log(f"Calibration done. KL={best[0]:.4f} in {(time.time()-t_start)/60:.1f}min")
    log(f"  expansion_prob={new_p.expansion_prob:.3f}")
    log(f"  pop_growth_rate={new_p.pop_growth_rate:.3f}")
    log(f"  expansion_min_pop={new_p.expansion_min_pop:.3f}")
    log(f"  winter_mean={new_p.winter_mean:.3f}")
    log(f"  food_per_forest={new_p.food_per_forest:.3f}")
    return new_p


def main():
    log("=" * 55)
    log("  Astar Island Training Script")
    log("=" * 55)

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {TOKEN}"

    # Fetch all training data
    all_data = fetch_training_data(session)
    log(f"Total training examples: {len(all_data)}")

    if not all_data:
        log("No training data — exiting.")
        return

    # Load current params
    if PARAMS_FILE.exists():
        current = SimParams(**json.loads(PARAMS_FILE.read_text()))
    else:
        current = SimParams()

    # Calibrate
    new_params = calibrate_on_all_data(all_data, current)

    # Save
    PARAMS_FILE.write_text(json.dumps(dc.asdict(new_params), indent=2))
    log("Saved improved params to best_params.json")

    # Print comparison
    KEYS = ["expansion_prob", "pop_growth_rate", "expansion_min_pop",
            "winter_mean", "food_per_forest", "pop_maintenance"]
    log("Parameter changes:")
    for k in KEYS:
        old = getattr(current, k)
        new = getattr(new_params, k)
        log(f"  {k}: {old:.3f} -> {new:.3f} ({new-old:+.3f})")


if __name__ == "__main__":
    main()
