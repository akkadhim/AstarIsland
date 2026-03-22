"""
Train v6: full retraining on rounds 1-10 using the current production blend.
Writes improved params to checkpoints/best_params.json as soon as it finds them.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import requests
from scipy.optimize import differential_evolution

from fast_sim import FastViking, SimParams
from neural_cell_model import train_neural_cell_model
from prediction import apply_floor_and_normalize, blend_predictions, build_learned_prior
from prior_calibrator import train_prior_calibrator
from watcher import TOKEN

CHECKPOINT_DIR = Path("/workspace/checkpoints")
TRAIN_DIR = CHECKPOINT_DIR / "training_data"
PARAMS_FILE = CHECKPOINT_DIR / "best_params.json"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sync_training_data() -> None:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {TOKEN}"
    try:
        my_rounds = session.get("https://api.ainm.no/astar-island/my-rounds", timeout=30).json()
    except Exception as e:
        log(f"Training sync skipped: {e}")
        return

    fetched = 0
    for r in my_rounds:
        if r.get("status") != "completed" or not r.get("seeds_submitted"):
            continue
        round_num = int(r["round_number"])
        round_id = r["id"]
        try:
            detail = session.get(f"https://api.ainm.no/astar-island/rounds/{round_id}", timeout=30).json()
        except Exception as e:
            log(f"  round #{round_num}: detail fetch failed: {e}")
            continue
        for si, state in enumerate(detail["initial_states"]):
            fp = TRAIN_DIR / f"r{round_num}_s{si}.npz"
            if fp.exists():
                continue
            try:
                analysis = session.get(
                    f"https://api.ainm.no/astar-island/analysis/{round_id}/{si}", timeout=30
                )
                if analysis.status_code != 200:
                    continue
                data = analysis.json()
                np.savez(
                    fp,
                    initial_grid=np.array(data.get("initial_grid") or state["grid"], dtype=np.int32),
                    settlements=np.array(state.get("settlements", []), dtype=object),
                    ground_truth=np.array(data["ground_truth"], dtype=np.float32),
                )
                fetched += 1
                log(f"  synced r{round_num}_s{si}")
            except Exception as e:
                log(f"  sync failed for r{round_num}_s{si}: {e}")
    if fetched:
        log(f"Training sync added {fetched} new seed files")


with open(PARAMS_FILE) as f:
    current = SimParams(**json.load(f))


sync_training_data()
neural_info = train_neural_cell_model()
if neural_info is not None:
    log(
        "Neural cell model updated: "
        f"val_loss={neural_info['val_loss']:.4f} "
        f"train={neural_info['train_examples']} val={neural_info['val_examples']}"
    )
cal_info = train_prior_calibrator()
if cal_info is not None:
    log(
        "Prior calibrator updated: "
        f"val_loss={cal_info['val_loss']:.4f} "
        f"train={cal_info['train_examples']} val={cal_info['val_examples']}"
    )


all_data = []
for fp in sorted(TRAIN_DIR.glob("r*_s*.npz")):
    d = np.load(fp, allow_pickle=True)
    round_num = int(fp.stem.split("_")[0][1:])
    seed_num = int(fp.stem.split("_")[1][1:])
    gt_sett = float(d["ground_truth"][:, :, 1].mean())
    all_data.append({
        "round": round_num,
        "seed": seed_num,
        "initial_grid": d["initial_grid"].astype(np.int32),
        "settlements": list(d["settlements"]),
        "ground_truth": d["ground_truth"].astype(np.float32),
        "gt_sett": gt_sett,
    })
    log(f"  Loaded r{round_num}_s{seed_num}: GT_sett={gt_sett:.3f}")

log(f"Total examples: {len(all_data)}")

cells_by_di = {}
for di, d in enumerate(all_data):
    gt = d["ground_truth"]
    p_clip = np.clip(gt, 1e-9, 1)
    p_clip /= p_clip.sum(axis=-1, keepdims=True)
    ent = -(p_clip * np.log(p_clip)).sum(axis=-1)
    ys, xs = np.where(ent > 0.05)
    cell_list = [(int(y), int(x), gt[y, x].copy(), float(ent[y, x])) for y, x in zip(ys, xs)]
    if len(cell_list) > 36:
        weights = np.array([c[3] for c in cell_list], dtype=np.float64)
        weights /= weights.sum()
        keep = np.random.choice(len(cell_list), 36, replace=False, p=weights)
        cell_list = [cell_list[i] for i in keep]
    cells_by_di[di] = cell_list

log(f"Sampled cells: {sum(len(v) for v in cells_by_di.values())}")

KEYS = [
    "winter_mean",
    "food_per_forest",
    "pop_maintenance",
    "pop_growth_rate",
    "forest_reclaim_prob",
    "expansion_min_pop",
    "expansion_min_food",
    "food_collapse_threshold",
    "food_base",
    "trade_food_gain",
]
BOUNDS = [
    (0.08, 0.80),
    (0.08, 0.90),
    (0.02, 0.40),
    (0.02, 0.25),
    (0.01, 0.25),
    (0.30, 5.00),
    (0.03, 0.70),
    (-0.60, 0.15),
    (0.03, 0.45),
    (0.01, 0.40),
]
x0 = np.clip([getattr(current, k) for k in KEYS],
             [b[0] for b in BOUNDS],
             [b[1] for b in BOUNDS])

best = [float("inf")]
best_v = [x0.copy()]
iter_count = [0]
t_start = time.time()


def obj(v):
    iter_count[0] += 1
    params = dataclasses.replace(current, **{k: float(v[i]) for i, k in enumerate(KEYS)})

    loss = 0.0
    weight = 0.0
    for di, cell_list in cells_by_di.items():
        d = all_data[di]
        try:
            sim = FastViking(
                d["initial_grid"],
                d["settlements"],
                params,
                batch_size=100,
                device="cuda:0",
            )
            sim_probs = sim.run_and_aggregate(50)
        except Exception:
            return 1e6

        prior = build_learned_prior(d["initial_grid"], d["settlements"])
        final = blend_predictions(sim_probs, None, None, prior, d["initial_grid"], sim_weight=None)
        final = apply_floor_and_normalize(final)

        for y, x, gt_vec, w in cell_list:
            pred_vec = np.clip(final[y, x], 1e-7, 1.0)
            pred_vec /= pred_vec.sum()
            gt_vec = np.clip(gt_vec, 1e-7, 1.0)
            gt_vec /= gt_vec.sum()
            loss += w * float((gt_vec * np.log(gt_vec / pred_vec)).sum())
            weight += w

    score = loss / max(weight, 1e-9)
    if score < best[0]:
        best[0] = score
        best_v[0] = v.copy()
        elapsed = time.time() - t_start
        log(
            f"  iter {iter_count[0]} ({elapsed:.0f}s): KL={score:.4f} "
            + " ".join(f"{k}={v[i]:.3f}" for i, k in enumerate(KEYS))
        )
        improved = dataclasses.replace(current, **{k: float(v[i]) for i, k in enumerate(KEYS)})
        PARAMS_FILE.write_text(json.dumps(dataclasses.asdict(improved), indent=2))
    return score


log(f"Running DE with {len(KEYS)} params on {len(all_data)} examples...")
result = differential_evolution(
    obj,
    BOUNDS,
    x0=x0,
    seed=42,
    maxiter=60,
    popsize=10,
    tol=0.0005,
    polish=True,
    disp=False,
    workers=1,
    mutation=(0.5, 1.2),
    recombination=0.8,
)

new_params = dataclasses.replace(current, **{k: float(result.x[i]) for i, k in enumerate(KEYS)})
PARAMS_FILE.write_text(json.dumps(dataclasses.asdict(new_params), indent=2))

log(f"Done. Best KL={best[0]:.4f}")
for k in KEYS:
    log(f"  {k}: {getattr(current, k):.4f} -> {getattr(new_params, k):.4f}")
