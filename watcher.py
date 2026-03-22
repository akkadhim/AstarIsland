"""
Autonomous Astar Island watcher + self-improvement loop.
Calibration runs in a background thread — never blocks round execution.
"""

import time, json, dataclasses, traceback, threading, subprocess, os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from api_client import AstarClient
from fast_sim import SimParams, FastViking
from strategy import plan_staged_queries, build_observation_map
from prediction import (build_learned_prior, blend_predictions,
                        apply_floor_and_normalize, validate_prediction)
from parameter_estimation import observed_probs_from_counts
from multi_gpu import run_sequential_multi_gpu

TOKEN = os.environ.get("ASTAR_TOKEN", "")

POLL_INTERVAL  = 20     # seconds between round polls
SCORING_POLL   = 40     # seconds between scoring checks
TOTAL_SIMS     = 150000
FINAL_SUBMIT_BUFFER = 8 * 60

# Submission control:
# The watcher now submits automatically after validation so we do not miss a round.
AUTO_SUBMIT    = True
SUBMIT_NOW_FLAG = Path("/workspace/SUBMIT_NOW")
CHECKPOINT_DIR = Path("/workspace/checkpoints")
PARAMS_FILE    = CHECKPOINT_DIR / "best_params.json"
HISTORY_FILE   = CHECKPOINT_DIR / "round_history.json"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Shared params — protected by lock, updated by calibration thread
_params_lock  = threading.Lock()
_current_params: SimParams = None
_calibrating  = threading.Event()   # set while calibration running


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def load_params() -> SimParams:
    if PARAMS_FILE.exists():
        try:
            p = SimParams(**json.loads(PARAMS_FILE.read_text()))
            log(f"Loaded params: expansion_prob={p.expansion_prob:.3f} "
                f"winter_mean={p.winter_mean:.3f}")
            return p
        except Exception as e:
            log(f"Bad params file ({e}), using defaults")
    return SimParams()


def save_params(p: SimParams):
    PARAMS_FILE.write_text(json.dumps(dataclasses.asdict(p), indent=2))
    log(f"Params saved: expansion_prob={p.expansion_prob:.3f} "
        f"winter_mean={p.winter_mean:.3f}")


def get_params() -> SimParams:
    with _params_lock:
        return _current_params


def set_params(p: SimParams):
    global _current_params
    with _params_lock:
        _current_params = p
    save_params(p)


def load_history():
    return json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []


def save_history(h):
    HISTORY_FILE.write_text(json.dumps(h, indent=2))


def _parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _seconds_to_close(active: dict) -> float:
    return (_parse_time(active["closes_at"]) - datetime.now(timezone.utc)).total_seconds()


def _round_still_active(client: AstarClient, round_id: str) -> bool:
    try:
        active = client.get_active_round()
    except Exception:
        return True
    return bool(active and active["id"] == round_id)


def _stage_budgets(total_budget: int) -> list:
    """Three-stage focus: Discover (45) → Focus (remaining) → Predict.
    Maximize observation coverage since global_rescale needs class fraction estimates."""
    if total_budget <= 0:
        return []
    if total_budget <= 15:
        return [total_budget]
    # Stage 1: broad coverage (45 queries = 9 tiles × 5 seeds)
    stage1 = min(45, total_budget)
    remaining = total_budget - stage1
    if remaining <= 0:
        return [stage1]
    # Stage 2: focused re-observation + final predict
    return [stage1, remaining]


def _stage_sim_budget(stage_name: str) -> int:
    if stage_name == "discover":
        return min(50000, TOTAL_SIMS)
    return TOTAL_SIMS


def _stage_names(stage_count: int) -> list:
    if stage_count <= 0:
        return []
    if stage_count == 1:
        return ["final"]
    return ["discover", "final"][:stage_count]


def _wait_for_final_window(client: AstarClient, round_id: str, closes_at: str) -> bool:
    target = _parse_time(closes_at).timestamp() - FINAL_SUBMIT_BUFFER
    while time.time() < target:
        if not _round_still_active(client, round_id):
            log("  Round ended before final overwrite window")
            return False
        sleep_for = min(30, max(1, target - time.time()))
        time.sleep(sleep_for)
    return True


def _execute_queries(client: AstarClient, round_id: str, queries: list,
                     observations: dict, stage_name: str) -> None:
    if not queries:
        return
    log(f"  {stage_name}: executing {len(queries)} queries...")
    for i, q in enumerate(queries):
        if not _round_still_active(client, round_id):
            log(f"  {stage_name}: round closed during query phase")
            return
        try:
            res = client.simulate(round_id=round_id,
                                  seed_index=q["seed_index"],
                                  viewport_x=q["viewport_x"],
                                  viewport_y=q["viewport_y"],
                                  viewport_w=q["viewport_w"],
                                  viewport_h=q["viewport_h"])
            if res:
                observations[q["seed_index"]].append(res)
        except Exception as e:
            log(f"  {stage_name}: query {i + 1} error: {e}")
        time.sleep(0.22)
    log(f"  {stage_name}: queries done")


def _save_observations(round_id: str, observations: dict, W: int, H: int):
    """Save observations to disk so late_submit_worker can use them."""
    data = {"round_id": round_id, "W": W, "H": H,
            "observations_by_seed": {str(k): v for k, v in observations.items()}}
    obs_path = CHECKPOINT_DIR / "observations.json"
    obs_path.write_text(json.dumps(data))


def _obs_inputs(obs_list: list, W: int, H: int):
    if not obs_list:
        return None, None
    obs_counts, obs_n = build_observation_map(obs_list, W, H)
    return observed_probs_from_counts(obs_counts, obs_n), obs_n


def _build_predictions(initial_states: list, observations: dict, params: SimParams,
                       total_sims: int, W: int, H: int,
                       round_number: int, stage_name: str) -> dict:
    predictions = {}
    for si, state in enumerate(initial_states):
        log(f"  {stage_name}: seed {si} simulating with {total_sims} sims...")
        grid_np = np.array(state["grid"])
        sett = state.get("settlements", [])
        obs_probs, obs_weights = _obs_inputs(observations.get(si, []), W, H)

        sim_probs = run_sequential_multi_gpu(
            grid_np, sett, params,
            total_sims=total_sims, n_years=50, verbose=False)

        final = blend_predictions(
            sim_probs, obs_probs, obs_weights,
            build_learned_prior(grid_np, sett), grid_np)
        final = apply_floor_and_normalize(final)
        validate_prediction(final, H, W)

        predictions[si] = final
        np.save(CHECKPOINT_DIR / f"pred_r{round_number}_s{si}_{stage_name}.npy", final)
        np.save(CHECKPOINT_DIR / f"pred_r{round_number}_s{si}.npy", final)
        log(f"  {stage_name}: seed {si} prediction saved")
    return predictions


def _submit_predictions(client: AstarClient, round_id: str, predictions: dict, stage_name: str) -> list:
    submitted = []
    if not AUTO_SUBMIT:
        return submitted
    log(f"  {stage_name}: submitting predictions (overwrite if already submitted)...")
    for si, final in predictions.items():
        try:
            resp = client.submit(round_id, si, final.tolist())
            log(f"  {stage_name}: seed {si} ✓ {resp.get('status')}")
            submitted.append(si)
        except Exception as e:
            log(f"  {stage_name}: seed {si} submit error: {e}")
    return submitted


# ── calibration (runs in background thread) ──────────────────────────────────

def calibrate_from_gt(client: AstarClient, round_id: str,
                       initial_states: list, n_seeds: int):
    """Fit params to ground truth. Runs in background thread."""
    if _calibrating.is_set():
        log("Calibration already running, skipping")
        return
    _calibrating.set()
    try:
        _do_calibrate(client, round_id, initial_states, n_seeds)
    except Exception as e:
        log(f"Calibration error: {e}")
        traceback.print_exc()
    finally:
        _calibrating.clear()


def _do_calibrate(client, round_id, initial_states, n_seeds):
    import torch
    from scipy.optimize import differential_evolution

    TRAIN_DIR = CHECKPOINT_DIR / "training_data"
    TRAIN_DIR.mkdir(exist_ok=True)

    log("CALIBRATION: Fetching ground truth...")
    seed_gt = {}
    for si in range(n_seeds):
        try:
            r = client.session.get(
                f"https://api.ainm.no/astar-island/analysis/{round_id}/{si}")
            if r.status_code == 200:
                seed_gt[si] = np.array(r.json()["ground_truth"], dtype=np.float32)
                np.save(TRAIN_DIR / f"gt_r{round_id}_{si}.npy", seed_gt[si])
                log(f"  GT seed {si} ok")
        except Exception as e:
            log(f"  GT seed {si} error: {e}")

    if not seed_gt:
        return

    H = len(initial_states[0]["grid"])
    W = len(initial_states[0]["grid"][0])

    # Save as proper training data (npz format matching what prediction.py expects)
    round_num = None
    try:
        rounds = client.get_rounds()
        for r in rounds:
            if r["id"] == round_id:
                round_num = r["round_number"]
                break
    except Exception:
        pass

    if round_num is not None:
        for si, gt in seed_gt.items():
            npz_path = TRAIN_DIR / f"r{round_num}_s{si}.npz"
            if not npz_path.exists():
                state = initial_states[si]
                sett_list = state.get("settlements", [])
                np.savez_compressed(
                    npz_path,
                    initial_grid=np.array(state["grid"], dtype=np.int16),
                    ground_truth=gt,
                    settlements=np.array(sett_list),
                )
                log(f"  Saved training data: {npz_path.name}")
        # Invalidate prior cache so next round rebuilds empirical stats
        cache_file = CHECKPOINT_DIR / "empirical_prior_stats.json"
        if cache_file.exists():
            cache_file.unlink()
            log("  Invalidated empirical prior cache")

    # Build weighted cells: high-entropy cells from ground truth
    def entropy(p):
        p = np.clip(p, 1e-9, 1); p /= p.sum()
        return float(-(p * np.log(p)).sum())

    cells = []
    for si, gt in seed_gt.items():
        for y in range(H):
            for x in range(W):
                ent = entropy(gt[y, x])
                if ent > 0.15:
                    cells.append((si, y, x, gt[y, x].copy(), ent))

    log(f"CALIBRATION: {len(cells)} high-entropy cells")
    if len(cells) < 10:
        return

    # Sample 300 weighted cells for speed
    if len(cells) > 300:
        weights = np.array([c[4] for c in cells])
        weights /= weights.sum()
        idx = np.random.choice(len(cells), 300, replace=False, p=weights)
        cells = [cells[i] for i in idx]

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    current = get_params()

    # Focus on 8 key parameters for faster calibration
    KEYS = ["food_per_forest","food_base","pop_growth_rate","pop_maintenance",
            "expansion_min_pop","expansion_min_food","expansion_prob","winter_mean"]
    BOUNDS = [(0.10,0.90),(0.05,0.30),(0.02,0.20),(0.05,0.30),
              (1.0,4.0),(0.10,0.50),(0.10,0.70),(0.10,0.70)]
    x0 = np.clip([getattr(current, k) for k in KEYS],
                 [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

    best = [float("inf")]; best_v = [x0.copy()]

    def obj(v):
        p = dataclasses.replace(current, **{k: float(v[i]) for i,k in enumerate(KEYS)})
        by_seed = {}
        for si,y,x,gt,w in cells:
            by_seed.setdefault(si,[]).append((y,x,gt,w))
        loss = weight = 0.0
        for si, sc in by_seed.items():
            grid_np = np.array(initial_states[si]["grid"])
            sett    = initial_states[si].get("settlements",[])
            try:
                sim = FastViking(grid_np, sett, p, batch_size=200, device=dev)
                sp  = sim.run_and_aggregate(50)
            except:
                return 1e6
            for y,x,gt,w in sc:
                sq = np.clip(sp[y,x],1e-7,1); sq/=sq.sum()
                gq = np.clip(gt,1e-7,1);      gq/=gq.sum()
                loss   += w * float((gq*np.log(gq/sq)).sum())
                weight += w
        l = loss/max(weight,1e-9)
        if l < best[0]:
            best[0]=l; best_v[0]=v.copy()
            log(f"  CAL iter: KL={l:.4f} exp_prob={v[6]:.3f} "
                f"winter={v[7]:.3f} pop_growth={v[2]:.3f}")
        return l

    log("CALIBRATION: Running differential evolution (background)...")
    try:
        result = differential_evolution(
            obj, BOUNDS, x0=x0, seed=42,
            maxiter=25, popsize=5,
            tol=0.003, mutation=(0.5,1.0), recombination=0.7,
            polish=True, disp=False, workers=1)
        v = result.x
    except Exception:
        v = best_v[0]

    new_p = dataclasses.replace(current, **{k:float(v[i]) for i,k in enumerate(KEYS)})
    log(f"CALIBRATION done. KL={best[0]:.4f} "
        f"exp_prob={new_p.expansion_prob:.3f} "
        f"winter={new_p.winter_mean:.3f}")
    set_params(new_p)


# ── observation-based quick calibration ──────────────────────────────────────

def _obs_calibrate(initial_states: list, observations: dict,
                   params: SimParams, W: int, H: int) -> SimParams:
    """
    Adjust expansion_prob and winter_mean to match observed settlement + ruin fraction.
    Uses binary search on 2 key parameters.
    """
    import torch
    # Compute observed settlement and ruin fractions across all seeds/queries
    total_cells = total_sett = total_ruin = 0
    obs_pop = []
    obs_food = []
    obs_wealth = []
    obs_def = []
    obs_port = []
    static_codes = {5, 10}  # mountain, ocean
    for si, obs_list in observations.items():
        for obs in obs_list:
            grid = obs["grid"]
            for row in grid:
                for code in row:
                    if code not in static_codes:
                        total_cells += 1
                        if code in (1, 2):    # settlement or port
                            total_sett += 1
                        elif code == 3:        # ruin
                            total_ruin += 1
            for sv in obs.get("settlements", []):
                if sv.get("alive", True):
                    obs_pop.append(float(sv.get("population", 0.0)))
                    obs_food.append(float(sv.get("food", 0.0)))
                    obs_wealth.append(float(sv.get("wealth", 0.0)))
                    obs_def.append(float(sv.get("defense", 0.0)))
                    obs_port.append(1.0 if sv.get("has_port", False) else 0.0)

    if total_cells < 50:
        log("  Not enough observations for obs-calibration")
        return params

    obs_sett_frac = total_sett / total_cells
    obs_ruin_frac = total_ruin / total_cells
    obs_civ_frac  = (total_sett + total_ruin) / total_cells  # civilization density
    obs_pop_mean = float(np.mean(obs_pop)) if obs_pop else 0.0
    obs_food_mean = float(np.mean(obs_food)) if obs_food else 0.0
    obs_wealth_mean = float(np.mean(obs_wealth)) if obs_wealth else 0.0
    obs_def_mean = float(np.mean(obs_def)) if obs_def else 0.0
    obs_port_frac = float(np.mean(obs_port)) if obs_port else 0.0
    log(f"  Observed: sett={obs_sett_frac:.3f} ruin={obs_ruin_frac:.3f} "
        f"civ={obs_civ_frac:.3f} ({total_cells} cells)")
    if obs_pop:
        log(f"  Settlement stats: pop={obs_pop_mean:.3f} food={obs_food_mean:.3f} "
            f"wealth={obs_wealth_mean:.3f} def={obs_def_mean:.3f} port={obs_port_frac:.3f}")

    # Use seed with most observations for calibration
    si0 = max(observations.keys(), key=lambda k: len(observations[k]))
    state0 = initial_states[si0]
    grid_np = np.array(state0["grid"])
    sett = state0.get("settlements", [])
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def sim_summary(p: SimParams):
        try:
            sim = FastViking(grid_np, sett, p, batch_size=800, device=dev)
            return sim.run_and_summarize(50)
        except Exception:
            return None

    def summary_score(summary: dict):
        if summary is None:
            return 1e9
        score = 0.0
        score += abs(summary["sett_frac"] - obs_sett_frac) * 5.0
        score += abs(summary["ruin_frac"] - obs_ruin_frac) * 3.0
        if obs_pop:
            score += abs(summary["pop_mean"] - obs_pop_mean) * 0.4
            score += abs(summary["food_mean"] - obs_food_mean) * 0.7
            score += abs(summary["wealth_mean"] - obs_wealth_mean) * 0.6
            score += abs(summary["def_mean"] - obs_def_mean) * 0.5
            score += abs(summary["port_frac"] - obs_port_frac) * 1.0
        return score

    # Phase 1: Binary search on expansion_prob [0→1.0] to match observed settlement
    lo, hi = 0.0, 1.0
    best_p = params
    best_diff = float("inf")

    for _ in range(9):
        mid = (lo + hi) / 2
        test_p = dataclasses.replace(params, expansion_prob=mid)
        summary = sim_summary(test_p)
        diff = summary_score(summary)
        if diff < best_diff:
            best_diff = diff
            best_p = test_p
        sf = 0.0 if summary is None else summary["sett_frac"]
        if sf < obs_sett_frac:
            lo = mid
        else:
            hi = mid

    log(f"  ep: {params.expansion_prob:.3f} → {best_p.expansion_prob:.3f} "
        f"(obs score={best_diff:.3f})")

    # Phase 2a: If under-predicting (high settlement rounds like R17),
    # search winter_mean DOWN and food_per_forest UP to allow more growth
    p1_summary = sim_summary(best_p)
    p1_sf = 0.0 if p1_summary is None else p1_summary["sett_frac"]

    if p1_sf < obs_sett_frac * 0.7:
        # Sim can't produce enough settlements — search aggressive growth params
        log(f"  Phase 2a: sim under-predicting ({p1_sf:.3f} vs obs {obs_sett_frac:.3f}), wide search...")
        for wm in [0.20, 0.30, 0.40, 0.50, 0.60]:
            for pm in [0.05, 0.10, 0.15, 0.20]:
                for fpf in [0.35, 0.45, 0.55]:
                    test_p2 = dataclasses.replace(
                        best_p,
                        winter_mean=wm,
                        pop_maintenance=pm,
                        food_per_forest=fpf,
                        expansion_prob=min(best_p.expansion_prob + 0.15, 1.0),
                    )
                    diff2 = summary_score(sim_summary(test_p2))
                    if diff2 < best_diff:
                        best_diff = diff2
                        best_p = test_p2
        log(f"  Phase 2a result: wm={best_p.winter_mean:.3f} pm={best_p.pop_maintenance:.3f} "
            f"fpf={best_p.food_per_forest:.3f} ep={best_p.expansion_prob:.3f} score={best_diff:.3f}")

    # Phase 2b: If over-predicting with ep→0, reduce food_per_forest
    elif best_p.expansion_prob < 0.05 and p1_sf > obs_sett_frac:
        log(f"  Phase 2b: ep={best_p.expansion_prob:.3f} still over ({p1_sf:.3f}>{obs_sett_frac:.3f}), searching food_per_forest...")
        lo_f, hi_f = 0.10, best_p.food_per_forest
        p2 = best_p
        for _ in range(7):
            mid_f = (lo_f + hi_f) / 2
            test_p2 = dataclasses.replace(best_p, food_per_forest=mid_f)
            summary2 = sim_summary(test_p2)
            sf2 = 0.0 if summary2 is None else summary2["sett_frac"]
            diff2 = summary_score(summary2)
            if diff2 < best_diff:
                best_diff = diff2
                p2 = test_p2
            if sf2 > obs_sett_frac:
                hi_f = mid_f
            else:
                lo_f = mid_f
        log(f"  fpf: {best_p.food_per_forest:.3f} → {p2.food_per_forest:.3f} (score={best_diff:.3f})")
        best_p = p2

    # Phase 3: local refinement on winter_mean and pop_maintenance
    candidates = []
    for dw in (-0.12, -0.06, 0.0, 0.06, 0.12):
        for dm in (-0.06, -0.03, 0.0, 0.03, 0.06):
            test = dataclasses.replace(
                best_p,
                winter_mean=float(np.clip(best_p.winter_mean + dw, 0.05, 0.80)),
                pop_maintenance=float(np.clip(best_p.pop_maintenance + dm, 0.02, 0.40)),
            )
            candidates.append(test)

    for test_p3 in candidates:
        diff3 = summary_score(sim_summary(test_p3))
        if diff3 < best_diff:
            best_diff = diff3
            best_p = test_p3

    log(f"  winter/pop refinement: winter={best_p.winter_mean:.3f} "
        f"pop_maint={best_p.pop_maintenance:.3f} score={best_diff:.3f}")

    log(f"  Final: ep={best_p.expansion_prob:.3f} fpf={best_p.food_per_forest:.3f} "
        f"wm={best_p.winter_mean:.3f} pm={best_p.pop_maintenance:.3f} obs_sett={obs_sett_frac:.3f}")
    return best_p


def _spawn_late_worker(round_id: str, round_number: int, closes_at: str):
    """Spawn late_submit_worker.py as a background process for higher-sim resubmission."""
    script = Path("/workspace/late_submit_worker.py")
    if not script.exists():
        log("  late_submit_worker.py not found, skipping")
        return
    # Kill any existing late worker
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmd = (pid_dir / "cmdline").read_bytes().decode("utf-8", "ignore")
            if "late_submit_worker" in cmd and int(pid_dir.name) != os.getpid():
                os.kill(int(pid_dir.name), 9)
                log(f"  Killed old late worker PID {pid_dir.name}")
        except Exception:
            pass
    log_file = Path("/workspace/late_submit_worker.log")
    proc = subprocess.Popen(
        ["python3", "-u", str(script), round_id, str(round_number), closes_at],
        stdout=log_file.open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"  Spawned late_submit_worker PID={proc.pid} for R#{round_number}")


# ── round execution ───────────────────────────────────────────────────────────

def run_round(client: AstarClient, active: dict) -> dict:
    params    = get_params()
    round_id  = active["id"]
    W, H      = active["map_width"], active["map_height"]
    log(f"▶ Round #{active['round_number']} | closes {active['closes_at']} | "
        f"exp_prob={params.expansion_prob:.2f}")

    detail         = client.get_round_detail(round_id)
    initial_states = detail["initial_states"]
    n_seeds        = detail["seeds_count"]
    log(f"  Seeds: {n_seeds}, settlements: "
        f"{[len(s.get('settlements',[])) for s in initial_states]}")

    try:
        binfo      = client.get_budget()
        budget_rem = binfo["queries_max"] - binfo["queries_used"]
    except:
        budget_rem = 50
    log(f"  Budget: {budget_rem} queries")

    observations = {i: [] for i in range(n_seeds)}
    stage_budgets = _stage_budgets(budget_rem)
    if not stage_budgets:
        stage_budgets = [0]
    stage_names = _stage_names(len(stage_budgets))
    query_stages = plan_staged_queries(initial_states, stage_budgets, W=W, H=H) if sum(stage_budgets) > 0 else [[]]
    submitted = []

    for idx, stage_name in enumerate(stage_names):
        time_left = _seconds_to_close(active)
        if time_left <= 90:
            log(f"  {stage_name}: not enough time left ({time_left:.0f}s), stopping stage loop")
            break

        _execute_queries(client, round_id, query_stages[idx], observations, stage_name)
        # Save observations for late_submit_worker
        _save_observations(round_id, observations, W, H)
        params = _obs_calibrate(initial_states, observations, params, W, H)

        stage_preds = _build_predictions(
            initial_states=initial_states,
            observations=observations,
            params=params,
            total_sims=_stage_sim_budget(stage_name),
            W=W,
            H=H,
            round_number=active["round_number"],
            stage_name=stage_name,
        )
        submitted = _submit_predictions(client, round_id, stage_preds, stage_name) or submitted

        if not AUTO_SUBMIT and stage_name == stage_names[-1]:
            log(f"  AUTO_SUBMIT=False. Predictions saved to checkpoints/.")
            log(f"  Create /workspace/SUBMIT_NOW to submit, or set AUTO_SUBMIT=True.")
            while not SUBMIT_NOW_FLAG.exists():
                if not _round_still_active(client, round_id):
                    log("  Round closed without submission!")
                    return {"round_id": round_id, "round_number": active["round_number"],
                            "submitted": [], "initial_states": initial_states, "n_seeds": n_seeds}
                time.sleep(15)
            SUBMIT_NOW_FLAG.unlink()
            submitted = _submit_predictions(client, round_id, stage_preds, "manual-final")

        if not _round_still_active(client, round_id):
            log("  Round closed after stage submission")
            break

    log(f"  Latest submission covered {len(submitted)}/5 seeds")

    # Spawn late_submit_worker if enough time remains for a resubmit with more sims
    time_left = _seconds_to_close(active)
    if time_left > FINAL_SUBMIT_BUFFER + 120 and submitted:
        _spawn_late_worker(round_id, active["round_number"], active["closes_at"])

    return {"round_id": round_id, "round_number": active["round_number"],
            "submitted": submitted, "initial_states": initial_states,
            "n_seeds": n_seeds}


def wait_for_score(client, round_id, round_number):
    log(f"  Waiting for score on round #{round_number}...")
    for _ in range(300):
        time.sleep(SCORING_POLL)
        try:
            for r in client.get_my_rounds():
                if r["id"] == round_id and r.get("round_score") is not None:
                    log(f"  ★ Round #{round_number}: score={r['round_score']:.2f} "
                        f"rank={r.get('rank')}/{r.get('total_teams')}")
                    return r
                # Check for next round — don't block on scoring
                active = client.get_active_round()
                if active and active["id"] != round_id:
                    log("  New round detected while waiting for score — will process next")
                    return None
        except Exception as e:
            log(f"  Score poll error: {e}")
    return None


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    global _current_params
    log("=" * 55)
    log("  Astar Island Autonomous Watcher v2")
    log("=" * 55)

    _current_params = load_params()
    history  = load_history()
    seen     = {r["round_id"] for r in history}

    while True:
        try:
            active = client.get_active_round()

            incomplete_active = False
            if active and active["id"] in seen:
                try:
                    my_rounds = client.get_my_rounds()
                    match = next((r for r in my_rounds if r["id"] == active["id"]), None)
                    if match and match.get("status") == "active":
                        incomplete_active = match.get("seeds_submitted", 0) < match.get("seeds_count", 5)
                        if incomplete_active:
                            log(f"Round #{active['round_number']} is active but only "
                                f"{match.get('seeds_submitted', 0)}/{match.get('seeds_count', 5)} seeds submitted; re-entering")
                            seen.discard(active["id"])
                except Exception as e:
                    log(f"Active-round recovery check failed: {e}")

            if not active or (active["id"] in seen and not incomplete_active):
                status = f"Round #{active['round_number']} already done" if active else "No active round"
                log(f"{status}. Polling in {POLL_INTERVAL}s...")
                # Reload params from disk in case training updated them
                fresh = load_params()
                cur = get_params()
                if abs(fresh.expansion_prob - cur.expansion_prob) > 0.001 or \
                   abs(fresh.winter_mean - cur.winter_mean) > 0.001:
                    with _params_lock:
                        _current_params = fresh
                    log(f"Params reloaded: exp_prob={fresh.expansion_prob:.3f} winter={fresh.winter_mean:.3f}")
                time.sleep(POLL_INTERVAL)
                continue

            # Execute round
            round_data = run_round(client, active)
            seen.add(active["id"])

            # Save to history immediately
            history.append({"round_id": active["id"],
                             "round_number": active["round_number"],
                             "score": None, "rank": None,
                             "params": dataclasses.asdict(get_params())})
            save_history(history)

            # Wait for score (non-blocking: exits if new round appears)
            scored = wait_for_score(client, active["id"], active["round_number"])

            if scored:
                history[-1].update({"score": scored["round_score"],
                                     "rank": scored.get("rank"),
                                     "total_teams": scored.get("total_teams")})
                save_history(history)

                # Start calibration in background if score is poor
                score_val = scored["round_score"]
                rank      = scored.get("rank", 999)
                total     = scored.get("total_teams", 999)
                pct_rank  = rank / max(total, 1)

                # Always collect training data (GT) for future model improvement
                t = threading.Thread(
                    target=calibrate_from_gt,
                    args=(client, active["id"],
                          round_data["initial_states"],
                          round_data["n_seeds"]),
                    daemon=True)
                t.start()

                if score_val < 80 or pct_rank > 0.25:
                    log(f"Score {score_val:.1f} rank {rank}/{total} "
                        f"— calibration started in background")
                else:
                    log(f"Score {score_val:.1f} rank {rank}/{total} — good score, still collecting GT data")

        except KeyboardInterrupt:
            log("Stopped by user.")
            break
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
            time.sleep(30)


if __name__ == "__main__":
    client = AstarClient(TOKEN)
    main()
