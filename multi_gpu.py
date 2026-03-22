"""
Run simulations in parallel across all 8 H100 GPUs using multiprocessing.
Each GPU gets its own process with its share of simulations.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from fast_sim import FastViking, SimParams


def _worker(rank: int, n_gpus: int, initial_grid, initial_settlements_list,
             params_dict: dict, sims_per_gpu: int, n_years: int,
             result_queue):
    """Worker process: runs sims on GPU `rank`."""
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        # Reconstruct params
        params = SimParams(**params_dict)

        H, W = initial_grid.shape
        all_probs = np.zeros((H, W, 6), dtype=np.float64)

        for initial_settlements in initial_settlements_list:
            sim = FastViking(
                initial_grid=initial_grid,
                initial_settlements=initial_settlements,
                params=params,
                batch_size=sims_per_gpu,
                device=device,
            )
            probs = sim.run_and_aggregate(n_years=n_years)
            all_probs += probs

        all_probs /= len(initial_settlements_list)
        result_queue.put((rank, all_probs.astype(np.float32)))
    except Exception as e:
        result_queue.put((rank, e))


def run_all_gpus(initial_grid: np.ndarray, initial_settlements: list,
                 params: SimParams, total_sims: int = 80000,
                 n_years: int = 50, verbose: bool = True) -> np.ndarray:
    """
    Run total_sims simulations in parallel across all available GPUs.
    Returns H×W×6 probability distribution.
    """
    n_gpus = min(torch.cuda.device_count(), MAX_GPUS)
    if n_gpus == 0:
        if verbose: print("  No GPUs — running on CPU")
        sim = FastViking(initial_grid, initial_settlements, params, total_sims, "cpu")
        return sim.run_and_aggregate(n_years)

    sims_per_gpu = total_sims // n_gpus
    if verbose:
        print(f"  Using {n_gpus} GPUs × {sims_per_gpu} sims = {n_gpus*sims_per_gpu} total")

    import dataclasses
    params_dict = dataclasses.asdict(params)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(n_gpus):
        p = ctx.Process(
            target=_worker,
            args=(rank, n_gpus, initial_grid, [initial_settlements],
                  params_dict, sims_per_gpu, n_years, result_queue),
        )
        p.start()
        processes.append(p)

    results = {}
    for _ in range(n_gpus):
        rank, data = result_queue.get()
        if isinstance(data, Exception):
            print(f"  GPU {rank} error: {data}")
            results[rank] = np.full((initial_grid.shape[0], initial_grid.shape[1], 6),
                                     1.0/6, dtype=np.float32)
        else:
            results[rank] = data

    for p in processes:
        p.join()

    # Average across GPUs
    all_arrays = [results[i] for i in range(n_gpus)]
    final = np.mean(all_arrays, axis=0).astype(np.float32)
    return final


MAX_GPUS = 3  # Cap at 3 GPUs to leave resources for other tasks


def run_sequential_multi_gpu(initial_grid: np.ndarray, initial_settlements: list,
                               params: SimParams, total_sims: int = 30000,
                               n_years: int = 50, verbose: bool = True) -> np.ndarray:
    """
    Run total_sims sequentially across GPUs (simpler, no multiprocessing).
    Capped at MAX_GPUS=3.
    """
    import time
    n_gpus = min(torch.cuda.device_count(), MAX_GPUS)
    if n_gpus == 0:
        n_gpus = 1
        devices = ["cpu"]
    else:
        devices = [f"cuda:{i}" for i in range(n_gpus)]

    sims_per_gpu = total_sims // n_gpus
    H, W = initial_grid.shape
    all_probs = []

    for dev_i, dev in enumerate(devices):
        if verbose:
            print(f"  GPU {dev_i}: {sims_per_gpu} sims on {dev}...")
        t0 = time.time()
        sim = FastViking(initial_grid, initial_settlements, params,
                         sims_per_gpu, dev)
        probs = sim.run_and_aggregate(n_years)
        all_probs.append(probs)
        if verbose:
            print(f"    Done in {time.time()-t0:.1f}s")

    return np.mean(all_probs, axis=0).astype(np.float32)
