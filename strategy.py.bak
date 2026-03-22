"""
Query strategy for Astar Island.

With 50 queries across 5 seeds, we need to be strategic.
Key insight: hidden parameters are SHARED across all 5 seeds.
So observations from any seed help calibrate all seeds.

Strategy:
- Cover the full map with 9 queries per seed (3×3 tiling of 15×15)
- Use remaining queries for hot spots (settlement-dense areas, repeated sampling)
- Pool all observations for parameter estimation
"""

import numpy as np
from typing import List, Tuple, Dict


def plan_full_coverage(W: int, H: int, viewport_size: int = 15) -> List[Tuple[int, int]]:
    """
    Generate (x, y) viewport origins that cover the full W×H map.
    Uses minimum number of non-overlapping tiles.
    """
    tiles = []
    x = 0
    while x < W:
        y = 0
        while y < H:
            tiles.append((x, y))
            y += viewport_size
        x += viewport_size
    return tiles


def plan_queries(initial_states: list, budget: int = 50,
                 W: int = 40, H: int = 40) -> List[Dict]:
    """
    Plan all queries to maximize information extraction.

    Returns list of dicts: {seed_index, viewport_x, viewport_y, viewport_w, viewport_h}
    """
    n_seeds = len(initial_states)
    coverage_tiles = plan_full_coverage(W, H, viewport_size=15)
    n_tiles = len(coverage_tiles)  # Should be 9 for 40×40 with 15×15

    print(f"Full coverage requires {n_tiles} tiles, budget={budget}, seeds={n_seeds}")

    # Identify settlement-dense tiles for additional sampling
    settlement_density = _compute_settlement_density(initial_states, coverage_tiles, W, H)

    queries = []

    # Phase 1: Full coverage for all seeds (9 tiles × 5 seeds = 45 queries)
    base_queries = n_tiles * n_seeds
    remaining = budget - base_queries

    for seed_idx in range(n_seeds):
        for (tx, ty) in coverage_tiles:
            vw = min(15, W - tx)
            vh = min(15, H - ty)
            queries.append({
                "seed_index": seed_idx,
                "viewport_x": tx,
                "viewport_y": ty,
                "viewport_w": max(5, vw),
                "viewport_h": max(5, vh),
            })

    # Phase 2: Use remaining budget on highest-density settlement tiles
    if remaining > 0:
        # Sort tiles by settlement density (highest first)
        sorted_tiles = sorted(range(len(coverage_tiles)),
                               key=lambda i: settlement_density[i], reverse=True)
        extra = 0
        seed_cycle = 0
        for tile_i in sorted_tiles:
            if extra >= remaining:
                break
            tx, ty = coverage_tiles[tile_i]
            vw = min(15, W - tx)
            vh = min(15, H - ty)
            # Add to the seed that has fewest settlement observations
            queries.append({
                "seed_index": seed_cycle % n_seeds,
                "viewport_x": tx,
                "viewport_y": ty,
                "viewport_w": max(5, vw),
                "viewport_h": max(5, vh),
            })
            extra += 1
            seed_cycle += 1

    print(f"Planned {len(queries)} queries ({len(queries)}/{budget} budget used)")
    return queries[:budget]


def plan_staged_queries(initial_states: list,
                        stage_budgets: List[int],
                        W: int = 40,
                        H: int = 40) -> List[List[Dict]]:
    """
    Plan queries in stages for early submit, mid-round refinement, and final overwrite.

    Stage 1 gets broad coverage in a round-robin way across seeds.
    Later stages consume the remaining coverage queries and then hotspot repeats.
    """
    n_seeds = len(initial_states)
    coverage_tiles = plan_full_coverage(W, H, viewport_size=15)
    settlement_density = _compute_settlement_density(initial_states, coverage_tiles, W, H)
    sorted_tiles = [coverage_tiles[i] for i in sorted(
        range(len(coverage_tiles)),
        key=lambda i: settlement_density[i],
        reverse=True,
    )]

    def make_query(seed_idx: int, tx: int, ty: int) -> Dict:
        vw = min(15, W - tx)
        vh = min(15, H - ty)
        return {
            "seed_index": seed_idx,
            "viewport_x": tx,
            "viewport_y": ty,
            "viewport_w": max(5, vw),
            "viewport_h": max(5, vh),
        }

    query_pool = []

    # Coverage first, but interleave seeds so early stages are not biased to seed 0.
    for tile in sorted_tiles:
        tx, ty = tile
        for seed_idx in range(n_seeds):
            query_pool.append(make_query(seed_idx, tx, ty))

    # After full coverage, repeat hotspot tiles with the most starting settlements.
    hotspot_tiles = sorted_tiles[:max(1, min(4, len(sorted_tiles)))]
    extra_needed = max(0, sum(stage_budgets) - len(query_pool))
    hotspot_idx = 0
    for i in range(extra_needed):
        tx, ty = hotspot_tiles[hotspot_idx % len(hotspot_tiles)]
        seed_idx = i % n_seeds
        query_pool.append(make_query(seed_idx, tx, ty))
        hotspot_idx += 1

    stages = []
    cursor = 0
    for budget in stage_budgets:
        stages.append(query_pool[cursor:cursor + budget])
        cursor += budget
    return stages


def _compute_settlement_density(initial_states: list,
                                 coverage_tiles: List[Tuple[int, int]],
                                 W: int, H: int) -> List[float]:
    """Count how many settlements fall in each tile region."""
    density = []
    for (tx, ty) in coverage_tiles:
        count = 0
        for state in initial_states:
            for s in state.get("settlements", []):
                sx, sy = s["x"], s["y"]
                if tx <= sx < tx + 15 and ty <= sy < ty + 15:
                    count += 1
        density.append(float(count))
    return density


def build_observation_map(observations: List[Dict], W: int, H: int) -> np.ndarray:
    """
    Build an observation count map: H×W×6 with empirical class counts.
    Each observation covers a viewport with sample counts.
    """
    obs_counts = np.zeros((H, W, 6), dtype=np.float32)
    obs_n = np.zeros((H, W), dtype=np.float32)

    class_map = np.zeros(16, dtype=np.int8)
    class_map[0] = 0; class_map[1] = 1; class_map[2] = 2
    class_map[3] = 3; class_map[4] = 4; class_map[5] = 5
    class_map[10] = 0; class_map[11] = 0

    for obs in observations:
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        grid = obs["grid"]  # vh×vw terrain codes

        for row_i in range(min(len(grid), vh)):
            for col_i in range(min(len(grid[row_i]), vw)):
                gy = vy + row_i
                gx = vx + col_i
                if 0 <= gy < H and 0 <= gx < W:
                    code = grid[row_i][col_i]
                    if 0 <= code <= 15:
                        cls = class_map[code]
                        obs_counts[gy, gx, cls] += 1
                        obs_n[gy, gx] += 1

    return obs_counts, obs_n
