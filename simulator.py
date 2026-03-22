"""
Vectorized Norse civilization simulator.
Runs thousands of parallel simulations on GPU using PyTorch.

Terrain codes:
  0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
  10=Ocean, 11=Plains

Class indices for prediction:
  0=Empty(Ocean/Plains/Empty), 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Terrain codes
T_EMPTY = 0
T_SETTLEMENT = 1
T_PORT = 2
T_RUIN = 3
T_FOREST = 4
T_MOUNTAIN = 5
T_OCEAN = 10
T_PLAINS = 11

# Class index for prediction
def terrain_to_class(t: int) -> int:
    if t in (T_OCEAN, T_PLAINS, T_EMPTY):
        return 0
    return t  # 1,2,3,4,5 map directly


@dataclass
class SimParams:
    """Hidden simulation parameters to be estimated."""
    # Food production
    food_per_forest: float = 0.4        # food gained per adjacent forest tile per year
    food_base: float = 0.1              # base food per turn (plains/empty)
    food_sea_bonus: float = 0.05        # bonus food for ports from sea

    # Population dynamics
    pop_growth_rate: float = 0.08       # fraction of (food - maintenance) converted to pop
    pop_maintenance: float = 0.2        # food needed per pop unit per turn
    pop_collapse_threshold: float = 0.0 # pop below this → collapse

    # Expansion
    expansion_min_pop: float = 3.0      # min pop to attempt expansion
    expansion_min_food: float = 0.5     # min food to attempt expansion
    expansion_prob: float = 0.15        # probability of expansion per year (if conditions met)
    expansion_max_dist: int = 3         # max distance to expand

    # Port building
    port_prob: float = 0.12             # prob coastal settlement builds port per year
    port_min_wealth: float = 0.3        # min wealth to build port

    # Longships
    longship_prob: float = 0.08         # prob of building longship (if port and wealthy)
    longship_min_wealth: float = 0.5
    longship_range_bonus: int = 5       # extra raid range from longship

    # Raiding
    base_raid_range: int = 4            # max raid distance without longship
    raid_prob_normal: float = 0.08      # prob of raiding per year (well-fed)
    raid_prob_desperate: float = 0.35   # prob when food < desperate_threshold
    raid_desperate_threshold: float = 0.3
    raid_success_prob: float = 0.5      # base success probability
    raid_defense_factor: float = 0.3    # how much defense reduces success prob
    raid_loot_fraction: float = 0.3     # fraction of defender's food/wealth looted
    conquest_prob: float = 0.2          # prob conquered settlement changes allegiance

    # Trade
    trade_range: int = 8                # max port-to-port trade distance
    trade_food_gain: float = 0.15       # food gain from trade per year
    trade_wealth_gain: float = 0.2      # wealth gain from trade per year
    trade_tech_gain: float = 0.05       # tech gain from trade

    # Winter severity
    winter_mean: float = 0.5            # mean food loss per winter
    winter_std: float = 0.25            # std dev of food loss
    winter_min: float = 0.1            # minimum winter food loss
    winter_severe_prob: float = 0.15    # probability of severe winter (3× mean)
    winter_severe_mult: float = 2.5

    # Collapse threshold
    food_collapse_threshold: float = -0.1  # food below this → collapse

    # Environment
    forest_reclaim_prob: float = 0.06   # ruin → forest per year (no nearby settlement)
    ruin_plains_prob: float = 0.04      # ruin → plains per year (no nearby settlement)
    rebuild_range: int = 3              # max distance for settlement to rebuild ruin
    rebuild_prob: float = 0.12          # prob of rebuilding nearby ruin per year
    rebuild_as_port_prob: float = 0.4   # prob rebuilt coastal ruin becomes port


def make_static_masks(grid_np: np.ndarray) -> dict:
    """
    Precompute static properties of the initial grid.
    Returns torch tensors for fast simulation.
    grid_np: H×W numpy array of terrain codes.
    """
    H, W = grid_np.shape
    is_ocean = (grid_np == T_OCEAN)
    is_mountain = (grid_np == T_MOUNTAIN)
    is_forest_init = (grid_np == T_FOREST)
    is_static = is_ocean | is_mountain
    is_coastal = np.zeros((H, W), dtype=bool)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(is_ocean, dy, axis=0), dx, axis=1)
            is_coastal |= shifted
    is_coastal &= ~is_ocean & ~is_mountain

    # Precompute adjacency count to ocean (for coastal) and forest (for food)
    # These are static at init - forests can change but we'll recompute dynamically
    return {
        "H": H, "W": W,
        "is_ocean": torch.from_numpy(is_ocean),
        "is_mountain": torch.from_numpy(is_mountain),
        "is_forest_init": torch.from_numpy(is_forest_init),
        "is_static": torch.from_numpy(is_static),
        "is_coastal": torch.from_numpy(is_coastal),
    }


class VikingSimulator:
    """
    Batch Viking civilization simulator.
    Runs B independent simulations in parallel on a single GPU device.
    """

    def __init__(self, initial_grid: np.ndarray, initial_settlements: list,
                 params: SimParams, batch_size: int = 1000, device: str = "cuda"):
        """
        initial_grid: H×W array (terrain codes at start)
        initial_settlements: list of dicts with x, y, has_port, alive
        params: SimParams
        batch_size: number of parallel simulations
        """
        self.params = params
        self.B = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.statics = make_static_masks(initial_grid)
        self.H = self.statics["H"]
        self.W = self.statics["W"]
        self.initial_grid_np = initial_grid.copy()
        self.initial_settlements = [s for s in initial_settlements if s.get("alive", True)]
        self.max_settlements = max(200, len(self.initial_settlements) * 4)

        # Move static masks to device
        for k, v in self.statics.items():
            if isinstance(v, torch.Tensor):
                self.statics[k] = v.to(self.device)

        # Init offsets for neighbor lookup
        self._make_neighbor_offsets()

    def _make_neighbor_offsets(self):
        """Precompute 8-neighbor offsets."""
        offs = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy != 0 or dx != 0:
                    offs.append((dy, dx))
        self.neighbor_offsets_8 = offs

    def _init_batch_state(self) -> dict:
        """Initialize state tensors for B parallel simulations."""
        B, H, W, S = self.B, self.H, self.W, self.max_settlements
        dev = self.device

        # Grid: B×H×W terrain codes
        grid = torch.zeros(B, H, W, dtype=torch.int8, device=dev)
        base_grid = torch.from_numpy(self.initial_grid_np.astype(np.int8)).to(dev)
        grid[:] = base_grid.unsqueeze(0)

        # Settlement arrays: B×S
        n_init = len(self.initial_settlements)
        S_x = torch.full((B, S), -1, dtype=torch.int16, device=dev)
        S_y = torch.full((B, S), -1, dtype=torch.int16, device=dev)
        S_pop = torch.zeros(B, S, device=dev)
        S_food = torch.zeros(B, S, device=dev)
        S_wealth = torch.zeros(B, S, device=dev)
        S_defense = torch.zeros(B, S, device=dev)
        S_tech = torch.zeros(B, S, device=dev)
        S_port = torch.zeros(B, S, dtype=torch.bool, device=dev)
        S_longship = torch.zeros(B, S, dtype=torch.bool, device=dev)
        S_owner = torch.zeros(B, S, dtype=torch.int16, device=dev)
        S_alive = torch.zeros(B, S, dtype=torch.bool, device=dev)

        for i, s in enumerate(self.initial_settlements):
            S_x[:, i] = s["x"]
            S_y[:, i] = s["y"]
            S_alive[:, i] = True
            S_port[:, i] = s.get("has_port", False)
            S_owner[:, i] = i  # each settlement starts as its own faction

            # Random initial stats (hidden — we draw from prior)
            # These will be calibrated via parameter estimation
            S_pop[:, i] = torch.FloatTensor(B).uniform_(1.0, 3.0).to(dev)
            S_food[:, i] = torch.FloatTensor(B).uniform_(0.3, 1.0).to(dev)
            S_wealth[:, i] = torch.FloatTensor(B).uniform_(0.2, 0.8).to(dev)
            S_defense[:, i] = torch.FloatTensor(B).uniform_(0.3, 0.7).to(dev)
            S_tech[:, i] = torch.FloatTensor(B).uniform_(0.1, 0.5).to(dev)

        # Number of active settlements (starts with initial, can grow)
        S_count = torch.full((B,), n_init, dtype=torch.int32, device=dev)

        return {
            "grid": grid,
            "S_x": S_x, "S_y": S_y,
            "S_pop": S_pop, "S_food": S_food,
            "S_wealth": S_wealth, "S_defense": S_defense,
            "S_tech": S_tech,
            "S_port": S_port, "S_longship": S_longship,
            "S_owner": S_owner, "S_alive": S_alive,
            "S_count": S_count,
        }

    def _count_adjacent_forest(self, grid: torch.Tensor) -> torch.Tensor:
        """Count adjacent forest cells for each grid position. Returns B×H×W."""
        forest = (grid == T_FOREST).float()
        count = torch.zeros_like(forest)
        for dy, dx in self.neighbor_offsets_8:
            count += torch.roll(torch.roll(forest, dy, dims=1), dx, dims=2)
        return count

    def _count_adjacent_ocean(self) -> torch.Tensor:
        """Count adjacent ocean cells (static). Returns H×W."""
        ocean = self.statics["is_ocean"].float()
        count = torch.zeros_like(ocean)
        for dy, dx in self.neighbor_offsets_8:
            count += torch.roll(torch.roll(ocean, dy, dims=0), dx, dims=1)
        return count

    def _compute_food_production(self, state: dict) -> torch.Tensor:
        """Compute food production for each settlement. Returns B×S."""
        p = self.params
        B, S = self.B, self.max_settlements
        adj_forest = self._count_adjacent_forest(state["grid"])  # B×H×W

        food_prod = torch.zeros(B, S, device=self.device)
        alive = state["S_alive"]  # B×S
        sx = state["S_x"]  # B×S  (int16)
        sy = state["S_y"]

        # Clamp indices to valid range for gathering
        sx_c = sx.long().clamp(0, self.W - 1)
        sy_c = sy.long().clamp(0, self.H - 1)

        for s in range(S):
            mask = alive[:, s]  # B
            if not mask.any():
                continue
            # Forest adjacency food
            forest_adj = adj_forest[torch.arange(B, device=self.device), sy_c[:, s], sx_c[:, s]]
            food_prod[:, s] = (p.food_base + forest_adj * p.food_per_forest) * mask.float()
            # Port sea bonus
            port_bonus = state["S_port"][:, s].float() * p.food_sea_bonus
            food_prod[:, s] += port_bonus * mask.float()

        return food_prod

    def _step_growth(self, state: dict) -> dict:
        """Growth phase: food production, population growth, ports, longships."""
        p = self.params
        B, S = self.B, self.max_settlements
        alive = state["S_alive"]

        food_prod = self._compute_food_production(state)
        maintenance = state["S_pop"] * p.pop_maintenance

        # Update food
        state["S_food"] = state["S_food"] + food_prod - maintenance
        state["S_food"] = state["S_food"].clamp(min=-2.0, max=5.0)

        # Population growth
        surplus = (state["S_food"] - p.expansion_min_food).clamp(min=0)
        pop_growth = surplus * p.pop_growth_rate * alive.float()
        state["S_pop"] = (state["S_pop"] + pop_growth).clamp(min=0, max=20.0)

        # Port building: coastal settlement, enough wealth, no port yet
        coastal = self.statics["is_coastal"]  # H×W
        can_build_port = ~state["S_port"] & alive
        if can_build_port.any():
            sx_c = state["S_x"].long().clamp(0, self.W - 1)
            sy_c = state["S_y"].long().clamp(0, self.H - 1)
            for s in range(S):
                m = can_build_port[:, s]
                if not m.any():
                    continue
                is_coastal_s = coastal[sy_c[:, s], sx_c[:, s]]
                wealthy = state["S_wealth"][:, s] > p.port_min_wealth
                r = torch.rand(B, device=self.device)
                build = m & is_coastal_s & wealthy & (r < p.port_prob)
                state["S_port"][:, s] = state["S_port"][:, s] | build

        # Longship building
        can_build_ls = state["S_port"] & ~state["S_longship"] & alive
        if can_build_ls.any():
            wealthy = state["S_wealth"] > p.longship_min_wealth
            r = torch.rand(B, S, device=self.device)
            build_ls = can_build_ls & wealthy & (r < p.longship_prob)
            state["S_longship"] = state["S_longship"] | build_ls

        # Wealth accumulation
        state["S_wealth"] = (state["S_wealth"] + 0.05 * alive.float()).clamp(max=3.0)

        return state

    def _step_expansion(self, state: dict) -> dict:
        """Try to expand: found new settlements near prosperous ones."""
        p = self.params
        B, S = self.B, self.max_settlements

        # Find next free slot
        n_alive = state["S_alive"].sum(dim=1)  # B
        can_expand = (state["S_pop"] > p.expansion_min_pop) & \
                     (state["S_food"] > p.expansion_min_food) & \
                     state["S_alive"]

        r_expand = torch.rand(B, S, device=self.device)
        try_expand = can_expand & (r_expand < p.expansion_prob)

        if not try_expand.any():
            return state

        # For each simulation, find first expanding settlement and try to place
        # (simplified: process one expansion per sim per turn for speed)
        grid = state["grid"]
        is_ocean = self.statics["is_ocean"]
        is_mountain = self.statics["is_mountain"]

        for b in range(B):
            exp_slots = try_expand[b].nonzero(as_tuple=False).squeeze(-1)
            if exp_slots.numel() == 0:
                continue
            # Find first free slot
            free_slots = (~state["S_alive"][b]).nonzero(as_tuple=False).squeeze(-1)
            if free_slots.numel() == 0:
                continue  # No space for new settlement

            # Try one random expanding settlement
            perm = torch.randperm(exp_slots.numel(), device=self.device)
            for idx in perm[:2]:  # Try up to 2
                src = exp_slots[idx].item()
                px, py = state["S_x"][b, src].item(), state["S_y"][b, src].item()

                # Find empty land nearby
                candidates = []
                for dy in range(-p.expansion_max_dist, p.expansion_max_dist + 1):
                    for dx in range(-p.expansion_max_dist, p.expansion_max_dist + 1):
                        nx, ny = int(px + dx), int(py + dy)
                        if 0 <= nx < self.W and 0 <= ny < self.H:
                            cell = grid[b, ny, nx].item()
                            if cell in (T_EMPTY, T_PLAINS) and \
                               not is_ocean[ny, nx] and not is_mountain[ny, nx]:
                                candidates.append((nx, ny))

                if not candidates:
                    continue

                # Pick random candidate
                ci = np.random.randint(len(candidates))
                nx, ny = candidates[ci]
                ns = free_slots[0].item()  # Use first free slot

                # Found new settlement
                state["S_x"][b, ns] = nx
                state["S_y"][b, ns] = ny
                state["S_alive"][b, ns] = True
                state["S_pop"][b, ns] = state["S_pop"][b, src] * 0.3
                state["S_food"][b, ns] = state["S_food"][b, src] * 0.3
                state["S_wealth"][b, ns] = state["S_wealth"][b, src] * 0.2
                state["S_defense"][b, ns] = 0.3
                state["S_tech"][b, ns] = state["S_tech"][b, src] * 0.5
                state["S_port"][b, ns] = False
                state["S_longship"][b, ns] = False
                state["S_owner"][b, ns] = state["S_owner"][b, src]

                # Check if coastal for auto-port
                coastal = self.statics["is_coastal"]
                is_coastal_here = coastal[ny, nx].item()
                grid[b, ny, nx] = T_PORT if (is_coastal_here and
                    np.random.random() < p.rebuild_as_port_prob) else T_SETTLEMENT
                if grid[b, ny, nx] == T_PORT:
                    state["S_port"][b, ns] = True
                break  # Only one expansion per sim per turn

        return state

    def _step_conflict(self, state: dict) -> dict:
        """Raiding phase."""
        p = self.params
        B, S = self.B, self.max_settlements

        # Determine who raids whom
        # Simplified: for each settlement, randomly decide to raid and pick target
        desperate = state["S_food"] < p.raid_desperate_threshold
        raid_prob = torch.where(desperate, p.raid_prob_desperate, p.raid_prob_normal)
        r = torch.rand(B, S, device=self.device)
        raids = state["S_alive"] & (r < raid_prob)

        if not raids.any():
            return state

        # Process raids (simplified batch version)
        # For each batch independently
        alive = state["S_alive"]
        sx = state["S_x"].float()
        sy = state["S_y"].float()

        for b in range(B):
            raiders = raids[b].nonzero(as_tuple=False).squeeze(-1)
            if raiders.numel() == 0:
                continue
            targets_alive = alive[b].nonzero(as_tuple=False).squeeze(-1)
            if targets_alive.numel() < 2:
                continue

            for ri in raiders:
                ri = ri.item()
                if not alive[b, ri]:
                    continue
                range_b = p.base_raid_range + (p.longship_range_bonus if state["S_longship"][b, ri] else 0)
                rsx, rsy = state["S_x"][b, ri].float(), state["S_y"][b, ri].float()

                # Find valid targets (different owner, within range, alive)
                best_target = -1
                best_score = float('inf')
                for ti in targets_alive:
                    ti = ti.item()
                    if ti == ri or not alive[b, ti]:
                        continue
                    if state["S_owner"][b, ti] == state["S_owner"][b, ri]:
                        continue
                    dist = ((state["S_x"][b, ti] - rsx)**2 + (state["S_y"][b, ti] - rsy)**2).sqrt().item()
                    if dist > range_b:
                        continue
                    # Prefer weak, close targets
                    score = dist + state["S_defense"][b, ti].item() * 3
                    if score < best_score:
                        best_score = score
                        best_target = ti

                if best_target < 0:
                    continue

                # Raid outcome
                success_prob = p.raid_success_prob - \
                    state["S_defense"][b, best_target].item() * p.raid_defense_factor
                success_prob = max(0.05, min(0.95, success_prob))

                if np.random.random() < success_prob:
                    # Successful raid
                    loot = state["S_food"][b, best_target].item() * p.raid_loot_fraction
                    wealth_loot = state["S_wealth"][b, best_target].item() * p.raid_loot_fraction
                    state["S_food"][b, ri] = state["S_food"][b, ri] + loot
                    state["S_wealth"][b, ri] = state["S_wealth"][b, ri] + wealth_loot
                    state["S_food"][b, best_target] = state["S_food"][b, best_target] - loot
                    state["S_defense"][b, best_target] = state["S_defense"][b, best_target] * 0.85
                    # Conquest
                    if np.random.random() < p.conquest_prob:
                        state["S_owner"][b, best_target] = state["S_owner"][b, ri]

        return state

    def _step_trade(self, state: dict) -> dict:
        """Trade phase: ports within range trade."""
        p = self.params
        B, S = self.B, self.max_settlements

        for b in range(B):
            ports = (state["S_alive"][b] & state["S_port"][b]).nonzero(as_tuple=False).squeeze(-1)
            if ports.numel() < 2:
                continue
            for i in range(ports.numel()):
                pi = ports[i].item()
                for j in range(i + 1, ports.numel()):
                    pj = ports[j].item()
                    dist = ((state["S_x"][b, pi] - state["S_x"][b, pj]).float()**2 +
                            (state["S_y"][b, pi] - state["S_y"][b, pj]).float()**2).sqrt().item()
                    if dist > p.trade_range:
                        continue
                    # Same faction can always trade; different factions too (unless at war - simplified: always trade)
                    state["S_food"][b, pi] = state["S_food"][b, pi] + p.trade_food_gain
                    state["S_food"][b, pj] = state["S_food"][b, pj] + p.trade_food_gain
                    state["S_wealth"][b, pi] = state["S_wealth"][b, pi] + p.trade_wealth_gain
                    state["S_wealth"][b, pj] = state["S_wealth"][b, pj] + p.trade_wealth_gain
                    state["S_tech"][b, pi] = state["S_tech"][b, pi] + p.trade_tech_gain
                    state["S_tech"][b, pj] = state["S_tech"][b, pj] + p.trade_tech_gain

        return state

    def _step_winter(self, state: dict) -> dict:
        """Winter: food loss, possible collapse."""
        p = self.params
        B, S = self.B, self.max_settlements

        # Draw winter severity for each simulation
        base_loss = torch.randn(B, device=self.device) * p.winter_std + p.winter_mean
        base_loss = base_loss.clamp(min=p.winter_min)

        # Severe winter
        severe = torch.rand(B, device=self.device) < p.winter_severe_prob
        base_loss = torch.where(severe, base_loss * p.winter_severe_mult, base_loss)

        # Apply to all settlements
        food_loss = base_loss.unsqueeze(1) * state["S_alive"].float()
        state["S_food"] = state["S_food"] - food_loss

        # Collapse: food too low or pop too low
        collapse = state["S_alive"] & (state["S_food"] < p.food_collapse_threshold)
        if collapse.any():
            # Disperse population to nearby friendly settlements
            for b in range(B):
                col_slots = collapse[b].nonzero(as_tuple=False).squeeze(-1)
                for ci in col_slots:
                    ci = ci.item()
                    state["S_alive"][b, ci] = False
                    state["S_food"][b, ci] = 0
                    state["S_pop"][b, ci] = 0
                    # Mark grid as ruin
                    cx = state["S_x"][b, ci].item()
                    cy = state["S_y"][b, ci].item()
                    if 0 <= cx < self.W and 0 <= cy < self.H:
                        state["grid"][b, cy, cx] = T_RUIN

        return state

    def _step_environment(self, state: dict) -> dict:
        """Environment: forest reclaims ruins, settlements rebuild ruins."""
        p = self.params
        B, H, W = self.B, self.H, self.W

        grid = state["grid"]
        ruin_mask = (grid == T_RUIN)

        if not ruin_mask.any():
            return state

        for b in range(B):
            ruins = ruin_mask[b].nonzero(as_tuple=False)  # N×2 (y,x)
            if ruins.numel() == 0:
                continue
            for r in ruins:
                ry, rx = r[0].item(), r[1].item()

                # Check if any alive settlement is nearby
                nearby_settlement = False
                best_donor = -1
                best_dist = float('inf')
                for si in range(self.max_settlements):
                    if not state["S_alive"][b, si]:
                        continue
                    sx_i = state["S_x"][b, si].item()
                    sy_i = state["S_y"][b, si].item()
                    dist = abs(sx_i - rx) + abs(sy_i - ry)
                    if dist <= p.rebuild_range:
                        nearby_settlement = True
                        if dist < best_dist:
                            best_dist = dist
                            best_donor = si

                if nearby_settlement and best_donor >= 0:
                    if np.random.random() < p.rebuild_prob:
                        # Rebuild
                        free_slots = (~state["S_alive"][b]).nonzero(as_tuple=False).squeeze(-1)
                        if free_slots.numel() > 0:
                            ns = free_slots[0].item()
                            coastal = self.statics["is_coastal"][ry, rx].item()
                            is_port = coastal and np.random.random() < p.rebuild_as_port_prob
                            grid[b, ry, rx] = T_PORT if is_port else T_SETTLEMENT
                            state["S_x"][b, ns] = rx
                            state["S_y"][b, ns] = ry
                            state["S_alive"][b, ns] = True
                            state["S_pop"][b, ns] = state["S_pop"][b, best_donor] * 0.2
                            state["S_food"][b, ns] = state["S_food"][b, best_donor] * 0.2
                            state["S_wealth"][b, ns] = 0.2
                            state["S_defense"][b, ns] = 0.3
                            state["S_tech"][b, ns] = state["S_tech"][b, best_donor] * 0.4
                            state["S_port"][b, ns] = is_port
                            state["S_longship"][b, ns] = False
                            state["S_owner"][b, ns] = state["S_owner"][b, best_donor]
                else:
                    # No nearby settlement — forest or plains reclamation
                    rr = np.random.random()
                    if rr < p.forest_reclaim_prob:
                        grid[b, ry, rx] = T_FOREST
                    elif rr < p.forest_reclaim_prob + p.ruin_plains_prob:
                        grid[b, ry, rx] = T_PLAINS

        return state

    def _sync_grid(self, state: dict) -> None:
        """Update grid with current alive settlement positions/types."""
        grid = state["grid"]
        B, S = self.B, self.max_settlements
        # Mark alive settlements on grid (overwrite their cell)
        for s in range(S):
            alive_b = state["S_alive"][:, s]
            if not alive_b.any():
                continue
            for b in range(B):
                if not alive_b[b]:
                    continue
                sx = state["S_x"][b, s].item()
                sy = state["S_y"][b, s].item()
                if 0 <= sx < self.W and 0 <= sy < self.H:
                    grid[b, sy, sx] = T_PORT if state["S_port"][b, s] else T_SETTLEMENT

    def run(self, n_years: int = 50) -> np.ndarray:
        """
        Run B simulations for n_years.
        Returns B×H×W×6 probability tensor (each sim gives one sample).
        """
        state = self._init_batch_state()
        self._sync_grid(state)

        for year in range(n_years):
            state = self._step_growth(state)
            state = self._step_expansion(state)
            state = self._step_conflict(state)
            state = self._step_trade(state)
            state = self._step_winter(state)
            state = self._step_environment(state)
            self._sync_grid(state)

        # Convert final grid to one-hot class probabilities (each sim = 1 sample)
        grid_np = state["grid"].cpu().numpy()  # B×H×W
        return grid_np

    def run_and_aggregate(self, n_years: int = 50) -> np.ndarray:
        """
        Run B simulations and aggregate into H×W×6 probability distribution.
        """
        grid_np = self.run(n_years)
        H, W = self.H, self.W
        B = self.B

        # Map terrain codes to class indices
        # 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10/11→0
        class_map = np.zeros(16, dtype=np.int8)
        class_map[0] = 0   # Empty
        class_map[1] = 1   # Settlement
        class_map[2] = 2   # Port
        class_map[3] = 3   # Ruin
        class_map[4] = 4   # Forest
        class_map[5] = 5   # Mountain
        class_map[10] = 0  # Ocean
        class_map[11] = 0  # Plains

        # Count class occurrences across all B simulations
        probs = np.zeros((H, W, 6), dtype=np.float32)
        grid_clipped = np.clip(grid_np, 0, 15).astype(np.int8)
        classes = class_map[grid_clipped]  # B×H×W

        for c in range(6):
            probs[:, :, c] = (classes == c).sum(axis=0) / B

        return probs


def run_multi_gpu(initial_grid: np.ndarray, initial_settlements: list,
                  params: SimParams, total_sims: int = 10000,
                  n_years: int = 50) -> np.ndarray:
    """
    Run total_sims simulations across all available GPUs.
    Returns H×W×6 aggregated probability distribution.
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        n_gpus = 1
        devices = ["cpu"]
    else:
        devices = [f"cuda:{i}" for i in range(n_gpus)]

    sims_per_gpu = total_sims // n_gpus
    H, W = initial_grid.shape
    all_probs = np.zeros((H, W, 6), dtype=np.float64)

    # Run on each GPU (can be parallelized with multiprocessing, but PyTorch GPU is already parallel)
    for dev_i, dev in enumerate(devices):
        print(f"  Running {sims_per_gpu} sims on {dev}...")
        sim = VikingSimulator(
            initial_grid=initial_grid,
            initial_settlements=initial_settlements,
            params=params,
            batch_size=sims_per_gpu,
            device=dev,
        )
        probs_i = sim.run_and_aggregate(n_years=n_years)
        all_probs += probs_i

    all_probs /= n_gpus
    return all_probs.astype(np.float32)
