"""
Fully vectorized Norse civilization simulator on GPU.
No Python loops over batch dimension — everything uses PyTorch scatter/gather.

Terrain codes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains
Class indices: 0=Empty(0/10/11), 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional

T_EMPTY, T_SETTLE, T_PORT, T_RUIN, T_FOREST, T_MOUNTAIN, T_OCEAN, T_PLAINS = 0,1,2,3,4,5,10,11


@dataclass
class SimParams:
    food_per_forest: float = 0.30
    food_base: float = 0.10
    food_sea_bonus: float = 0.05
    pop_growth_rate: float = 0.04   # per unit net production
    pop_maintenance: float = 0.15
    expansion_min_pop: float = 2.5
    expansion_min_food: float = 0.3
    expansion_prob: float = 0.12
    expansion_max_dist: int = 3
    port_prob: float = 0.10
    port_min_wealth: float = 0.3
    longship_prob: float = 0.07
    longship_min_wealth: float = 0.5
    base_raid_range: int = 4
    longship_range_bonus: int = 5
    raid_prob_normal: float = 0.10
    raid_prob_desperate: float = 0.35
    raid_desperate_threshold: float = 0.35
    raid_success_prob: float = 0.55
    raid_defense_factor: float = 0.25
    raid_loot_fraction: float = 0.25
    conquest_prob: float = 0.18
    trade_range: int = 8
    trade_food_gain: float = 0.12
    trade_wealth_gain: float = 0.18
    trade_tech_gain: float = 0.04
    trade_cooldown_years: float = 2.0
    winter_mean: float = 0.45
    winter_std: float = 0.20
    winter_min: float = 0.12
    winter_severe_prob: float = 0.12
    winter_severe_mult: float = 2.2
    food_collapse_threshold: float = -0.15
    forest_reclaim_prob: float = 0.07
    ruin_plains_prob: float = 0.05
    rebuild_range: int = 3
    rebuild_prob: float = 0.10
    rebuild_as_port_prob: float = 0.35


CLASS_MAP = torch.zeros(16, dtype=torch.long)
CLASS_MAP[0]=0; CLASS_MAP[1]=1; CLASS_MAP[2]=2; CLASS_MAP[3]=3
CLASS_MAP[4]=4; CLASS_MAP[5]=5; CLASS_MAP[10]=0; CLASS_MAP[11]=0


class FastViking:
    """
    Batch Norse civilization simulator — fully vectorized.
    B = batch (parallel sims), S = max settlements (padded), H×W = grid.
    """

    def __init__(self, initial_grid: np.ndarray, initial_settlements: list,
                 params: SimParams, batch_size: int = 5000, device: str = "cuda:0"):
        self.p = params
        self.B = batch_size
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.H, self.W = initial_grid.shape
        live_sett = [s for s in initial_settlements if s.get("alive", True)]
        self.n_init = len(live_sett)
        self.S = max(250, self.n_init * 6)   # padded max settlements
        self.g0 = initial_grid.copy()
        self.live_sett = live_sett
        self._build_statics()

    def _build_statics(self):
        g, H, W, dev = self.g0, self.H, self.W, self.dev
        is_ocean = torch.from_numpy(g == T_OCEAN).to(dev)
        is_mtn   = torch.from_numpy(g == T_MOUNTAIN).to(dev)
        ocean_f  = is_ocean.float()[None, None]   # 1×1×H×W
        # coastal = adjacent to ocean, not ocean/mountain
        coast_f  = F.max_pool2d(ocean_f, 3, 1, 1).squeeze() > 0
        is_coast = coast_f & ~is_ocean & ~is_mtn
        # adjacent ocean count (for port food bonus)
        adj_oc   = F.conv2d(ocean_f, torch.ones(1,1,3,3,device=dev), padding=1).squeeze()
        self.is_ocean = is_ocean
        self.is_mtn   = is_mtn
        self.is_coast = is_coast    # H×W bool
        self.adj_oc   = adj_oc      # H×W float
        self.is_land  = ~is_ocean & ~is_mtn
        # Kernel for neighbour-forest counting
        self.neigh_k  = torch.ones(1,1,3,3,device=dev); self.neigh_k[0,0,1,1]=0

    # ── helpers ──────────────────────────────────────────────────────────

    def _adj_forest(self, grid: torch.Tensor) -> torch.Tensor:
        """B×H×W → B×H×W count of adjacent forest cells."""
        f = (grid == T_FOREST).float().unsqueeze(1)
        return F.conv2d(f, self.neigh_k, padding=1).squeeze(1)

    def _scatter_map(self, sx: torch.Tensor, sy: torch.Tensor,
                     vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Scatter vals[b,s] → map[b, sy[b,s], sx[b,s]] for alive (b,s).
        sx,sy: B×S int, vals: B×S float, mask: B×S bool
        Returns B×H×W float.
        """
        B, H, W = self.B, self.H, self.W
        out = torch.zeros(B, H*W, device=self.dev)
        flat = (sy.long().clamp(0,H-1)*W + sx.long().clamp(0,W-1)).clamp(0,H*W-1)
        flat_m = torch.where(mask, flat, torch.zeros_like(flat))  # put unmask at 0
        v_m    = torch.where(mask, vals.float(), torch.zeros_like(vals.float()))
        out.scatter_reduce_(1, flat_m, v_m, reduce='sum', include_self=True)
        return out.view(B, H, W)

    def _gather_map(self, field: torch.Tensor,
                    sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        """Gather field[b, sy[b,s], sx[b,s]]. field: B×H×W → B×S."""
        B, H, W = self.B, self.H, self.W
        flat = field.view(B, -1)
        idx  = (sy.long().clamp(0,H-1)*W + sx.long().clamp(0,W-1)).clamp(0,H*W-1)
        return flat.gather(1, idx)

    def _sync_grid(self, st: dict):
        """Vectorized: scatter all alive settlement positions to grid in one pass."""
        alive = st["S_alive"]  # B×S
        if not alive.any():
            return
        bi, si = alive.nonzero(as_tuple=True)
        sx = st["S_x"][bi, si].long().clamp(0, self.W-1)
        sy = st["S_y"][bi, si].long().clamp(0, self.H-1)
        typ = torch.where(st["S_port"][bi, si],
                          st["grid"].new_full((len(bi),), T_PORT),
                          st["grid"].new_full((len(bi),), T_SETTLE))
        st["grid"][bi, sy, sx] = typ

    # ── simulation phases ─────────────────────────────────────────────────

    def _phase_growth(self, st: dict):
        p = self.p
        alive = st["S_alive"]          # B×S
        grid  = st["grid"]

        # Food production
        adj_f = self._adj_forest(grid)                  # B×H×W
        af_at = self._gather_map(adj_f, st["S_x"], st["S_y"])  # B×S
        oc_at = self._gather_map(
            self.adj_oc.unsqueeze(0).expand(self.B,-1,-1),
            st["S_x"], st["S_y"])                        # B×S
        prod  = (p.food_base + af_at*p.food_per_forest +
                 st["S_port"].float()*p.food_sea_bonus*(oc_at>0).float()) * alive.float()
        maint = st["S_pop"] * p.pop_maintenance * alive.float()
        st["S_food"]   = (st["S_food"] + prod - maint).clamp(-4.0, 10.0)

        # Population grows from NET production (not stored food → avoids runaway growth)
        net_prod = (prod - maint).clamp(min=0)  # B×S
        st["S_pop"] = (st["S_pop"] + net_prod * p.pop_growth_rate).clamp(0, 15.0)

        # Port building
        coast_at = self._gather_map(
            self.is_coast.float().unsqueeze(0).expand(self.B,-1,-1),
            st["S_x"], st["S_y"]).bool()
        can_port = ~st["S_port"] & alive & coast_at & (st["S_wealth"]>p.port_min_wealth)
        st["S_port"] = st["S_port"] | (can_port & (torch.rand_like(st["S_food"]) < p.port_prob))

        # Longship
        can_ls = st["S_port"] & ~st["S_ship"] & alive & (st["S_wealth"]>p.longship_min_wealth)
        st["S_ship"] = st["S_ship"] | (can_ls & (torch.rand_like(st["S_food"]) < p.longship_prob))

        # Wealth / tech
        st["S_wealth"] = (st["S_wealth"] + 0.04*alive.float()).clamp(0,6.0)
        st["S_tech"]   = (st["S_tech"]   + 0.015*st["S_port"].float()*alive.float()).clamp(0,6.0)

    def _phase_expansion(self, st: dict):
        """
        Fully vectorized expansion — no Python loops over B or S.
        For each (b,s) trying to expand: sample K random target cells,
        pick first valid empty-land cell, assign to first free slot.
        """
        p = self.p
        alive  = st["S_alive"]  # B×S
        can_ex = alive & (st["S_pop"]>p.expansion_min_pop) & (st["S_food"]>p.expansion_min_food)
        if not can_ex.any(): return

        r = torch.rand(self.B, self.S, device=self.dev)
        try_ex = can_ex & (r < p.expansion_prob)  # B×S
        if not try_ex.any(): return

        d = int(p.expansion_max_dist)
        K = 12  # random attempts per (b,s)

        # ── sample K random target positions per (b,s) ──────────────────
        ox = torch.randint(-d, d+1, (self.B, self.S, K), device=self.dev)
        oy = torch.randint(-d, d+1, (self.B, self.S, K), device=self.dev)
        tx = (st["S_x"].long().unsqueeze(2) + ox).clamp(0, self.W-1)  # B×S×K
        ty = (st["S_y"].long().unsqueeze(2) + oy).clamp(0, self.H-1)

        # ── check validity: empty land ────────────────────────────────────
        flat = (ty * self.W + tx).clamp(0, self.H*self.W-1)           # B×S×K
        g_flat = st["grid"].view(self.B, self.H*self.W).long()         # B×HW
        g_at   = g_flat.gather(1, flat.view(self.B, self.S*K)).view(self.B, self.S, K)
        land_f = self.is_land.view(self.H*self.W)                      # HW bool
        l_at   = land_f[flat.view(-1)].view(self.B, self.S, K)        # B×S×K

        valid  = l_at & ((g_at == T_EMPTY) | (g_at == T_PLAINS)) & try_ex.unsqueeze(2)

        has_valid = valid.any(dim=2)                                    # B×S
        first_k   = valid.to(torch.uint8).argmax(dim=2)               # B×S

        chosen_tx = tx.gather(2, first_k.unsqueeze(2)).squeeze(2)     # B×S
        chosen_ty = ty.gather(2, first_k.unsqueeze(2)).squeeze(2)

        can_place = has_valid & try_ex                                  # B×S
        if not can_place.any(): return

        # ── assign free slots ─────────────────────────────────────────────
        # rank of each new settlement within its batch (0-indexed)
        new_rank = (can_place.long().cumsum(dim=1) - 1) * can_place.long()  # B×S

        # free_order[b,k] = index of k-th free slot in batch b
        # built by sorting ~alive descending (free slots come first)
        free_priority = (~alive).long() * self.S + \
                        torch.arange(self.S, device=self.dev).unsqueeze(0)
        free_order = free_priority.argsort(dim=1, descending=True)    # B×S

        max_rank = int(new_rank.max().item())
        # assigned_slot[b,s] = free_order[b, new_rank[b,s]]
        assigned_slot = free_order.gather(1, new_rank.clamp(0, self.S-1))  # B×S

        # guard: ensure the assigned slot is actually free
        is_free = (~alive).gather(1, assigned_slot)
        # guard: rank must be within available free slots per batch
        n_free  = (~alive).long().sum(dim=1, keepdim=True)             # B×1
        can_place = can_place & is_free & (new_rank < n_free)
        if not can_place.any(): return

        # ── scatter placement ─────────────────────────────────────────────
        bi, si = can_place.nonzero(as_tuple=True)                      # flat indices
        ns_i   = assigned_slot[bi, si]
        nx_new = chosen_tx[bi, si]
        ny_new = chosen_ty[bi, si]

        # coast → port assignment (vectorized)
        coast_f  = self.is_coast.view(self.H*self.W)
        flat_new = (ny_new * self.W + nx_new).clamp(0, self.H*self.W-1)
        use_port = coast_f[flat_new] & \
                   (torch.rand(len(bi), device=self.dev) < p.rebuild_as_port_prob)

        grid_vals = torch.where(use_port,
                                torch.full_like(nx_new, T_PORT,   dtype=st["grid"].dtype),
                                torch.full_like(nx_new, T_SETTLE, dtype=st["grid"].dtype))

        # update grid (last writer wins for conflicts — rare and acceptable)
        st["grid"][bi, ny_new, nx_new] = grid_vals

        # update settlement state arrays
        st["S_x"][bi, ns_i]     = nx_new.to(st["S_x"].dtype)
        st["S_y"][bi, ns_i]     = ny_new.to(st["S_y"].dtype)
        st["S_alive"][bi, ns_i] = True
        st["S_port"][bi, ns_i]  = use_port
        st["S_ship"][bi, ns_i]  = False
        st["S_pop"][bi, ns_i]   = st["S_pop"][bi, si]    * 0.3
        st["S_food"][bi, ns_i]  = st["S_food"][bi, si]   * 0.25
        st["S_wealth"][bi, ns_i]= 0.2
        st["S_def"][bi, ns_i]   = 0.3
        st["S_tech"][bi, ns_i]  = st["S_tech"][bi, si]   * 0.4
        st["S_owner"][bi, ns_i] = st["S_owner"][bi, si]

    def _phase_raids(self, st: dict):
        p = self.p
        alive = st["S_alive"]  # B×S
        desperate = st["S_food"] < p.raid_desperate_threshold
        rp = torch.where(desperate,
                         torch.full_like(st["S_food"],p.raid_prob_desperate),
                         torch.full_like(st["S_food"],p.raid_prob_normal))
        raids = alive & (torch.rand_like(st["S_food"]) < rp)
        if not raids.any(): return

        sx = st["S_x"].float(); sy = st["S_y"].float()

        # Chunk over B to avoid B×S×S OOM (each chunk is CHUNK×S×S)
        CHUNK = max(1, min(self.B, 2048 * 2048 // (self.S * self.S + 1)))
        best_score = torch.zeros(self.B, self.S, device=self.dev)
        best_tgt   = torch.zeros(self.B, self.S, dtype=torch.long, device=self.dev)
        eye_ss = torch.eye(self.S, dtype=torch.bool, device=self.dev).unsqueeze(0)

        for b0 in range(0, self.B, CHUNK):
            b1 = min(b0 + CHUNK, self.B)
            sx_c = sx[b0:b1]; sy_c = sy[b0:b1]
            dx = sx_c.unsqueeze(2) - sx_c.unsqueeze(1)
            dy = sy_c.unsqueeze(2) - sy_c.unsqueeze(1)
            dist = (dx**2 + dy**2).sqrt()
            rr = (p.base_raid_range + st["S_ship"][b0:b1].float()*p.longship_range_bonus).unsqueeze(2)
            so = st["S_owner"][b0:b1].unsqueeze(2) == st["S_owner"][b0:b1].unsqueeze(1)
            ta = alive[b0:b1].unsqueeze(1).expand(-1,self.S,self.S)
            valid = ta & ~so & (dist <= rr) & ~eye_ss
            sc = valid.float() * (1.0/(dist+1.0)) * torch.rand(b1-b0,self.S,self.S,device=self.dev)
            bs, bt = sc.max(dim=2)
            best_score[b0:b1] = bs
            best_tgt[b0:b1]   = bt

        has_tgt = raids & (best_score>0)
        if not has_tgt.any(): return

        success = has_tgt & (torch.rand_like(st["S_food"]) < p.raid_success_prob)
        if not success.any(): return

        tgt = best_tgt.clamp(0,self.S-1)
        food_tgt   = st["S_food"].gather(1,tgt)
        wealth_tgt = st["S_wealth"].gather(1,tgt)
        floot = food_tgt   * p.raid_loot_fraction * success.float()
        wloot = wealth_tgt * p.raid_loot_fraction * success.float()

        st["S_food"]   += floot
        st["S_wealth"] += wloot

        # Defenders lose (scatter_add)
        fl = torch.zeros_like(st["S_food"]); fl.scatter_add_(1,tgt,floot)
        wl = torch.zeros_like(st["S_food"]); wl.scatter_add_(1,tgt,wloot)
        dl = torch.zeros_like(st["S_food"]); dl.scatter_add_(1,tgt,0.12*success.float())
        st["S_food"]   -= fl
        st["S_wealth"] -= wl
        st["S_def"]     = (st["S_def"] - dl).clamp(0)

        # Conquest
        if p.conquest_prob > 0:
            cq = success & (torch.rand_like(st["S_food"]) < p.conquest_prob)
            if cq.any():
                attacker_owner = st["S_owner"]
                target_mask = torch.zeros_like(cq)
                target_mask.scatter_(1, tgt, cq)
                conquered_owner = torch.zeros_like(st["S_owner"])
                conquered_owner.scatter_(1, tgt, attacker_owner * cq.to(attacker_owner.dtype))
                st["S_owner"] = torch.where(target_mask, conquered_owner, st["S_owner"])

        if p.trade_cooldown_years > 0:
            cooldown = torch.full_like(st["S_food"], p.trade_cooldown_years)
            st["S_trade_cooldown"] = torch.where(success, torch.maximum(st["S_trade_cooldown"], cooldown), st["S_trade_cooldown"])
            target_cooldown = torch.zeros_like(st["S_food"])
            target_cooldown.scatter_reduce_(
                1,
                tgt,
                success.float() * p.trade_cooldown_years,
                reduce="amax",
                include_self=True,
            )
            st["S_trade_cooldown"] = torch.maximum(st["S_trade_cooldown"], target_cooldown)

    def _phase_trade(self, st: dict):
        p = self.p
        peaceful = st["S_trade_cooldown"] <= 0.0
        ports = st["S_port"] & st["S_alive"] & peaceful  # B×S
        if ports.sum() < 2: return

        sx = st["S_x"].float(); sy = st["S_y"].float()
        eye_ss = torch.eye(self.S, dtype=torch.bool, device=self.dev).unsqueeze(0)

        # Chunked B×S×S to avoid OOM
        CHUNK = max(1, min(self.B, 2048 * 2048 // (self.S * self.S + 1)))
        n_trade = torch.zeros(self.B, self.S, device=self.dev)
        for b0 in range(0, self.B, CHUNK):
            b1 = min(b0 + CHUNK, self.B)
            sx_c = sx[b0:b1]; sy_c = sy[b0:b1]
            ports_c = ports[b0:b1]
            dx = sx_c.unsqueeze(2) - sx_c.unsqueeze(1)
            dy = sy_c.unsqueeze(2) - sy_c.unsqueeze(1)
            dist = (dx**2 + dy**2).sqrt()
            p_i = ports_c.unsqueeze(2); p_j = ports_c.unsqueeze(1)
            tradeable = p_i & p_j & (dist <= p.trade_range) & ~eye_ss
            n_trade[b0:b1] = tradeable.float().sum(dim=2).clamp(max=5.0)

        st["S_food"]   += n_trade * p.trade_food_gain   * ports.float()
        st["S_wealth"] += n_trade * p.trade_wealth_gain * ports.float()
        st["S_tech"]   += n_trade * p.trade_tech_gain   * ports.float()

    def _phase_winter(self, st: dict):
        p = self.p
        alive = st["S_alive"]
        sev = (torch.randn(self.B,device=self.dev)*p.winter_std+p.winter_mean).clamp(min=p.winter_min)
        bad  = torch.rand(self.B,device=self.dev) < p.winter_severe_prob
        sev  = torch.where(bad, sev*p.winter_severe_mult, sev)
        st["S_food"] -= sev.unsqueeze(1) * alive.float()

        collapse = alive & (st["S_food"] < p.food_collapse_threshold)
        if collapse.any():
            # Disperse population to nearby friendly settlements BEFORE zeroing
            alive_nc = alive & ~collapse  # non-collapsed alive settlements
            if alive_nc.any():
                # Build collapse pop map: B×H×W
                collapse_pop_map = self._scatter_map(
                    st["S_x"], st["S_y"], st["S_pop"] * collapse.float(), collapse)
                # Build alive (non-collapsed) map: B×H×W
                alive_map = self._scatter_map(
                    st["S_x"], st["S_y"],
                    torch.ones(self.B, self.S, device=self.dev), alive_nc)
                # Sum-pool to find nearby collapsed pop and nearby alive count
                R = int(p.rebuild_range)
                ks = 2 * R + 1
                pool_k = torch.ones(1, 1, ks, ks, device=self.dev)
                nearby_cpop = F.conv2d(
                    collapse_pop_map.unsqueeze(1), pool_k, padding=R).squeeze(1)
                nearby_alive = F.conv2d(
                    alive_map.unsqueeze(1).float(), pool_k, padding=R).squeeze(1)
                # Each alive settlement gets its equal share of nearby collapsed pop
                per_sett = nearby_cpop / (nearby_alive + 1e-6)  # B×H×W
                gained = self._gather_map(per_sett, st["S_x"], st["S_y"]) * alive_nc.float()
                # 35% of the per-settlement share is absorbed (rest lost to chaos)
                dispersal = gained.clamp(0, 3.0) * 0.35
                st["S_pop"]  = (st["S_pop"]  + dispersal).clamp(0, 15.0)
                st["S_food"] = (st["S_food"] + dispersal * 0.2).clamp(-4.0, 10.0)

            # Vectorized: mark collapsed settlement positions as ruin
            bi, si = collapse.nonzero(as_tuple=True)
            sx = st["S_x"][bi, si].long().clamp(0, self.W-1)
            sy = st["S_y"][bi, si].long().clamp(0, self.H-1)
            st["grid"][bi, sy, sx] = T_RUIN
            st["S_alive"] = alive & ~collapse
            st["S_food"][collapse]  = 0
            st["S_pop"][collapse]   = 0

    def _phase_environment(self, st: dict):
        p = self.p
        grid, alive = st["grid"], st["S_alive"]
        ruins = (grid == T_RUIN)
        if not ruins.any(): return

        # Build settlement proximity map using scatter
        settle_map = self._scatter_map(st["S_x"], st["S_y"],
                                        torch.ones(self.B,self.S,device=self.dev),
                                        alive)  # B×H×W
        # Dilate to rebuild_range
        ks = 2*p.rebuild_range+1
        near_settle = F.max_pool2d(settle_map.unsqueeze(1).float(),
                                    ks, 1, p.rebuild_range).squeeze(1) > 0  # B×H×W

        ruin_w_settle  = ruins & near_settle
        ruin_no_settle = ruins & ~near_settle

        coast_b = self.is_coast.unsqueeze(0).expand(self.B,-1,-1)
        r1 = torch.rand(self.B,self.H,self.W,device=self.dev)
        r2 = torch.rand(self.B,self.H,self.W,device=self.dev)
        r3 = torch.rand(self.B,self.H,self.W,device=self.dev)

        rebuild  = ruin_w_settle  & (r1 < p.rebuild_prob)
        as_port  = rebuild & coast_b & (r2 < p.rebuild_as_port_prob)
        as_sett  = rebuild & ~as_port

        do_forest= ruin_no_settle & (r3 < p.forest_reclaim_prob)
        do_plains= ruin_no_settle & ~do_forest & (r3 < p.forest_reclaim_prob+p.ruin_plains_prob)

        grid[as_sett]  = T_SETTLE
        grid[as_port]  = T_PORT
        grid[do_forest]= T_FOREST
        grid[do_plains] = T_PLAINS

        # Track rebuilt settlements in state so they can grow/expand in future years
        # Flatten rebuilt to B×(H*W) and assign free slots using cumsum (vectorized)
        rebuilt = (as_sett | as_port).view(self.B, self.H * self.W)  # B×HW
        if rebuilt.any():
            alive_now = st["S_alive"]
            free_priority = (~alive_now).long() * self.S + \
                            torch.arange(self.S, device=self.dev).unsqueeze(0)
            free_order = free_priority.argsort(dim=1, descending=True)  # B×S
            n_free = (~alive_now).long().sum(dim=1)  # B

            # For each batch, take only first n_free rebuilt cells
            # Assign rank within batch using cumsum
            new_rank_flat = (rebuilt.long().cumsum(dim=1) - 1) * rebuilt.long()  # B×HW

            # Flatten for scatter: only process cells where rank < n_free
            bi_f, hw_f = rebuilt.nonzero(as_tuple=True)
            if len(bi_f) > 0:
                rank_f = new_rank_flat[bi_f, hw_f]
                valid = rank_f < n_free[bi_f]
                if valid.any():
                    bi_v = bi_f[valid]; hw_v = hw_f[valid]; rk_v = rank_f[valid]
                    ry_v = (hw_v // self.W).to(torch.int16)
                    rx_v = (hw_v  % self.W).to(torch.int16)
                    ns_v = free_order[bi_v, rk_v]
                    is_port_v = as_port.view(self.B, self.H*self.W)[bi_v, hw_v]

                    donor_idx = torch.full((len(bi_v),), -1, dtype=torch.long, device=self.dev)
                    donor_dist = torch.full((len(bi_v),), 10**9, dtype=torch.long, device=self.dev)
                    for s in range(self.S):
                        alive_s = alive_now[bi_v, s]
                        if not alive_s.any():
                            continue
                        dx = (st["S_x"][bi_v, s].long() - rx_v.long()).abs()
                        dy = (st["S_y"][bi_v, s].long() - ry_v.long()).abs()
                        dist = dx + dy
                        better = alive_s & (dist <= p.rebuild_range) & (dist < donor_dist)
                        donor_idx = torch.where(better, torch.full_like(donor_idx, s), donor_idx)
                        donor_dist = torch.where(better, dist, donor_dist)

                    has_donor = donor_idx >= 0
                    if has_donor.any():
                        bi_d = bi_v[has_donor]
                        ns_d = ns_v[has_donor]
                        rx_d = rx_v[has_donor]
                        ry_d = ry_v[has_donor]
                        donor_d = donor_idx[has_donor]
                        port_d = is_port_v[has_donor]

                        st["S_x"][bi_d, ns_d] = rx_d
                        st["S_y"][bi_d, ns_d] = ry_d
                        st["S_alive"][bi_d, ns_d] = True
                        st["S_port"][bi_d, ns_d] = port_d
                        st["S_ship"][bi_d, ns_d] = False
                        st["S_pop"][bi_d, ns_d] = (st["S_pop"][bi_d, donor_d] * 0.20).clamp(0.6, 4.0)
                        st["S_food"][bi_d, ns_d] = (st["S_food"][bi_d, donor_d] * 0.20).clamp(0.2, 3.0)
                        st["S_wealth"][bi_d, ns_d] = (st["S_wealth"][bi_d, donor_d] * 0.25).clamp(0.1, 2.0)
                        st["S_def"][bi_d, ns_d] = (st["S_def"][bi_d, donor_d] * 0.65).clamp(0.2, 2.0)
                        st["S_tech"][bi_d, ns_d] = (st["S_tech"][bi_d, donor_d] * 0.40).clamp(0.05, 3.0)
                        st["S_owner"][bi_d, ns_d] = st["S_owner"][bi_d, donor_d]
                        st["S_trade_cooldown"][bi_d, ns_d] = 0.0

        st["S_trade_cooldown"] = (st["S_trade_cooldown"] - 1.0).clamp(min=0.0)

    def run(self, n_years: int = 50) -> np.ndarray:
        """Run B sims × n_years. Return B×H×W numpy grid."""
        st = self._init_state()
        self._sync_grid(st)
        for yr in range(n_years):
            self._phase_growth(st)
            self._phase_expansion(st)
            self._phase_raids(st)
            self._phase_trade(st)
            self._phase_winter(st)
            self._phase_environment(st)
            self._sync_grid(st)
        return st["grid"].cpu().numpy()

    def run_with_state(self, n_years: int = 50):
        """Run B sims × n_years and return final grid and simulator state."""
        st = self._init_state()
        self._sync_grid(st)
        for yr in range(n_years):
            self._phase_growth(st)
            self._phase_expansion(st)
            self._phase_raids(st)
            self._phase_trade(st)
            self._phase_winter(st)
            self._phase_environment(st)
            self._sync_grid(st)
        return st["grid"].cpu().numpy(), st

    def run_and_aggregate(self, n_years: int = 50) -> np.ndarray:
        """Run B sims and return H×W×6 probability distribution."""
        grid_np = self.run(n_years)  # B×H×W
        B, H, W = grid_np.shape
        cm = CLASS_MAP.numpy()
        probs = np.zeros((H,W,6), dtype=np.float32)
        clipped = np.clip(grid_np, 0, 15).astype(np.int32)
        classes = cm[clipped]  # B×H×W
        for c in range(6):
            probs[:,:,c] = (classes==c).mean(axis=0)
        return probs

    def run_and_summarize(self, n_years: int = 50) -> dict:
        """
        Run B sims and return coarse summary statistics for observation calibration.
        """
        grid_np, st = self.run_with_state(n_years)
        B, H, W = grid_np.shape
        cm = CLASS_MAP.numpy()
        clipped = np.clip(grid_np, 0, 15).astype(np.int32)
        classes = cm[clipped]

        class_probs = np.zeros((H, W, 6), dtype=np.float32)
        for c in range(6):
            class_probs[:, :, c] = (classes == c).mean(axis=0)

        alive = st["S_alive"]
        alive_n = int(alive.sum().item())
        if alive_n > 0:
            pop_mean = float(st["S_pop"][alive].float().mean().item())
            food_mean = float(st["S_food"][alive].float().mean().item())
            wealth_mean = float(st["S_wealth"][alive].float().mean().item())
            def_mean = float(st["S_def"][alive].float().mean().item())
            port_frac = float(st["S_port"][alive].float().mean().item())
        else:
            pop_mean = food_mean = wealth_mean = def_mean = port_frac = 0.0

        return {
            "class_probs": class_probs,
            "sett_frac": float((class_probs[:, :, 1] + class_probs[:, :, 2]).mean()),
            "ruin_frac": float(class_probs[:, :, 3].mean()),
            "pop_mean": pop_mean,
            "food_mean": food_mean,
            "wealth_mean": wealth_mean,
            "def_mean": def_mean,
            "port_frac": port_frac,
        }

    def _init_state(self) -> dict:
        B, S, H, W, dev = self.B, self.S, self.H, self.W, self.dev
        g0 = torch.from_numpy(self.g0.astype(np.int16)).to(dev)
        grid = g0.unsqueeze(0).expand(B,H,W).clone()

        S_x     = torch.full((B,S),-1,dtype=torch.int16,device=dev)
        S_y     = torch.full((B,S),-1,dtype=torch.int16,device=dev)
        S_pop   = torch.zeros(B,S,device=dev)
        S_food  = torch.zeros(B,S,device=dev)
        S_wealth= torch.zeros(B,S,device=dev)
        S_def   = torch.zeros(B,S,device=dev)
        S_tech  = torch.zeros(B,S,device=dev)
        S_port  = torch.zeros(B,S,dtype=torch.bool,device=dev)
        S_ship  = torch.zeros(B,S,dtype=torch.bool,device=dev)
        S_owner = torch.zeros(B,S,dtype=torch.int16,device=dev)
        S_alive = torch.zeros(B,S,dtype=torch.bool,device=dev)
        S_trade_cooldown = torch.zeros(B,S,device=dev)

        for i,s in enumerate(self.live_sett[:S]):
            S_x[:,i]=s["x"]; S_y[:,i]=s["y"]
            S_alive[:,i]=True; S_port[:,i]=s.get("has_port",False); S_owner[:,i]=i
            S_pop[:,i]   = torch.FloatTensor(B).uniform_(1.5,3.0).to(dev)
            S_food[:,i]  = torch.FloatTensor(B).uniform_(0.6,1.2).to(dev)
            S_wealth[:,i]= torch.FloatTensor(B).uniform_(0.3,0.7).to(dev)
            S_def[:,i]   = torch.FloatTensor(B).uniform_(0.4,0.7).to(dev)
            S_tech[:,i]  = torch.FloatTensor(B).uniform_(0.2,0.5).to(dev)

        return dict(grid=grid,S_x=S_x,S_y=S_y,S_pop=S_pop,S_food=S_food,
                    S_wealth=S_wealth,S_def=S_def,S_tech=S_tech,
                    S_port=S_port,S_ship=S_ship,S_owner=S_owner,S_alive=S_alive,
                    S_trade_cooldown=S_trade_cooldown)


def run_multi_gpu(initial_grid: np.ndarray, initial_settlements: list,
                  params: SimParams, total_sims: int = 80000,
                  n_years: int = 50) -> np.ndarray:
    """Run total_sims across all GPUs in parallel (separate processes per GPU)."""
    import torch.multiprocessing as mp

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if n_gpus == 0: n_gpus = 1

    devices = [f"cuda:{i}" for i in range(n_gpus)] if torch.cuda.is_available() else ["cpu"]
    sims_per = total_sims // n_gpus

    H, W = initial_grid.shape
    all_probs = []

    for dev in devices:
        sim = FastViking(initial_grid, initial_settlements, params, sims_per, dev)
        p = sim.run_and_aggregate(n_years)
        all_probs.append(p)

    return np.mean(all_probs, axis=0).astype(np.float32)
