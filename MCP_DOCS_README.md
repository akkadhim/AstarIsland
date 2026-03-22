# How MCP Was Used to Fetch Task Documentation

## Overview

This project uses an **MCP (Model Context Protocol) server** to give the AI agent access to the live Astar Island competition documentation. Instead of hard-coding rules or relying on outdated snapshots, the agent can query the task spec directly through a structured API.

---

## MCP Server Configuration

The server is registered in `~/.claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "nmiai": {
      "type": "http",
      "url": "https://mcp-docs.ainm.no/mcp"
    }
  }
}
```

| Field | Value |
|-------|-------|
| Server name | `nmiai` |
| Transport | HTTP (stateless, no SSE) |
| Endpoint | `https://mcp-docs.ainm.no/mcp` |

The server was registered using the Claude Code `/update-config` command:
> *"add MCP server named 'nmiai' with transport http at URL https://mcp-docs.ainm.no/mcp"*

---

## What the MCP Server Provides

The `nmiai` MCP server exposes the Astar Island competition documentation as queryable tools/resources. This includes:

- **Game mechanics**: all 5 simulation phases (Growth → Conflict → Trade → Winter → Environment)
- **Terrain types**: the 8 terrain codes and how they map to the 6 prediction classes
- **Scoring formula**: `score = 100 × exp(-3 × entropy-weighted KL divergence)`
- **API endpoints**: how to submit predictions, run viewport queries, fetch ground truth
- **Hidden parameters**: descriptions of all SimParams that govern the simulation
- **Viewport rules**: max 15×15 window, 50 queries per round shared across 5 seeds

This is the same content as `/workspace/Astar Island.txt` but served live — so if the competition rules update mid-season, the agent always reads the current version.

---

## How It Was Used in This Session

### Step 1 — Understanding the Rules
The agent read the full task specification at the start of the session. Key facts extracted:

```
Task:     Predict W×H×6 probability tensor (40×40 map, 6 terrain classes)
Budget:   50 viewport queries per round, shared across 5 seeds
Scoring:  score = 100 × exp(−3 × KL_entropy_weighted)
GT:       Ground truth = actual server simulation outcome after 50 years
```

### Step 2 — Identifying Simulation Gaps
After reading the rules, the agent cross-referenced them against `fast_sim.py` to find missing mechanics. Example finding:
- The spec describes **population dispersal on settlement collapse** (survivors scatter to nearby cells)
- This was confirmed implemented in `fast_sim.py::_phase_winter`

### Step 3 — Calibrating the Static Prior
The documentation defines all 6 terrain classes and their transition rules. Combined with ground truth data from past rounds (R1–R9), the agent built an empirically calibrated prior:

```python
# Plains cells (19,479 cells across R1-R9):
probs[plains] = [0.80, 0.14, 0.01, 0.015, 0.039, 0.005]
#               Empty  Set  Port  Ruin  Forest  Mtn

# Settlement cells (918 cells):
probs[settle] = [0.45, 0.31, 0.01, 0.030, 0.21, 0.004]
```

### Step 4 — Building the Simulation Pipeline
The full pipeline derived from the documentation:

```
Viewport Queries (50 budget)
        ↓
obs_calibrate: binary search on expansion_prob
to match observed settlement fraction
        ↓
Monte Carlo Simulation (150,000 runs × 50 years)
using FastViking GPU simulator
        ↓
blend_predictions(sim=0.75, prior=0.25)
        ↓
apply_floor_and_normalize (floor=0.01)
        ↓
Submit W×H×6 tensor via REST API
```

---

## Why MCP Matters Here

Without the MCP server, the agent would need to:
1. Ask the user to paste in the rules manually
2. Rely on outdated cached documentation
3. Guess mechanics from the code alone

With the MCP server:
- The agent can query game rules on-demand during reasoning
- Changes to scoring, viewport limits, or simulation mechanics are immediately visible
- The documentation is structured (tools/resources), not just raw text — the agent can ask specific questions like *"what phases does the simulation have?"* or *"how is the scoring KL weighted?"*

---

## Round Results (So Far)

| Round | Score | Rank | Key Change |
|-------|-------|------|------------|
| R6 | 20.6 | 155/186 | Baseline |
| R7 | 11.5 | 186/199 | Low settlement, bad prior |
| R8 | 38.7 | 173/214 | Improved calibration |
| R9 | **57.3** | 156/221 | New prior + sim_weight=0.75 |
| R10 | pending | — | 150k sims + v5 params |

The improvements from R8→R9 (+18.6 pts) came directly from correctly reading the rules and fixing the static prior to match observed GT distributions.

---

## Files Created During This Session

| File | Purpose |
|------|---------|
| `/tmp/train_v5.py` | Full training: 45 examples, 10 params, 2.5h |
| `/workspace/prediction.py` | Fixed static prior (empirically calibrated) |
| `/workspace/watcher.py` | Added Phase 2 obs_calibrate, AUTO_SUBMIT=False, 150k sims |
| `/workspace/checkpoints/best_params.json` | Optimized SimParams |
| `/workspace/checkpoints/training_data/r{1-9}_s{0-4}.npz` | Ground truth datasets |
