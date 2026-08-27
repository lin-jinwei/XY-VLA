# XY-VLA

Code for paper: XY-VLA: Zero-Shot Cross-Embodiment UAV VLA via X-VLA.

Language-guided portal navigation in PyBullet using the [X-VLA](https://github.com/2toINF/X-VLA) vision-language-action model. Fly a virtual end-effector through numbered rectangular portals, orbit frames, hover, or execute multi-leg missions from natural-language `--cmd` instructions.

Built on **X-VLA 0.9B (WidowX Edition)** with **EE6D** action semantics inside a WidowX arm workspace, extended with a three-phase navigation stack (coarse X-VLA planning → top-down A* → action-aware trajectory refinement).

## Features

- **Natural-language missions** — `billboard_id=N` portal targeting, multi-clause chains (`then`, `;`), and five basic actions: pass-through, fly-by, orbit, hover, collision (debug).
- **Three-phase navigation** — Phase 1 corridor + X-VLA coarse path, Phase 2 grid A* with portal-aware obstacles, Phase 3 per-object refinement and translucent feedback spheres.
- **Closed-loop or coarse-plan modes** — Per-step `/act` inference, or one-shot Phase 1+2+3 planning with optional simulation replay (no inference during replay).
- **Rich simulation scene** — 20 numbered `rect_frame` portals on upper/lower rings with random tilt; cubes, ramps, and local obstacle avoidance.
- **Recording & visualization** — Multi-view MP4/GIF under `recordings/`, top-down overlays, Phase 3 color feedback, optional GUI.
- **Configurable schemes** — `config.json` presets for GUI debug, fast batch runs, and demo playback (`--scheme 1|2|3|widowx_ee6d`).


## Test Result

#### 1-1_Fly through the yellow opening

<p>
<img src="recordings/1-1_Fly%20through%20the%20yellow%20opening/world_45deg.gif" width="49%" alt="45deg" />
<img src="recordings/1-1_Fly%20through%20the%20yellow%20opening/world_front.gif" width="49%" alt="front" />
</p>
<p>
<img src="recordings/1-1_Fly%20through%20the%20yellow%20opening/world_right.gif" width="49%" alt="right" />
<img src="recordings/1-1_Fly%20through%20the%20yellow%20opening/world_top.gif" width="49%" alt="top" />
</p>

#### 1-2_Pass through the nearest red portal

<p>
<img src="recordings/1-2_Pass%20through%20the%20nearest%20red%20portal/world_45deg.gif" width="49%" alt="45deg" />
<img src="recordings/1-2_Pass%20through%20the%20nearest%20red%20portal/world_front.gif" width="49%" alt="front" />
</p>
<p>
<img src="recordings/1-2_Pass%20through%20the%20nearest%20red%20portal/world_right.gif" width="49%" alt="right" />
<img src="recordings/1-2_Pass%20through%20the%20nearest%20red%20portal/world_top.gif" width="49%" alt="top" />
</p>

#### 2-1_Fly a figure-eight path in the air over the workspace

<p>
<img src="recordings/2-1_Fly%20a%20figure-eight%20path%20in%20the%20air%20over%20the%20workspace/world_45deg.gif" width="49%" alt="45deg" />
<img src="recordings/2-1_Fly%20a%20figure-eight%20path%20in%20the%20air%20over%20the%20workspace/world_front.gif" width="49%" alt="front" />
</p>
<p>
<img src="recordings/2-1_Fly%20a%20figure-eight%20path%20in%20the%20air%20over%20the%20workspace/world_right.gif" width="49%" alt="right" />
<img src="recordings/2-1_Fly%20a%20figure-eight%20path%20in%20the%20air%20over%20the%20workspace/world_top.gif" width="49%" alt="top" />
</p>

#### 2-2_Fly a racetrack oval in the air over the workspace

<p>
<img src="recordings/2-2_Fly%20a%20racetrack%20oval%20in%20the%20air%20over%20the%20workspace/world_45deg.gif" width="49%" alt="45deg" />
<img src="recordings/2-2_Fly%20a%20racetrack%20oval%20in%20the%20air%20over%20the%20workspace/world_front.gif" width="49%" alt="front" />
</p>
<p>
<img src="recordings/2-2_Fly%20a%20racetrack%20oval%20in%20the%20air%20over%20the%20workspace/world_right.gif" width="49%" alt="right" />
<img src="recordings/2-2_Fly%20a%20racetrack%20oval%20in%20the%20air%20over%20the%20workspace/world_top.gif" width="49%" alt="top" />
</p>

## Prerequisites

- **Python 3.10** (conda recommended)
- **CUDA GPU** recommended for X-VLA inference (CPU works for smoke tests)
- **Anaconda** or Miniconda
- X-VLA model weights in `xVLAModel/` (see [Model setup](#model-setup))

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd XY-VLA
```

### 2. Create the environment

One-click setup (creates conda env `xy-vla`, installs PyTorch and dependencies):

```bash
python set_enviroment_simple.py
```

Or with live pip output:

```bash
python set_enviroment.py
```

Recreate the environment from scratch:

```bash
python set_enviroment_simple.py --recreate
```

### 3. Model setup

Place the X-VLA WidowX checkpoint in `xVLAModel/`:

```bash
# Example: download from Hugging Face
# huggingface-cli download 2toINF/X-VLA-WidowX --local-dir xVLAModel
```

Required files include `model.safetensors`, `config.json`, `modeling_xvla.py`, `processing_xvla.py`, and tokenizer assets. Verify with:

```bash
python test_xVLA.py
```

## Quick start

### Single portal (language command)

```bash
python run_xyVLA.py \
  --cmd "Fly through only the rectangular portal marked billboard_id=3" \
  --sim-steps 200 \
  --infer-every 6 \
  --xvla-steps 4 \
  --speed 5
```

The demo auto-starts a local X-VLA server when `auto_start_xvla_server` is enabled in `config.json`.

### With GUI

```bash
python run_xyVLA.py --gui --scheme 1 \
  --cmd "Fly through billboard_id=3 only"
```

### Sequential 20-portal task (no `--cmd`)

Uses `task_sequence` from `config.json` (`schemes.widowx_ee6d`):

```bash
python run_xyVLA.py --scheme widowx_ee6d
```

> When `--cmd` is provided, a non-empty `task_sequence` in JSON is ignored.

### Multi-leg mission + Phase 3 refinement

```bash
python run_xyVLA.py \
  --cmd "Pass through portal billboard number 2 then pass through portal billboard number 3, then pass through portal billboard number 5" \
  --sim-steps 200 \
  --infer-every 6 \
  --xvla-steps 4
```

With default `cmd_coarse_plan_once: true`, this runs Phase 1+2+3 planning. Set `navigation_use_phase3_refined_path_in_sim: true` to replay the refined path in simulation without further `/act` calls.

## Configuration

Main file: `config.json`

| Key / scheme | Description |
|--------------|-------------|
| `scheme` | Top-level preset: `1` (GUI debug), `2` (fast headless), `3` (demo playback), or `widowx_ee6d` (full workspace). |
| `--scheme` | CLI override for top-level `scheme`. |
| `schemes.widowx_ee6d` | Workspace bounds, cubes, `task_sequence`, navigation flags, motion scales. |
| `cmd_coarse_plan_once` | Run Phase 1+2+3 once instead of closed-loop main sim. |
| `navigation_use_phase3_refined_path_in_sim` | Replay Phase 3 path in main simulation. |
| `auto_start_xvla_server` | Spawn local `/act` server on startup. |

See inline `_readme_*` keys in `config.json` for detailed option documentation.

## Project structure

```
XY-VLA/
├── run_xyVLA.py              # Main entry: PyBullet sim + X-VLA client
├── xvla_local_server.py      # Local X-VLA server helpers
├── config.json               # Schemes and runtime defaults
├── command.md                # Extended command examples and test matrix
├── test_xVLA.py              # Model + server smoke test
├── set_enviroment.py         # Conda environment setup (verbose)
├── set_enviroment_simple.py  # Conda environment setup (compact)
├── algorithms/
│   ├── phase1.py             # Phase 1 corridor + coarse planning
│   ├── phase2.py             # Phase 2 top-down A*
│   ├── phase3.py             # Phase 3 refinement + feedback zones
│   ├── phase3_actions.py     # Basic action geometry + keyword parsing
│   ├── phase3_xvla_actions.py# X-VLA action classification
│   ├── multi_leg.py          # Multi-leg mission stitching
│   ├── instruction_parse.py  # --cmd clause / billboard_id parsing
│   ├── portal_geometry.py    # Portal opening / collision geometry
│   └── phase_recording.py    # Recording and overlay rendering
├── xVLAModel/                # X-VLA WidowX weights + model code
├── recordings/               # Output videos and phase JSON artifacts
└── results.md                # Test-result gallery (four-view GIFs)
```

## Portal billboard IDs

The default scene places **20** `rect_frame` portals (`billboard_id` 1–20) on upper and lower rings. Prefer explicit IDs in instructions:

```
Fly through only billboard_id=18
Pass through portal billboard number 12 only
Orbit billboard_id=3 twice for inspection, then pass through billboard_id=10
```

Color-only commands work for legacy compatibility but may be ambiguous when two portals share a color on different layers.

## Common CLI flags

| Flag | Purpose |
|------|---------|
| `--cmd` | Natural-language mission (overrides `task_sequence`). |
| `--config` | Path to `config.json`. |
| `--scheme` | Config scheme key (`1`, `2`, `3`, `widowx_ee6d`). |
| `--sim-steps` | Simulation horizon. |
| `--infer-every` | Steps between X-VLA `/act` calls. |
| `--xvla-steps` | Action horizon per `/act` request. |
| `--gui` / `--no-gui` | PyBullet viewer. |
| `--no-local-avoidance` | Disable near-field obstacle repulsion. |
| `--no-xvla-scene-catalog` | Disable scene object catalog in instructions. |
| `--cmd-motion-amplify` | Scale language-only target displacement. |
| `--infer-displacement-scale` | Scale decoded target displacement per inference. |

Full examples: see [`command.md`](command.md).

## Recording output

When `record_visualization` is enabled, outputs are written under `recordings/<timestamp>/` (or a named mission folder):

- `sim/` — main simulation multi-view video / GIF
- `phase1/`, `phase2/`, `phase3/` — planning artifacts, overlays, and path JSON


Recorded missions are collected in [`results.md`](results.md): **1-1 ~ 2-26** show four GIFs from the mission folder; **3-1 ~ 3-15** show four GIFs from each folder's `sim/` directory.