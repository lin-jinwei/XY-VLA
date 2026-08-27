### 1.1 Core Example (`billboard_id=3`, default config is upper yellow frame — use ID as reference)

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through only the rectangular portal marked billboard_id=18" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

### 1.2 Other IDs (change the number in `--cmd` only)

Lower ring example (default config **`billboard_id=12`** ≈ orange frame; still use ID as reference):

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Pass through portal billboard number 12 only; stay parallel to the long edge of billboard_id=12." `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

Short phrase (suitable for batch scripts changing the last digit):

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Navigate through rectangular frame billboard_id=16; long-opening alignment; no other portals." `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

### 1.3 Alternative: Color Commands (legacy compatibility, lower precision)

Easy to confuse when 2 portals share the same color on upper/lower layers; use only when **ambiguity is acceptable** or for **temporary comparison**.

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through the yellow rectangular frame safely, align with the long opening" `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

---

## 2. Capability Test B — Obstacle Avoidance & Safety Semantics (still targeting portal traversal)

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through only billboard_id=3 safely; keep clearance from all other frames and obstacles" `
  --sim-steps 280 `
  --infer-every 5 `
  --xvla-steps 4 `
  --log-every 12 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.5
```

**Control**: Disable near-field repulsion (comparison experiment only):

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through billboard_id=3 only" `
  --no-local-avoidance `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

---

## 3. Capability Test C — Traverse Specified ID Only Under Multi-Frame Interference

**Test objective**: Whether **`only billboard_id=N`** / **`ignore other billboard markers`** in the instruction suppresses neighboring distractor frames.

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through ONLY billboard_id=7; numbered markers 6 and 8 are distractors — do not enter them." `
  --sim-steps 260 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Pass through portal billboard 16 only; all other billboard_id values must be skipped." `
  --sim-steps 280 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```
---

## 4. Capability Test D — Visual Alignment + Traversal (Two-Stage Semantics)

**Test objective**: **Align/center** first, then **traverse** — evaluate whether temporal language is interpreted by the model as pose/position first, then crossing.

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Center billboard_id=5 opening in view, then fly through it along the long opening only." `
  --sim-steps 300 `
  --infer-every 5 `
  --xvla-steps 4 `
  --log-every 12 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.5
```

Near-range gate pose estimation relies on **`fpv_slot_align`**, **`gate_pose_estimator`** (`opencv` / `xvla`); see `config.json`.

---

## 5. Capability Test E — Tilt / Narrow Slot Geometry Hints

Mid-layer frames in the scene have **random tilt/yaw**; explicitly requiring **tilted / narrow slot** in the instruction tests the model's response to geometric language (still combined with real simulation poses).

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through the tilted rectangular portal billboard_id=1; stay parallel to the long edge to avoid hitting the rails." `
  --sim-steps 280 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.5
```

---

## 6. Capability Test F — Motion Scale Trio (Smoke / Robust / Aggressive)

Same `--cmd`, only change dynamics-related parameters; for **video comparison**, best use **`--gui`** or increase `recording_scene_margin`.

**Smoke (small steps, easy to stabilize)**

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through billboard_id=3 only" `
  --sim-steps 240 `
  --infer-every 8 `
  --xvla-steps 2 `
  --log-every 12 `
  --speed 4 `
  --cmd-motion-amplify 2.0 `
  --cmd-min-step 0.04 `
  --infer-displacement-scale 1.2
```

**Robust (default)** — same as **§1.1**.

**Aggressive (faster approach, prone to going off-screen or overshoot)**

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through billboard_id=3 only" `
  --sim-steps 200 `
  --infer-every 4 `
  --xvla-steps 4 `
  --log-every 8 `
  --speed 6 `
  --cmd-motion-amplify 3.0 `
  --cmd-min-step 0.05 `
  --infer-displacement-scale 2.2
```

---

## 7. Capability Test G — Full Scene Language Catalog Ablation

**Catalog off**: Test pure vision + short instruction; compare the effect of **having vs. not having a numeric object list** on frame selection / path.

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through billboard_id=3 only" `
  --no-xvla-scene-catalog `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

---

## 8. Full 20-Frame Sequential Traversal vs `--cmd`

Code convention: when **`--cmd`** is present, **non-empty `task_sequence` in JSON is ignored**; language instruction only. For **20-frame sequential endurance**, **do not pass `--cmd`**, e.g.:

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --config e:/X-VLA/config.json `
  --scheme widowx_ee6d `
  --instruction "Sequential portal flight demo"
```

---

## 9. Top-Down Coarse Planning Phase1+2+3 (`cmd_coarse_plan_once`)

### 9.1 Copy-Paste Command — Single Frame Traversal (default Phase3 action)

Same `--cmd` as **§1.1**; with **`cmd_coarse_plan_once: true`** runs Phase3 instead of main simulation:

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly through only the rectangular portal marked billboard_id=3" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

Expected log: `[Phase 3][Refine] object '…' base_action=pass_through`.

### 9.2 Copy-Paste Commands — Explicit Five Actions (single-frame debug)

**Fly by** (does not emphasize center through):

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Fly by portal billboard number 3 without entering the opening center" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

**Orbit 2 laps**:

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Orbit billboard_id=3 twice for inspection, then pass through billboard_id=3" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

**Hover**:

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Hover over portal billboard number 5" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

**Collision** (geometry debug, not safe flight):

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Collide with rectangular portal billboard_id=7; ram the frame directly" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

### 9.3 Copy-Paste Command — Multi-Frame Chain + Phase3 Refinement

Same as **`cmd_multi_1.md` §5.1**; Phase3 turns each **feedback sphere contacted along the global trajectory** red sequentially and refines:

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --cmd "Pass through portal billboard number 2 then pass through portal billboard number 3, then pass through portal billboard number 5" `
  --sim-steps 200 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6
```

### 9.4 Disable Coarse Planning to Restore Closed-Loop Portal Traversal

Set **`"cmd_coarse_plan_once": false`** in **`config.json`** → `schemes.widowx_ee6d`, then run **§2–§7** commands to enter the X‑VLA main simulation loop.


## 10. Appendix: Full Command Template with `config` / `scheme` / `QS`

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --config e:/X-VLA/config.json `
  --scheme widowx_ee6d `
  --cmd "Fly through only billboard_id=3; align with the long opening; ignore other portals." `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6 `
  --qs-path e:/X-VLA/QS.json
```

If v‑vla is also needed: add `` ` `` at the end of the **`--qs-path`** line above, and **start a new line** for `--vvla-url` (no continuation on the last line); full example:

```powershell
& C:/Users/ydook/anaconda3/envs/xy-vla/python.exe d:/XY-VLA/run_xyVLA.py `
  --config e:/X-VLA/config.json `
  --scheme widowx_ee6d `
  --cmd "Fly through only billboard_id=3; align with the long opening; ignore other portals." `
  --sim-steps 220 `
  --infer-every 6 `
  --xvla-steps 4 `
  --log-every 10 `
  --speed 5 `
  --cmd-motion-amplify 2.5 `
  --cmd-min-step 0.045 `
  --infer-displacement-scale 1.6 `
  --qs-path e:/X-VLA/QS.json `
  --vvla-url http://127.0.0.1:9000/vvla
```

(`--vvla-url` should match your actual service address.) Seeing `[===> TransCMD]: ...` when multiple `Q` entries match is normal.
