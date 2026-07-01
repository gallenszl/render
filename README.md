# RnG Phase 2 Render Pipeline

TRELLIS-based Objaverse rendering pipeline for CVPR'26 RnG paper (VGGT4LVSM sim-to-real).
Produces 40 posed views per mesh (RGB + depth + camera metadata) with sim-to-real
patches (A1/A2/R2 lookAt/FoV/radius jitter) and 3-layer quality gates.

**Status**: Phase 2 vanilla+Layers pipeline validated on 100/2000 obj smoke against
FluffyElephant (TRELLIS original) baseline. Wash bug (Cycles `use_persistent_data`
device cache leak) identified and fixed via toggle. See `DEVELOPMENT_JOURNAL.md`
for the full journey.

---

## Repository layout

```
render.py                     # Orchestrator (per-obj job sampling, tar+atomic move, batch flush)
blender_script/render.py      # Blender-side (TRELLIS init + Layer 1a/1b/1c + render loop)
utils.py                      # TRELLIS sphere_hammersley_sequence
scan_quality.py               # Post-render fg quality scan (produces blacklist.txt)
verify_dataset.py             # Dataset structure verification

sbatch_gpu_multi.sh           # Production 4-GPU × N-procs sbatch
sbatch_gpu_array.sh           # Single-GPU array fallback
sbatch_gpu_multi_sweep.sh     # N_PROCS_PER_GPU sweep tool
sbatch_cpu_array.sh           # CPU-only (debug)
sbatch_gso_sim2real_render.sh # GSO sim2real re-render (Phase 1b eval)
monitor_prod.sh               # Watch production job status

scripts/
├── audit/                    # fg/color audit tools (used for FE alignment validation)
├── wash_bisect/              # Root-cause bisect for wash bug (persistent_data leak)
└── sbatch_templates/         # 100 / 2000 obj smoke sbatch templates

DEVELOPMENT_JOURNAL.md        # Full dev history — every phase, rationale, decision
```

---

## Final render setting (Phase 2)

| Param | Value | Purpose |
|---|---|---|
| `r_frame` | 0.62 | Framing sphere inscribed in frustum (0.684 = TRELLIS orig) |
| `samples` (Cycles) | 128 | TRELLIS default |
| `bounces` (diff/gloss/transp/transm) | 1/1/3/3 | TRELLIS default |
| Resolution | 512×512 | RGBA PNG + 16-bit depth |
| `num_views` | 40 per obj | |
| **A1** LookAt offset | `max_ratio=0.15, zero_prob=0.20` | sim2real per-view centering jitter |
| **A2** per-obj FoV | `U(40°, 75°)` | matches handheld phone 1× wide camera |
| **A3** pitch | TRELLIS `sphere_hammersley_sequence` | 75/25 upper-hemisphere bias |
| **R2** radius jitter | `U(0.95, 1.10)` | per-view distance jitter |
| **R1** roll (train-side) | `±10°` (P2 patch in dataloader) | handheld roll |
| **Layer 1a** SkipRender | `non_finite_bbox` / `empty_bbox` / `min<1e-4·max` | degenerate mesh gate |
| **Layer 1b** Strategy E | `aspect_long>5 AND aspect_second>3` → `k=clamp(silhouette/0.95, 0.5, 1.0)` | thin-obj adaptive zoom |
| **Layer 1c** post-render alpha | view 0 `fg_ratio<0.005 → skip` | catch empty renders |
| **Wash fix** | `use_persistent_data` toggle per obj | Cycles cache leak fix (see §Wash Bug) |

---

## Quickstart

### 1. Environment

```bash
# Blender 4.2.9 LTS (bundled Python 3.11)
BLENDER=/path/to/blender-4.2.9-linux-x64/blender
BLPY=/path/to/blender-4.2.9-linux-x64/4.2/python/bin/python3.11

# OPTIX kernel cache (persist across jobs)
export OPTIX_CACHE_PATH=$HOME/.cache/OptixCache
mkdir -p $OPTIX_CACHE_PATH

# X11 user libs (needed on compute nodes without system libSM)
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH
```

### 2. Debug render (1 mesh)

```bash
$BLPY render.py \
    --object mesh.glb \
    --output_dir out/ \
    --num_views 40 \
    --resolution 512 \
    --device GPU \
    --samples 128 \
    --offset_max 0.15 \
    --radius_min_factor 0.95 --radius_max_factor 1.10 \
    --fov_min_deg 40 --fov_max_deg 75 \
    --r_frame 0.62 \
    --persistent
```

Output: `out/000.png`, `out/000_depth.png`, ..., `out/transforms.json`, `out/phases.json`.

### 3. Production 4-GPU × N=4 procs (via sbatch)

```bash
OBJ_LIST=/path/to/obj_list.txt \
OUTPUT_DIR=/path/to/render_out/ \
NUM_VIEWS=40 N_GPUS=4 N_PROCS_PER_GPU=4 \
sbatch --time=20:00:00 --cpus-per-task=96 --mem=1000G sbatch_gpu_multi.sh
```

- `obj_list.txt`: one absolute `.glb` path per line
- Renders to `/tmp/render_$SLURM_JOB_ID/scratch_g$i_p$j/` then atomic-move to `$OUTPUT_DIR/<uid>.tar`
- `.tar` = 40 RGB PNGs + 40 16-bit depth PNGs + `transforms.json` (per-frame camera) + `phases.json` (obj-level metadata)

### 4. 2000-obj smoke (verify pipeline changes)

See `scripts/sbatch_templates/sbatch_smoke_2000_fix2.sh` — 4 H200 × 4 procs, ~45-70 min, produces
1974 tar + 26 quar for random-42 seed sample of 44,798.

### 5. Post-render color audit

```bash
python scripts/audit/audit_color.py       # mean-RGB diff based (legacy)
python scripts/audit/reclassify_v2.py     # HSV S_pct + Lab chroma (recommended)
```

Produces `color_audit_v2.json` with per-obj classification:
- `NORMAL_COLOR` (66.5% typical): mesh has texture + we render it correctly
- `MESH_GRAY` (30.4%): mesh is inherently colorless (single-color baseColorFactor)
- `WASH_HEAVY / MILD / INVERSE`: render vs FE discrepancy classes (should be <5% after wash fix)

### 6. fg + clip metrics vs FE / old r44

```bash
python scripts/audit/verify_final_2000.py   # paired stats: VL vs FE vs old r44
```

---

## Wash Bug — key finding + fix

### Symptom
~6.4% of meshes rendered gray or with wrong texture in production, even though the mesh
GLB has valid `baseColorTexture` binding. Same mesh rendered 100× in a loop gave different
colors on different iterations (e.g., croissant rendered as brown, gray, white, or blue).

### Root cause
`use_persistent_data = True` (Cycles setting to share BVH across frames) also keeps the
**device-side** (GPU) texture cache across renders. Blender's Python-side `bpy.data.images.remove()`
doesn't reach the device cache. On subsequent renders, Cycles reuses old texture buffers by
internal pointer → texture from previous mesh leaks into current render.

### Fix (`blender_script/render.py:render_one_object`)
```python
# Toggle OFF before cleanup — forces Cycles to release device cache
bpy.context.scene.render.use_persistent_data = False
init_scene()
load_object(mesh_path)
# Toggle back ON — 40 views of THIS obj share BVH (fast)
bpy.context.scene.render.use_persistent_data = True
```

### Alternatives tested (all failed to fix wash)
- `frame_current += 1` — Cycles cache not frame-keyed
- `img.user_clear() + remove() + gc.collect()` — Python-side only, doesn't touch device
- Rename images to UUIDs — Cycles caches by internal pointer, not name

Only the `use_persistent_data` toggle reaches Cycles' device cache. See `scripts/wash_bisect/`
for the bisect harness.

### Cost
~1-11s overhead per obj (BVH rebuild + OPTIX pipeline init).
2000-obj wall: 43 min → 66 min (+53%).
44,798-obj estimate: 16h → 25h.

---

## Data outputs

```
render_out_2000_fix2/                   # 2000-obj toggle-fix (Phase 2 template)
├── <uid>.tar                           # per-obj tar (82 files, ~8 MB)
│   ├── <uid>/000.png ~ 039.png         # RGBA 512×512
│   ├── <uid>/000_depth.png ~ 039_depth.png  # 16-bit BW
│   ├── <uid>/transforms.json           # per-frame yaw/pitch/radius/fov/lookat_offset + depth min/max
│   └── <uid>/phases.json               # obj-level: k_zoom, a_ext, aspect_long, timing breakdown
├── quarantine.txt                      # <uid>\t<reason> for Layer 1a/1c skipped meshes
└── _logs/                              # per-worker Blender stdout logs
```

`transforms.json` schema per frame:
```json
{
  "file_path": "<uid>/000",
  "camera_angle_x": 1.048,       // FoV in radians (same as fov below)
  "fov": 1.048,                  // A2: per-obj FoV
  "yaw": 3.14,                   // A3: TRELLIS sphere_hammersley
  "pitch": 0.52,
  "radius": 1.35,                // A1: r_frame / sin(fov/2) × R2 × k_zoom
  "lookat_offset": [-0.02, 0.03, 0.0],  // A1: per-view centering offset
  "transform_matrix": [[...4×4...]],
  "depth": {"min": 0.85, "max": 2.60}
}
```

---

## Metric snapshot (2000-obj smoke, paired 1961 UIDs vs FE / old r44)

| Metric | vanilla+Layers (fixed) | old r44 (Phase 1a) | FE (TRELLIS orig) |
|---|---|---|---|
| fg mean | 0.189 | 0.151 | 0.212 |
| fg p50 | 0.165 | 0.135 | 0.190 |
| **VL/FE mean ratio** | **0.892** | 0.714 | 1.0 |
| VL/FE p50 ratio | 0.872 | 0.711 | 1.0 |
| per-obj median ratio vs FE | **0.891** | 0.716 | — |
| Total view-weighted clip | 10.55% | — | — |
| Normal obj clip (1866) | 9.22% | — | — |
| Trig obj (Strategy E, 108) | 33.54% | — | — |
| Wash rate (real bugs) | **~0%** | ~6-7% (est) | 0% |
| FoV range | [40.0°, 74.7°] | [25°, 70°] | 40° (fixed) |

**Improvement vs old r44**: per-obj fg distribution alignment to FE: **0.716 → 0.891 (+24%)**.
Wash bug fully eliminated.

---

## Full development journey

See `DEVELOPMENT_JOURNAL.md` for:
- Phase 1a training + P1-P7 dataloader patches
- Phase 2 render strategy iterations (MAD/Pct99.5 robust normalize rejected → vanilla+Layers)
- Wash bug diagnosis: from Blender internals to GPU device cache
- All 100/2000-obj smoke results + FE alignment analysis
- Full metric tables + decision rationale

---

## Attribution

- TRELLIS render base: [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
- Objaverse mesh source: Sketchfab (via Objaverse-XL, LVIS subset)
- FluffyElephant reference dataset: TRELLIS team

Contact: zhelun.shen25@imperial.ac.uk (Imperial College London, RnG project)
