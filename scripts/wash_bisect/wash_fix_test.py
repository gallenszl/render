"""Test fix modes on real wash cases using minimal replication of prod pipeline.
   Includes: A baseline (OPTIX denoise), B OIDN denoise, C no denoise samples=256,
             D low top light 3000W, E clamped emission.
"""
import bpy, os, sys, json, math, time
import numpy as np
from PIL import Image as PILImage

# Real wash cases (VL_S=0% VL_chroma=0, texture completely gone)
WASH_UIDS = [
    '65ad095e176940b98a2458656d9c56ed',  # croissant, all pure gray
    'fbcaa2e44d784de5b8f971ee130741b6',  # cutting board+tomato, all near-black
    'd32f3f2eedf2476fba8c6b4dce14cbc5',  # pear, all uniform gray
    '3941b1c6743348d2860c1c3d2e97ddc4',  # unknown wash target
]

GLB_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/.objaverse/hf-objaverse-v1/glbs'
glb_idx = {}
for sub in sorted(os.listdir(GLB_BASE)):
    sp = os.path.join(GLB_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.glb'):
                glb_idx[fn[:-4]] = os.path.join(sp, fn)

OUT = '/home/z50057756/tmp/wash_fix_bisect'
os.makedirs(OUT, exist_ok=True)


def init_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures,
              bpy.data.lights, bpy.data.cameras, bpy.data.node_groups):
        for it in list(c):
            try: c.remove(it, do_unlink=True)
            except Exception: pass


def setup_render_mode(mode):
    """Configure cycles device/denoiser/samples per test mode."""
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    for d in prefs.devices:
        d.use = 'GPU' in d.type or 'OPTIX' in d.type or 'CUDA' in d.type
    prefs.compute_device_type = 'OPTIX'  # keep OPTIX baseline
    sc.render.resolution_x = 256
    sc.render.resolution_y = 256
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.world.use_nodes = True
    # Bounces (TRELLIS orig)
    sc.cycles.diffuse_bounces = 1
    sc.cycles.glossy_bounces = 1
    sc.cycles.transmission_bounces = 3
    sc.cycles.transparent_max_bounces = 3
    if mode == 'A_baseline':
        sc.cycles.samples = 128
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPTIX'
    elif mode == 'B_OIDN':
        sc.cycles.samples = 128
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    elif mode == 'C_no_denoise':
        sc.cycles.samples = 256
        sc.cycles.use_denoising = False
    elif mode == 'D_low_top':
        sc.cycles.samples = 128
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPTIX'
    elif mode == 'E_no_transp':
        # No film_transparent — solid gray background
        sc.render.film_transparent = False
        sc.cycles.samples = 128
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPTIX'


def lighting(mode):
    """TRELLIS 3-light setup, with variant for D_low_top."""
    top_energy = 3000 if mode == 'D_low_top' else 10000
    bpy.ops.object.light_add(type='POINT', location=(4, 1, 6))
    bpy.context.object.data.energy = 1000
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
    top = bpy.context.object
    top.data.energy = top_energy
    top.scale = (100, 100, 100)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, -10))
    bpy.context.object.data.energy = 1000


def normalize_hierarchy():
    """Copy the production normalize_scene approach (parent-based, not per-mesh)."""
    # Find root objects
    roots = [o for o in bpy.context.scene.objects if not o.parent and o.type == 'MESH']
    if not roots:
        roots = [o for o in bpy.context.scene.objects if not o.parent]
    if len(roots) > 1:
        empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.collection.objects.link(empty)
        for r in roots: r.parent = empty
        scene = empty
    else:
        scene = roots[0]
    # bbox
    bmin = (math.inf,)*3; bmax = (-math.inf,)*3
    for o in bpy.context.scene.objects:
        if isinstance(o.data, bpy.types.Mesh):
            for c in o.bound_box:
                import mathutils
                wv = o.matrix_world @ mathutils.Vector(c)
                bmin = tuple(min(a,b) for a,b in zip(bmin, wv))
                bmax = tuple(max(a,b) for a,b in zip(bmax, wv))
    ext = tuple(b-a for a,b in zip(bmin, bmax))
    if not all(math.isfinite(x) for x in (*bmin, *bmax)) or max(ext) < 1e-6:
        return False
    scale = 1.0 / max(ext)
    scene.scale = tuple(s * scale for s in scene.scale)
    bpy.context.view_layer.update()
    # Re-eval bbox for offset
    import mathutils
    bmin2 = (math.inf,)*3; bmax2 = (-math.inf,)*3
    for o in bpy.context.scene.objects:
        if isinstance(o.data, bpy.types.Mesh):
            for c in o.bound_box:
                wv = o.matrix_world @ mathutils.Vector(c)
                bmin2 = tuple(min(a,b) for a,b in zip(bmin2, wv))
                bmax2 = tuple(max(a,b) for a,b in zip(bmax2, wv))
    center = tuple((a+b)/2 for a,b in zip(bmin2, bmax2))
    scene.matrix_world.translation = tuple(scene.matrix_world.translation[i] - center[i] for i in range(3))
    bpy.context.view_layer.update()
    return True


def add_camera(yaw=0.5, pitch=0.3, fov_deg=52.0, r_frame=0.62):
    fov_rad = math.radians(fov_deg)
    radius = r_frame / math.sin(fov_rad / 2)
    x = radius * math.cos(pitch) * math.cos(yaw)
    y = radius * math.cos(pitch) * math.sin(yaw)
    z = radius * math.sin(pitch)
    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.object
    cam.data.lens_unit = 'FOV'
    cam.data.angle = fov_rad
    import mathutils
    direction = mathutils.Vector((0,0,0)) - cam.location
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    bpy.context.scene.camera = cam


def render(out_path):
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def measure(png_path):
    if not os.path.exists(png_path): return None
    arr = np.array(PILImage.open(png_path))
    if arr.ndim != 3 or arr.shape[2] < 4:
        # E mode has no alpha — use everything
        if arr.shape[2] == 3:
            rgb = arr[..., :3].astype(np.float32).reshape(-1, 3)
            R, G, B = rgb.mean(axis=0)
            diff = max(abs(R-G), abs(G-B), abs(R-B))
            sat = ((rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1) >= 0.15).mean() * 100
            return {'R': float(R), 'G': float(G), 'B': float(B), 'diff': float(diff), 'sat': float(sat), 'n_fg_pct': 100.0}
        return None
    fg = arr[..., 3] > 127
    if fg.sum() < 100: return None
    rgb = arr[..., :3][fg].astype(np.float32)
    R, G, B = rgb.mean(axis=0)
    diff = max(abs(R-G), abs(G-B), abs(R-B))
    sat = ((rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1) >= 0.15).mean() * 100
    return {'R': float(R), 'G': float(G), 'B': float(B), 'diff': float(diff),
            'sat': float(sat), 'n_fg_pct': float(fg.mean()*100)}


MODES = ['A_baseline', 'B_OIDN', 'C_no_denoise', 'D_low_top', 'E_no_transp']

all_results = []
for uid in WASH_UIDS:
    glb = glb_idx.get(uid)
    if not glb:
        print(f"MISSING glb: {uid}", flush=True); continue
    for mode in MODES:
        print(f"\n>>> {uid[:14]} mode={mode}", flush=True)
        init_scene()
        setup_render_mode(mode)
        lighting(mode)
        try:
            bpy.ops.import_scene.gltf(filepath=glb, merge_vertices=True, import_shading='NORMALS')
        except Exception as e:
            print(f"  import FAILED: {e}", flush=True)
            continue
        if not normalize_hierarchy():
            print(f"  normalize FAILED", flush=True); continue
        add_camera(yaw=0.5, pitch=0.3, fov_deg=52.0)
        png = f"{OUT}/{uid[:14]}_{mode}.png"
        try:
            t0 = time.time()
            render(png)
            stats = measure(png)
            print(f"  RGB=({stats['R']:.0f},{stats['G']:.0f},{stats['B']:.0f})  diff={stats['diff']:.1f}  sat={stats['sat']:.1f}%  fg={stats['n_fg_pct']:.1f}%  wall={time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  render FAILED: {e}", flush=True)
            stats = None
        all_results.append({'uid': uid, 'mode': mode, 'stats': stats})

with open(f'{OUT}/wash_fix_result.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\n\n=== SUMMARY per-mode ===", flush=True)
for uid in WASH_UIDS:
    print(f"\n{uid[:14]}:")
    for mode in MODES:
        r = next((x for x in all_results if x['uid']==uid and x['mode']==mode), None)
        if r and r['stats']:
            s = r['stats']
            print(f"  {mode:<15} RGB=({s['R']:.0f},{s['G']:.0f},{s['B']:.0f}) diff={s['diff']:.1f} sat={s['sat']:.0f}%")
        else:
            print(f"  {mode:<15} FAILED")
