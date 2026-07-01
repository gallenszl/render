"""Replicate production's persistent Blender + init_scene loop.
   Render 3 wash meshes 50 times each (sequential, in one Blender process)
   to see if wash emerges after N iterations."""
import bpy, os, sys, json, math, time
import numpy as np
from PIL import Image as PILImage

WASH_UIDS = ['65ad095e176940b98a2458656d9c56ed',
             'fbcaa2e44d784de5b8f971ee130741b6',
             'd32f3f2eedf2476fba8c6b4dce14cbc5']

# Also include NORMAL_COLOR obj as "carrier" to build up state faster (like prod does 125 obj)
NORMAL_UIDS = ['0000ecca9a234cae994be239f6fec552', '000074a334c541878360457c672b6c2e']

GLB_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/.objaverse/hf-objaverse-v1/glbs'
glb_idx = {}
for sub in sorted(os.listdir(GLB_BASE)):
    sp = os.path.join(GLB_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.glb'):
                glb_idx[fn[:-4]] = os.path.join(sp, fn)

OUT = '/home/z50057756/tmp/wash_persistent'
os.makedirs(OUT, exist_ok=True)


def init_render_prod():
    """Copy production init_render exactly (called ONCE)."""
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.render.resolution_x = 256
    sc.render.resolution_y = 256
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.film_transparent = True
    sc.render.use_persistent_data = False
    sc.cycles.samples = 128
    sc.cycles.filter_type = 'BOX'
    sc.cycles.filter_width = 1
    sc.cycles.diffuse_bounces = 1
    sc.cycles.glossy_bounces = 1
    sc.cycles.transmission_bounces = 3
    sc.cycles.transparent_max_bounces = 3
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPTIX'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    sc.cycles.device = 'GPU'
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = d.type in ('OPTIX', 'CUDA')


def init_scene_prod():
    """Copy production init_scene (per-obj cleanup)."""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for m in bpy.data.materials:
        bpy.data.materials.remove(m, do_unlink=True)
    for t in bpy.data.textures:
        bpy.data.textures.remove(t, do_unlink=True)
    for i in bpy.data.images:
        bpy.data.images.remove(i, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def lighting():
    bpy.ops.object.light_add(type='POINT', location=(4, 1, 6))
    bpy.context.object.data.energy = 1000
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
    top = bpy.context.object
    top.data.energy = 10000
    top.scale = (100, 100, 100)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, -10))
    bpy.context.object.data.energy = 1000


def normalize_hierarchy():
    import mathutils
    roots = [o for o in bpy.context.scene.objects if not o.parent and o.type == 'MESH']
    if not roots:
        roots = [o for o in bpy.context.scene.objects if not o.parent]
    if not roots: return False
    if len(roots) > 1:
        empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.collection.objects.link(empty)
        for r in roots: r.parent = empty
        scene = empty
    else:
        scene = roots[0]
    bmin = (math.inf,)*3; bmax = (-math.inf,)*3
    for o in bpy.context.scene.objects:
        if isinstance(o.data, bpy.types.Mesh):
            for c in o.bound_box:
                wv = o.matrix_world @ mathutils.Vector(c)
                bmin = tuple(min(a,b) for a,b in zip(bmin, wv))
                bmax = tuple(max(a,b) for a,b in zip(bmax, wv))
    ext = tuple(b-a for a,b in zip(bmin, bmax))
    if not all(math.isfinite(x) for x in (*bmin, *bmax)) or max(ext) < 1e-6:
        return False
    scale = 1.0 / max(ext)
    scene.scale = tuple(s * scale for s in scene.scale)
    bpy.context.view_layer.update()
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


def measure(png_path):
    if not os.path.exists(png_path): return None
    arr = np.array(PILImage.open(png_path))
    if arr.ndim != 3 or arr.shape[2] < 4: return None
    fg = arr[..., 3] > 127
    if fg.sum() < 100: return None
    rgb = arr[..., :3][fg].astype(np.float32)
    R, G, B = rgb.mean(axis=0)
    diff = max(abs(R-G), abs(G-B), abs(R-B))
    sat = ((rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1) >= 0.15).mean() * 100
    return {'R': float(R), 'G': float(G), 'B': float(B), 'diff': float(diff), 'sat': float(sat)}


# Init render ONCE (like production)
init_render_prod()

# Sequential loop over sequence [normal, normal, wash1, normal, wash2, normal, wash3, ...]
# Repeat this pattern to see if wash emerges after N iterations
SEQUENCE = []
for cycle in range(20):
    for u in NORMAL_UIDS:
        SEQUENCE.append(('normal', u))
    for u in WASH_UIDS:
        SEQUENCE.append(('wash', u))

results = []
for idx, (tag, uid) in enumerate(SEQUENCE):
    glb = glb_idx.get(uid)
    if not glb:
        print(f"[{idx}] MISSING {uid}", flush=True); continue
    init_scene_prod()
    lighting()
    try:
        bpy.ops.import_scene.gltf(filepath=glb, merge_vertices=True, import_shading='NORMALS')
    except Exception as e:
        print(f"[{idx}] {tag} {uid[:14]} import FAILED: {e}", flush=True); continue
    if not normalize_hierarchy():
        print(f"[{idx}] {tag} {uid[:14]} normalize FAILED", flush=True); continue
    add_camera(yaw=0.5, pitch=0.3, fov_deg=52.0)
    png = f"{OUT}/iter{idx:03d}_{tag}_{uid[:8]}.png"
    bpy.context.scene.render.filepath = png
    try:
        bpy.ops.render.render(write_still=True)
        stats = measure(png)
    except Exception as e:
        print(f"[{idx}] render FAILED: {e}", flush=True); stats = None
    if stats:
        print(f"[{idx:>3}] {tag:<6} {uid[:14]}  RGB=({stats['R']:.0f},{stats['G']:.0f},{stats['B']:.0f}) diff={stats['diff']:.0f} sat={stats['sat']:.0f}%  meshes={len(bpy.data.meshes)} imgs={len(bpy.data.images)}", flush=True)
        results.append({'idx': idx, 'tag': tag, 'uid': uid, **stats,
                        'meshes': len(bpy.data.meshes), 'imgs': len(bpy.data.images)})

with open(f'{OUT}/persistent_result.json', 'w') as f:
    json.dump(results, f, indent=2)

# Analysis: for each wash uid, print iter sequence of RGB
print("\n\n=== ITERATION ANALYSIS ===", flush=True)
from collections import defaultdict
by_uid = defaultdict(list)
for r in results:
    by_uid[r['uid']].append(r)
for uid, items in by_uid.items():
    print(f"\n{uid[:14]} ({items[0]['tag']}):")
    for r in items[:20]:
        print(f"  iter {r['idx']:>3}: RGB=({r['R']:.0f},{r['G']:.0f},{r['B']:.0f}) diff={r['diff']:.0f} sat={r['sat']:.0f}%")
