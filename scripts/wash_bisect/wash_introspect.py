"""Blender Python script: import WASH GLBs, dump shader node graph + image
   colorspace. Render 1 frame each so we can compare.

   Run with: blender --background --python wash_introspect.py
"""
import bpy, os, sys, json, math, time
import numpy as np
from PIL import Image as PILImage

# Hard-coded UIDs (will edit if I want to vary)
UIDS = {
    'WASH_HEAVY':  ['32d5b9d4db3f41d8a0f9b7380e8ef2cc', '0b3813b60dda4ecc880fbcbbed10b3c0',
                    '6cf41ee1a601438b8efed1cbc4d8a1f8', '65ad095e1769409484d8d4fa9c4ef88a',
                    'd1a388d93c0745408e9b5060196b5dad'],
    'WASH_INVERSE': ['ad28003205884849899e64d3257cd64e', 'cd14b616da35470eae6894259697314c',
                     '18145bc088cd4c5bbe35c7e6f4bd2c87', 'ee1fde7ec1514f1d97c01b0b81f5b7a3',
                     'ac1c48faad454b138748e975f9be929e'],
    'NORMAL_COLOR': ['31263bb85c694f54b7a7795d57ac9b40', '2d5107a9476d47cc8af3b71cba2c11c8',
                     'a3e8c51f850a4147b6c3f73e74f6f10c'],
}

# Build UID -> GLB path
GLB_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/.objaverse/hf-objaverse-v1/glbs'
glb_idx = {}
for sub in sorted(os.listdir(GLB_BASE)):
    sp = os.path.join(GLB_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.glb'):
                glb_idx[fn[:-4]] = os.path.join(sp, fn)

OUT_DIR = '/home/z50057756/tmp/wash_fix_bisect'
os.makedirs(OUT_DIR, exist_ok=True)


def init_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures,
              bpy.data.lights, bpy.data.cameras, bpy.data.node_groups):
        for it in list(c):
            try: c.remove(it, do_unlink=True)
            except Exception: pass


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'GPU'
    sc.cycles.samples = 32  # low for fast diagnosis
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPTIX'
    sc.render.resolution_x = 256
    sc.render.resolution_y = 256
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.world.use_nodes = True


def lighting():
    # Identical to blender_script/render.py
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
    top = bpy.context.object
    top.data.energy = 10000
    top.scale = (100, 100, 100)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, -10))
    bpy.context.object.data.energy = 1000
    bpy.ops.object.light_add(type='POINT', location=(4, 1, 6))
    bpy.context.object.data.energy = 1000


def normalize_scene():
    bmin = (math.inf,)*3; bmax = (-math.inf,)*3
    for o in bpy.context.scene.objects:
        if isinstance(o.data, bpy.types.Mesh):
            for v in o.data.vertices:
                wv = o.matrix_world @ v.co
                bmin = tuple(min(a,b) for a,b in zip(bmin, wv))
                bmax = tuple(max(a,b) for a,b in zip(bmax, wv))
    ext = tuple(b-a for a,b in zip(bmin, bmax))
    if not all(math.isfinite(x) for x in (*bmin, *bmax)) or max(ext) < 1e-6:
        return False
    scale = 1.0 / max(ext)
    center = tuple((a+b)/2 for a,b in zip(bmin, bmax))
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.matrix_world.translation = (
                (o.matrix_world.translation[0] - center[0]) * scale,
                (o.matrix_world.translation[1] - center[1]) * scale,
                (o.matrix_world.translation[2] - center[2]) * scale,
            )
            o.scale = (o.scale[0]*scale, o.scale[1]*scale, o.scale[2]*scale)
    bpy.context.view_layer.update()
    return True


def add_camera(yaw=0.5, pitch=0.3, fov_deg=60.0, r_frame=0.62):
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


def dump_materials():
    """Walk all materials, dump shader node graph + image colorspace info."""
    report = []
    for mat in bpy.data.materials:
        if not mat.use_nodes: continue
        info = {'name': mat.name, 'blend_method': mat.blend_method,
                'shadow_method': mat.shadow_method, 'use_backface_culling': mat.use_backface_culling,
                'principled_found': False, 'base_color_constant': None,
                'base_color_image': None, 'base_color_connected': False,
                'image_colorspace': None, 'image_size': None,
                'has_emission': False, 'has_alpha_connection': False}
        nt = mat.node_tree
        for node in nt.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                info['principled_found'] = True
                # Base Color input
                bc_input = node.inputs.get('Base Color')
                if bc_input:
                    if bc_input.is_linked:
                        info['base_color_connected'] = True
                        src = bc_input.links[0].from_node
                        if src.type == 'TEX_IMAGE' and src.image:
                            info['base_color_image'] = src.image.name
                            info['image_colorspace'] = src.image.colorspace_settings.name
                            info['image_size'] = list(src.image.size) if src.image.size[0] > 0 else None
                    else:
                        info['base_color_constant'] = list(bc_input.default_value)
                # Emission
                em_input = node.inputs.get('Emission Color') or node.inputs.get('Emission')
                if em_input and (em_input.is_linked or any(v > 0 for v in em_input.default_value[:3])):
                    info['has_emission'] = True
                # Alpha
                alpha_input = node.inputs.get('Alpha')
                if alpha_input and alpha_input.is_linked:
                    info['has_alpha_connection'] = True
        report.append(info)
    return report


def render(out_path):
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def measure(png_path):
    if not os.path.exists(png_path): return None
    arr = np.array(PILImage.open(png_path))
    if arr.ndim != 3 or arr.shape[2] < 4: return None
    fg = arr[..., 3] > 127
    if fg.sum() < 100: return None
    rgb = arr[..., :3][fg].astype(np.float32)
    R, G, B = rgb.mean(axis=0)
    diff = max(abs(R-G), abs(G-B), abs(R-B))
    sat = ((rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)).mean()
    return {'R': float(R), 'G': float(G), 'B': float(B), 'diff': float(diff),
            'sat': float(sat), 'n_fg_pct': float(fg.mean()*100)}


# Modes
MODES = {
    'A_baseline':  {'merge_vertices': True, 'import_shading': 'NORMALS'},
    'B_smooth':    {'merge_vertices': True, 'import_shading': 'SMOOTH'},
    'C_flat':      {'merge_vertices': True, 'import_shading': 'FLAT'},
    'D_no_merge':  {'merge_vertices': False, 'import_shading': 'NORMALS'},
    'E_no_merge_smooth': {'merge_vertices': False, 'import_shading': 'SMOOTH'},
}

all_results = []
for cls, uids in UIDS.items():
    for uid in uids:
        glb = glb_idx.get(uid)
        if not glb:
            print(f"  MISSING glb: {uid}", flush=True)
            continue
        for mode_name, kwargs in MODES.items():
            print(f"\n>>> {cls} {uid[:14]} mode={mode_name}", flush=True)
            init_scene()
            setup_render()
            lighting()
            try:
                bpy.ops.import_scene.gltf(filepath=glb, **kwargs)
            except Exception as e:
                print(f"  import FAILED: {e}", flush=True)
                continue
            if not normalize_scene():
                print(f"  normalize FAILED", flush=True)
                continue
            mats = dump_materials()
            add_camera(yaw=0.5, pitch=0.3, fov_deg=60.0)
            png_path = f"{OUT_DIR}/{uid[:14]}_{mode_name}.png"
            try:
                render(png_path)
                stats = measure(png_path)
            except Exception as e:
                print(f"  render FAILED: {e}", flush=True)
                stats = None
            all_results.append({
                'cls': cls, 'uid': uid, 'mode': mode_name,
                'stats': stats, 'n_materials': len(mats),
                'mats': mats,
            })
            if stats:
                print(f"  RGB=({stats['R']:.0f},{stats['G']:.0f},{stats['B']:.0f})  diff={stats['diff']:.1f}  sat={stats['sat']:.3f}", flush=True)

with open(f'{OUT_DIR}/result.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults -> {OUT_DIR}/result.json", flush=True)
