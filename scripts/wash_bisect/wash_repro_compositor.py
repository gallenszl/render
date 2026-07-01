"""Reproduce wash: add init_nodes-style compositor to minimal test.
   If wash reproduces here, compositor is the culprit.
"""
import bpy, os, sys, json, math, time
import numpy as np
from PIL import Image as PILImage

# Same wash UIDs
WASH_UIDS = ['65ad095e176940b98a2458656d9c56ed',  # croissant
             'fbcaa2e44d784de5b8f971ee130741b6',  # cutting board
             'd32f3f2eedf2476fba8c6b4dce14cbc5']  # pear

GLB_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/.objaverse/hf-objaverse-v1/glbs'
glb_idx = {}
for sub in sorted(os.listdir(GLB_BASE)):
    sp = os.path.join(GLB_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.glb'):
                glb_idx[fn[:-4]] = os.path.join(sp, fn)

OUT = '/home/z50057756/tmp/wash_fix_bisect'


def init_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures,
              bpy.data.lights, bpy.data.cameras, bpy.data.node_groups):
        for it in list(c):
            try: c.remove(it, do_unlink=True)
            except Exception: pass


def setup_render_prod_like():
    """Copy production init_render + init_nodes verbatim."""
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'GPU'
    sc.cycles.samples = 128
    sc.cycles.filter_type = 'BOX'
    sc.cycles.filter_width = 1
    sc.cycles.diffuse_bounces = 1
    sc.cycles.glossy_bounces = 1
    sc.cycles.transmission_bounces = 3
    sc.cycles.transparent_max_bounces = 3
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPTIX'
    sc.render.resolution_x = 256
    sc.render.resolution_y = 256
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.use_persistent_data = True  # <-- production has this
    sc.world.use_nodes = True


def init_nodes_prod():
    """Copy production init_nodes verbatim — creates depth output compositor
       nodes but NO Composite node to route RGB back."""
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[0].use_pass_z = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    rl = nodes.new("CompositorNodeRLayers")
    depth_out = nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = OUT
    depth_out.file_slots[0].use_node_format = True
    depth_out.format.file_format = "PNG"
    depth_out.format.color_depth = "16"
    depth_out.format.color_mode = "BW"
    depth_map = nodes.new(type="CompositorNodeMapRange")
    depth_map.inputs[1].default_value = 0
    depth_map.inputs[2].default_value = 10
    depth_map.inputs[3].default_value = 0
    depth_map.inputs[4].default_value = 1
    links.new(rl.outputs["Depth"], depth_map.inputs[0])
    links.new(depth_map.outputs[0], depth_out.inputs[0])


def init_nodes_with_composite():
    """Alternative: add CompositorNodeComposite so RGB gets routed properly."""
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[0].use_pass_z = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    rl = nodes.new("CompositorNodeRLayers")
    # Add composite node (routes RGB back)
    comp = nodes.new("CompositorNodeComposite")
    links.new(rl.outputs["Image"], comp.inputs["Image"])
    # Depth chain (same as before)
    depth_out = nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = OUT
    depth_out.file_slots[0].use_node_format = True
    depth_out.format.file_format = "PNG"
    depth_out.format.color_depth = "16"
    depth_out.format.color_mode = "BW"
    depth_map = nodes.new(type="CompositorNodeMapRange")
    depth_map.inputs[1].default_value = 0
    depth_map.inputs[2].default_value = 10
    depth_map.inputs[3].default_value = 0
    depth_map.inputs[4].default_value = 1
    links.new(rl.outputs["Depth"], depth_map.inputs[0])
    links.new(depth_map.outputs[0], depth_out.inputs[0])


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


def render(out_path):
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def measure(png_path):
    if not os.path.exists(png_path): return None
    arr = np.array(PILImage.open(png_path))
    if arr.ndim != 3: return None
    if arr.shape[2] >= 4:
        fg = arr[..., 3] > 127
        if fg.sum() < 100: return None
        rgb = arr[..., :3][fg].astype(np.float32)
    else:
        rgb = arr[..., :3].reshape(-1, 3).astype(np.float32)
    R, G, B = rgb.mean(axis=0)
    diff = max(abs(R-G), abs(G-B), abs(R-B))
    sat = ((rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1) >= 0.15).mean() * 100
    return {'R': float(R), 'G': float(G), 'B': float(B), 'diff': float(diff),
            'sat': float(sat)}


# 3 modes to test the compositor hypothesis
MODES = [
    ('NO_COMPOSITOR', lambda: None),                # baseline (no compositor) — control (should render color)
    ('PROD_COMPOSITOR', init_nodes_prod),           # prod's compositor (no Composite node) — expected to wash
    ('COMPOSITOR_WITH_COMPOSITE', init_nodes_with_composite),  # with Composite node — expected fix
]

all_results = []
for uid in WASH_UIDS:
    glb = glb_idx.get(uid)
    if not glb:
        print(f"MISSING glb: {uid}", flush=True); continue
    for mode_name, node_setup in MODES:
        print(f"\n>>> {uid[:14]} mode={mode_name}", flush=True)
        init_scene()
        setup_render_prod_like()
        lighting()
        try:
            bpy.ops.import_scene.gltf(filepath=glb, merge_vertices=True, import_shading='NORMALS')
        except Exception as e:
            print(f"  import FAILED: {e}", flush=True); continue
        if not normalize_hierarchy():
            print(f"  normalize FAILED", flush=True); continue
        if node_setup:
            node_setup()
        add_camera(yaw=0.5, pitch=0.3, fov_deg=52.0)
        png = f"{OUT}/{uid[:14]}_{mode_name}.png"
        try:
            render(png)
            stats = measure(png)
            if stats:
                print(f"  RGB=({stats['R']:.0f},{stats['G']:.0f},{stats['B']:.0f})  diff={stats['diff']:.1f}  sat={stats['sat']:.0f}%", flush=True)
            else:
                print(f"  measure FAILED", flush=True)
        except Exception as e:
            print(f"  render FAILED: {e}", flush=True)
            stats = None
        all_results.append({'uid': uid, 'mode': mode_name, 'stats': stats})

with open(f'{OUT}/wash_repro_compositor.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n\n=== SUMMARY ===", flush=True)
for uid in WASH_UIDS:
    print(f"\n{uid[:14]}:")
    for mode_name, _ in MODES:
        r = next((x for x in all_results if x['uid']==uid and x['mode']==mode_name), None)
        if r and r['stats']:
            s = r['stats']
            print(f"  {mode_name:<28} RGB=({s['R']:.0f},{s['G']:.0f},{s['B']:.0f}) diff={s['diff']:.1f} sat={s['sat']:.0f}%")
        else:
            print(f"  {mode_name:<28} FAILED")
