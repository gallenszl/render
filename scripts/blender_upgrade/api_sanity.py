"""Blender API sanity check for 4.5/5.2 upgrade.
   Verifies the 6 HIGH/MED-RISK touchpoints identified in the upgrade plan.

   Run with: blender --background --python api_sanity.py
   Prints PASS/FAIL for each check + a JSON report.
"""
import bpy, sys, json, os

REPORT = {}


def probe(name, fn):
    try:
        result = fn()
        REPORT[name] = {'status': 'PASS', 'detail': result}
        print(f"  [PASS] {name}: {result}", flush=True)
        return True
    except Exception as e:
        REPORT[name] = {'status': 'FAIL', 'error': f"{type(e).__name__}: {e}"}
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return False


print(f"=== Blender {bpy.app.version_string} sanity ===")
print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"  build_date: {bpy.app.build_date}")

# 1. glTF importer params
def check_gltf():
    op_rna = bpy.ops.import_scene.gltf.get_rna_type()
    props = {p.identifier: p.default if hasattr(p, 'default') else None for p in op_rna.properties}
    needed = ['filepath', 'merge_vertices', 'import_shading']
    missing = [p for p in needed if p not in props]
    if missing:
        raise KeyError(f"missing params: {missing}")
    # check import_shading enum
    if 'import_shading' in props:
        prop_meta = op_rna.properties['import_shading']
        enum_vals = [i.identifier for i in prop_meta.enum_items] if hasattr(prop_meta, 'enum_items') else []
        if 'NORMALS' not in enum_vals:
            raise ValueError(f"import_shading 'NORMALS' not in enum: {enum_vals}")
    return f"merge_vertices+import_shading+NORMALS enum present"

probe('gltf_import_params', check_gltf)

# 2. Compositor node types
def check_compositor():
    for cls_name in ['CompositorNodeRLayers', 'CompositorNodeOutputFile', 'CompositorNodeMapRange']:
        if not hasattr(bpy.types, cls_name):
            raise AttributeError(f"{cls_name} missing")
    return "3 compositor node types present"

probe('compositor_node_types', check_compositor)

# 3. Cycles denoiser OPTIX enum
def check_denoiser():
    scene = bpy.context.scene
    prop = scene.cycles.bl_rna.properties['denoiser']
    enum_vals = [i.identifier for i in prop.enum_items]
    if 'OPTIX' not in enum_vals:
        raise ValueError(f"OPTIX not in denoiser enum: {enum_vals}")
    return f"OPTIX + {[v for v in enum_vals if v != 'OPTIX']} available"

probe('cycles_denoiser_optix', check_denoiser)

# 4. Cycles bounces attributes
def check_bounces():
    sc = bpy.context.scene
    for attr in ['diffuse_bounces', 'glossy_bounces', 'transmission_bounces',
                 'transparent_max_bounces', 'samples', 'filter_type', 'filter_width',
                 'use_denoising', 'device']:
        if not hasattr(sc.cycles, attr):
            raise AttributeError(f"scene.cycles.{attr} missing")
    return "9 cycles attrs present"

probe('cycles_settings', check_bounces)

# 5. Device API
def check_devices():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.refresh_devices()
    devs = list(prefs.devices)
    types = set(d.type for d in devs)
    return f"n_devices={len(devs)} types={types}"

probe('device_api', check_devices)

# 6. use_persistent_data existence
def check_persistent():
    sc = bpy.context.scene
    orig = sc.render.use_persistent_data
    sc.render.use_persistent_data = False
    sc.render.use_persistent_data = True
    return f"attr exists, default was {orig}"

probe('use_persistent_data', check_persistent)

# 7. Bonus: check other essentials
def check_render_essentials():
    sc = bpy.context.scene
    for attr in ['resolution_x', 'resolution_y', 'film_transparent',
                 'image_settings', 'engine']:
        if not hasattr(sc.render, attr):
            raise AttributeError(f"render.{attr} missing")
    if not hasattr(sc.render.image_settings, 'file_format'):
        raise AttributeError("image_settings.file_format missing")
    return "render essentials OK"

probe('render_essentials', check_render_essentials)

# 8. bpy.ops.outliner.orphans_purge signature
def check_orphans_purge():
    op_rna = bpy.ops.outliner.orphans_purge.get_rna_type()
    props = {p.identifier for p in op_rna.properties}
    for f in ['do_local_ids', 'do_linked_ids', 'do_recursive']:
        if f not in props:
            raise KeyError(f"orphans_purge missing param {f}")
    return "3 do_* flags present"

probe('orphans_purge', check_orphans_purge)

# Summary
n_pass = sum(1 for v in REPORT.values() if v['status'] == 'PASS')
n_fail = len(REPORT) - n_pass
print(f"\n=== SUMMARY: {n_pass}/{len(REPORT)} PASS, {n_fail} FAIL ===")

# Write JSON report
out_dir = os.environ.get('OUT_DIR', '/tmp')
version_slug = bpy.app.version_string.replace(' ', '_').replace('/', '_')
out_path = f"{out_dir}/api_sanity_{version_slug}.json"
with open(out_path, 'w') as f:
    json.dump({'version': bpy.app.version_string, 'build_date': str(bpy.app.build_date),
               'python': f"{sys.version_info.major}.{sys.version_info.minor}",
               'summary': f"{n_pass}/{len(REPORT)} PASS",
               'report': REPORT}, f, indent=2)
print(f"Report → {out_path}")
