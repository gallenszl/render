"""Parse GLB files directly to check material/texture state.
No Blender, no rendering — just glTF structure inspection.
Tests H3: 'glTF importer fails to bind baseColorTexture for some meshes'.
"""
import json, os, random
from pygltflib import GLTF2
from concurrent.futures import ProcessPoolExecutor

with open('/home/z50057756/tmp/phase2_smoke/color_audit_v2.json') as f:
    data = json.load(f)

records = data['records']
classes = {}
for r in records:
    classes.setdefault(r['cls'], []).append(r['uid'])

# Sample 12 from each interesting class
random.seed(42)
SAMPLES = {
    'WASH_HEAVY':   random.sample(classes.get('WASH_HEAVY', []), min(15, len(classes.get('WASH_HEAVY', [])))),
    'WASH_INVERSE': random.sample(classes.get('WASH_INVERSE', []), min(15, len(classes.get('WASH_INVERSE', [])))),
    'NORMAL_COLOR': random.sample(classes.get('NORMAL_COLOR', []), 12),
    'MESH_GRAY':    random.sample(classes.get('MESH_GRAY', []), 12),
}
print("Class samples:")
for k, v in SAMPLES.items():
    print(f"  {k}: {len(v)}")

# Build UID → GLB path map
import glob
GLB_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/.objaverse/hf-objaverse-v1/glbs'
glb_index = {}
for sub in sorted(os.listdir(GLB_BASE)):
    sp = os.path.join(GLB_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.glb'):
                glb_index[fn[:-4]] = os.path.join(sp, fn)


def introspect_one(args):
    """Return per-mesh material/texture summary."""
    cls, uid = args
    path = glb_index.get(uid)
    if not path:
        return {'cls': cls, 'uid': uid, 'err': 'glb missing'}
    try:
        g = GLTF2().load(path)
    except Exception as e:
        return {'cls': cls, 'uid': uid, 'err': f'load: {e}'}

    n_mat = len(g.materials) if g.materials else 0
    n_tex = len(g.textures) if g.textures else 0
    n_img = len(g.images) if g.images else 0

    # Per-material analysis
    mats_info = []
    n_mat_with_base_tex = 0
    n_mat_with_base_color_factor = 0
    n_mat_emission = 0
    n_mat_unlit = 0
    base_color_factors = []
    for mi, m in enumerate(g.materials or []):
        has_base_tex = False
        has_base_color = False
        base_color = None
        has_emission = False
        is_unlit = False
        if m.pbrMetallicRoughness:
            pbr = m.pbrMetallicRoughness
            if pbr.baseColorTexture is not None:
                has_base_tex = True
                n_mat_with_base_tex += 1
            if pbr.baseColorFactor is not None and pbr.baseColorFactor != [1,1,1,1]:
                has_base_color = True
                base_color = list(pbr.baseColorFactor)
                n_mat_with_base_color_factor += 1
                base_color_factors.append(base_color)
        if m.emissiveTexture is not None or (m.emissiveFactor and any(v > 0 for v in m.emissiveFactor)):
            has_emission = True
            n_mat_emission += 1
        if m.extensions and 'KHR_materials_unlit' in m.extensions:
            is_unlit = True
            n_mat_unlit += 1
        mats_info.append({'name': m.name, 'has_base_tex': has_base_tex,
                          'has_base_color': has_base_color, 'base_color': base_color,
                          'emission': has_emission, 'unlit': is_unlit})

    # Image stats
    images_info = []
    for img in g.images or []:
        info = {'name': img.name, 'has_uri': bool(img.uri), 'has_bufferView': img.bufferView is not None,
                'mimeType': img.mimeType}
        images_info.append(info)

    return {
        'cls': cls, 'uid': uid,
        'n_materials': n_mat, 'n_textures': n_tex, 'n_images': n_img,
        'n_mat_with_base_tex': n_mat_with_base_tex,
        'n_mat_with_base_color_factor': n_mat_with_base_color_factor,
        'n_mat_emission': n_mat_emission,
        'n_mat_unlit': n_mat_unlit,
        'has_any_texture': n_tex > 0 and n_img > 0,
        'has_any_base_tex': n_mat_with_base_tex > 0,
        'mats': mats_info[:3],  # show first 3 mat names
        'images_partial': images_info[:3],
    }


jobs = [(cls, uid) for cls, uids in SAMPLES.items() for uid in uids]
print(f"\nIntrospecting {len(jobs)} GLBs ...")
with ProcessPoolExecutor(max_workers=24) as pool:
    results = list(pool.map(introspect_one, jobs))

# Save full results
with open('/home/z50057756/tmp/phase2_smoke/glb_introspect.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary per class
print()
print("=" * 100)
print("PER-CLASS texture+material summary:")
print("=" * 100)
print(f"{'class':<16} {'n':>3} {'has_tex':>9} {'has_basetex':>13} {'no_tex':>8} {'unlit':>7} {'emission':>10}")
for cls in ['WASH_HEAVY', 'WASH_INVERSE', 'NORMAL_COLOR', 'MESH_GRAY']:
    items = [r for r in results if r['cls'] == cls and 'err' not in r]
    if not items: continue
    n = len(items)
    n_has_tex = sum(r['has_any_texture'] for r in items)
    n_has_basetex = sum(r['has_any_base_tex'] for r in items)
    n_no_tex = sum(not r['has_any_texture'] for r in items)
    n_unlit = sum(r['n_mat_unlit'] > 0 for r in items)
    n_emission = sum(r['n_mat_emission'] > 0 for r in items)
    print(f"  {cls:<14} {n:>3} {n_has_tex:>9} {n_has_basetex:>13} {n_no_tex:>8} {n_unlit:>7} {n_emission:>10}")

# Detail print for WASH cases (the interesting ones)
print()
print("=" * 100)
print("WASH_HEAVY samples (should have texture per FE — does GLB confirm?):")
print("=" * 100)
for r in [r for r in results if r['cls'] == 'WASH_HEAVY' and 'err' not in r][:15]:
    flag = "✓ has texture" if r['has_any_base_tex'] else ("◯ NO texture (mesh truly gray)" if not r['has_any_texture'] else "△ images present but no baseColorTexture binding")
    print(f"  {r['uid'][:14]}  mats={r['n_materials']} tex={r['n_textures']} img={r['n_images']} basetex_mats={r['n_mat_with_base_tex']}/{r['n_materials']}  {flag}")

print()
print("=" * 100)
print("WASH_INVERSE samples (FE saw no color, VL saw color — GLB texture?):")
print("=" * 100)
for r in [r for r in results if r['cls'] == 'WASH_INVERSE' and 'err' not in r][:15]:
    flag = "✓ has texture" if r['has_any_base_tex'] else ("◯ NO texture (so VL's color is fake?)" if not r['has_any_texture'] else "△ images but no baseColorTexture")
    print(f"  {r['uid'][:14]}  mats={r['n_materials']} tex={r['n_textures']} img={r['n_images']} basetex_mats={r['n_mat_with_base_tex']}/{r['n_materials']}  {flag}")

print()
print("=" * 100)
print("NORMAL_COLOR samples (control — should all have texture):")
print("=" * 100)
for r in [r for r in results if r['cls'] == 'NORMAL_COLOR' and 'err' not in r][:12]:
    flag = "✓" if r['has_any_base_tex'] else "◯"
    print(f"  {r['uid'][:14]}  mats={r['n_materials']} tex={r['n_textures']} img={r['n_images']} basetex_mats={r['n_mat_with_base_tex']}/{r['n_materials']}  {flag}")

print()
print("=" * 100)
print("MESH_GRAY samples (control — should have NO texture or single-color baseColorFactor):")
print("=" * 100)
for r in [r for r in results if r['cls'] == 'MESH_GRAY' and 'err' not in r][:12]:
    flag = "✓" if r['has_any_base_tex'] else "◯"
    print(f"  {r['uid'][:14]}  mats={r['n_materials']} tex={r['n_textures']} img={r['n_images']} basetex_mats={r['n_mat_with_base_tex']}/{r['n_materials']}  {flag}")
