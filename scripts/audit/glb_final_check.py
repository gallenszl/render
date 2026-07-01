"""Read GLB of all remaining wash UIDs to check if mesh has texture."""
import os, json
from pygltflib import GLTF2
from concurrent.futures import ProcessPoolExecutor

d = json.load(open('/home/z50057756/tmp/phase2_smoke/color_audit_v2_fix2.json'))
records = d['records']

# All remaining wash cases
WASH_CLASSES = ['WASH_HEAVY', 'WASH_MILD', 'WASH_INVERSE', 'WASH_INVERSE_MILD', 'OTHER']
wash_recs = [r for r in records if r['cls'] in WASH_CLASSES]
print(f"Total remaining wash UIDs: {len(wash_recs)}")

GLB_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/.objaverse/hf-objaverse-v1/glbs'
glb_idx = {}
for sub in sorted(os.listdir(GLB_BASE)):
    sp = os.path.join(GLB_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.glb'):
                glb_idx[fn[:-4]] = os.path.join(sp, fn)


def introspect(args):
    r, path = args
    if not path: return {**r, 'err': 'MISS'}
    try:
        g = GLTF2().load(path)
    except Exception as e:
        return {**r, 'err': f'load: {e}'}
    n_mat = len(g.materials or [])
    n_tex = len(g.textures or [])
    n_img = len(g.images or [])
    n_basetex = 0
    n_unlit = 0
    n_specgloss = 0
    base_colors = []
    for m in g.materials or []:
        if m.pbrMetallicRoughness:
            if m.pbrMetallicRoughness.baseColorTexture is not None:
                n_basetex += 1
            if m.pbrMetallicRoughness.baseColorFactor:
                base_colors.append(m.pbrMetallicRoughness.baseColorFactor)
        if m.extensions:
            if 'KHR_materials_unlit' in m.extensions: n_unlit += 1
            if 'KHR_materials_pbrSpecularGlossiness' in m.extensions: n_specgloss += 1
    # Determine "has_texture" verdict
    has_baseColorTex = n_basetex > 0
    return {**r, 'n_mat': n_mat, 'n_tex': n_tex, 'n_img': n_img,
            'n_basetex': n_basetex, 'n_unlit': n_unlit, 'n_specgloss': n_specgloss,
            'has_baseColorTex': has_baseColorTex,
            'base_colors': base_colors[:2]}


jobs = [(r, glb_idx.get(r['uid'])) for r in wash_recs]
with ProcessPoolExecutor(max_workers=24) as pool:
    results = list(pool.map(introspect, jobs))

# Print per-class breakdown
print()
for cls in WASH_CLASSES:
    items = [r for r in results if r['cls'] == cls and 'err' not in r]
    if not items: continue
    n = len(items)
    with_tex = sum(r['has_baseColorTex'] for r in items)
    no_tex = n - with_tex
    unlit = sum(r['n_unlit'] > 0 for r in items)
    specgloss = sum(r['n_specgloss'] > 0 for r in items)
    print(f"\n{cls} ({n}):")
    print(f"  has baseColorTexture: {with_tex}/{n}  ({100*with_tex/n:.0f}%)")
    print(f"  no baseColorTexture:  {no_tex}/{n}  ({100*no_tex/n:.0f}%)")
    print(f"  has KHR_unlit:        {unlit}/{n}")
    print(f"  has SpecGloss:        {specgloss}/{n}")
    # Show individual
    print(f"  Details:")
    for r in items:
        flag = "✓TEX" if r['has_baseColorTex'] else "✗NO_TEX"
        ext_note = ""
        if r['n_unlit'] > 0: ext_note += " unlit"
        if r['n_specgloss'] > 0: ext_note += " specgloss"
        bc_note = ""
        if not r['has_baseColorTex'] and r['base_colors']:
            bc = r['base_colors'][0][:3]
            bc_note = f" bcf=({bc[0]:.2f},{bc[1]:.2f},{bc[2]:.2f})"
        print(f"    {r['uid'][:14]}  {flag:<8} mats={r['n_mat']} tex={r['n_tex']} FE_S={r['FE_S']:.0f}% VL_S={r['VL_S']:.0f}%{ext_note}{bc_note}")

# Overall verdict
print()
print("=" * 80)
print("OVERALL VERDICT for remaining wash cases:")
print("=" * 80)
all_ok = [r for r in results if 'err' not in r]
with_tex_total = sum(r['has_baseColorTex'] for r in all_ok)
no_tex_total = len(all_ok) - with_tex_total
print(f"  Total remaining wash: {len(all_ok)}")
print(f"  has baseColorTexture (render side quirk): {with_tex_total}")
print(f"  no baseColorTexture (mesh IS colorless):   {no_tex_total}")
