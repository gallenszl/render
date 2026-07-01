"""Reclassify color audit using HSV saturation + Lab chroma (robust to highlights).

Replaces mean-RGB-diff with:
  - S_pct  = % of fg pixels with HSV saturation >= 0.15
  - chroma_med = median euclidean distance from gray axis in Lab

Confirms or refutes the agent's WASH_INVERSE finding by also generating
visual montages for top candidates.
"""
import os, glob, tarfile, io, json, random
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

VL_DIR  = '/home/z50057756/tmp/phase2_smoke/render_out_2000'
FE_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/FluffyElephant_tar'


def rgb2hsv_sat(rgb_uint8):
    """Vectorized HSV saturation, RGB uint8 (N,3) -> sat (N,)."""
    arr = rgb_uint8.astype(np.float32) / 255.0
    maxc = arr.max(axis=1)
    minc = arr.min(axis=1)
    sat = np.zeros_like(maxc)
    nonzero = maxc > 0
    sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
    return sat


def rgb2lab_chroma(rgb_uint8):
    """Approx Lab chroma c*=sqrt(a^2+b^2), RGB uint8 (N,3) -> chroma (N,)."""
    # Approximate sRGB → XYZ → Lab path; fast linear approximation
    arr = rgb_uint8.astype(np.float32) / 255.0
    # sRGB → linear (gamma 2.2 approx)
    lin = np.where(arr <= 0.04045, arr/12.92, ((arr + 0.055)/1.055)**2.4)
    # XYZ via D65 matrix
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    xyz = lin @ M.T
    # Normalize by D65 white
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    # f(t) for Lab
    delta = 6/29
    fxyz = np.where(xyz > delta**3, xyz**(1/3), xyz/(3*delta**2) + 4/29)
    a_star = 500 * (fxyz[:,0] - fxyz[:,1])
    b_star = 200 * (fxyz[:,1] - fxyz[:,2])
    return np.sqrt(a_star*a_star + b_star*b_star)


def scan_one(args):
    src, tar_path = args
    if not os.path.exists(tar_path): return None
    uid = os.path.basename(tar_path)[:-4]
    sats = []
    chromas = []
    fg_total = 0
    try:
        with tarfile.open(tar_path) as tf:
            prefix = None
            members = tf.getmembers()
            for m in members:
                if m.name.endswith('.png') and 'depth' not in m.name:
                    if '/' in m.name:
                        prefix = m.name.rsplit('/',1)[0] + '/'
                    elif m.name.startswith('./'):
                        prefix = './'
                    else:
                        prefix = ''
                    break
            if prefix is None: return None
            for i in range(40):
                try:
                    f = tf.extractfile(f"{prefix}{i:03d}.png")
                    arr = np.array(Image.open(io.BytesIO(f.read())))
                    if arr.ndim != 3 or arr.shape[2] < 4: continue
                    fg_mask = arr[..., 3] > 127
                    if fg_mask.sum() < 100: continue
                    rgb = arr[..., :3][fg_mask].reshape(-1, 3)
                    # subsample for speed if too many pixels
                    if rgb.shape[0] > 5000:
                        idx = np.random.choice(rgb.shape[0], 5000, replace=False)
                        rgb = rgb[idx]
                    sats.append(rgb2hsv_sat(rgb))
                    chromas.append(rgb2lab_chroma(rgb))
                    fg_total += fg_mask.sum()
                except KeyError: break
                except Exception: continue
    except Exception:
        return None
    if not sats: return None
    all_sat = np.concatenate(sats)
    all_chroma = np.concatenate(chromas)
    S_pct = float((all_sat >= 0.15).mean() * 100)
    chroma_med = float(np.median(all_chroma))
    chroma_p90 = float(np.percentile(all_chroma, 90))
    return {'uid': uid, 'src': src, 'S_pct': S_pct, 'chroma_med': chroma_med,
            'chroma_p90': chroma_p90, 'n_views': len(sats)}


print("Scanning VL + FE with HSV/chroma metrics...")
vl_tars = sorted(glob.glob(f'{VL_DIR}/*.tar'))
vl_uids = [os.path.basename(t)[:-4] for t in vl_tars]

# Index FE
fe_index = {}
for sub in sorted(os.listdir(FE_BASE)):
    sp = os.path.join(FE_BASE, sub)
    if not os.path.isdir(sp): continue
    for fn in os.listdir(sp):
        if fn.endswith('.tar'):
            fe_index[fn[:-4]] = os.path.join(sp, fn)
fe_paths = {u: fe_index[u] for u in vl_uids if u in fe_index}

jobs = [('VL', t) for t in vl_tars] + [('FE', p) for p in fe_paths.values()]
print(f"Total tars to scan: {len(jobs)}")
with ProcessPoolExecutor(max_workers=24) as pool:
    results = [r for r in pool.map(scan_one, jobs) if r is not None]
VL = {r['uid']: r for r in results if r['src']=='VL'}
FE = {r['uid']: r for r in results if r['src']=='FE'}
paired = sorted(set(VL) & set(FE))
print(f"VL={len(VL)}  FE={len(FE)}  paired={len(paired)}")

# New rule
def classify(fe, vl):
    if fe['S_pct'] < 15 and vl['S_pct'] < 15 and fe['chroma_med'] < 6 and vl['chroma_med'] < 6:
        return 'MESH_GRAY'
    if fe['S_pct'] >= 25 and vl['S_pct'] < 15:
        return 'WASH_HEAVY'
    if fe['S_pct'] >= 25 and 15 <= vl['S_pct'] < 25 and vl['chroma_med'] < 0.65 * fe['chroma_med']:
        return 'WASH_MILD'
    if fe['S_pct'] >= 25 and vl['S_pct'] >= 25:
        return 'NORMAL_COLOR'
    if fe['S_pct'] < 15 and vl['S_pct'] >= 25:
        return 'WASH_INVERSE'  # NEW class
    if fe['S_pct'] < 15 and 15 <= vl['S_pct'] < 25:
        return 'WASH_INVERSE_MILD'
    if 15 <= fe['S_pct'] < 25 and vl['S_pct'] < 15:
        return 'WASH_MILD'  # mild source becomes gray
    if 15 <= fe['S_pct'] < 25 and 15 <= vl['S_pct'] < 25:
        # both mid -> compare ratio
        if vl['chroma_med'] < 0.65 * fe['chroma_med']:
            return 'WASH_MILD'
        return 'NORMAL_COLOR'  # both have weak but matching color
    if 15 <= fe['S_pct'] < 25 and vl['S_pct'] >= 25:
        return 'NORMAL_COLOR'  # VL has stronger color than weakly-colored FE
    return 'OTHER'

# Classify everything
from collections import Counter
cls_counts = Counter()
records = []
for u in paired:
    fe, vl = FE[u], VL[u]
    c = classify(fe, vl)
    cls_counts[c] += 1
    records.append({'uid': u, 'cls': c,
                    'FE_S': fe['S_pct'], 'FE_chroma': fe['chroma_med'],
                    'VL_S': vl['S_pct'], 'VL_chroma': vl['chroma_med']})

print("\n=== NEW classification (HSV S_pct + Lab chroma) ===")
total = sum(cls_counts.values())
for c, n in sorted(cls_counts.items(), key=lambda x: -x[1]):
    print(f"  {c:<20}  {n:>4}  ({100*n/total:>5.1f}%)")
print(f"  {'TOTAL':<20}  {total:>4}")

# Compare against old classification
with open('/home/z50057756/tmp/phase2_smoke/color_audit_table_2000.json') as f:
    old_table = json.load(f)
old_map = {}
for r in old_table:
    fe_d = r['FE_diff']; vl_d = r['VL_diff']
    if fe_d < 8 and vl_d < 8:
        old_map[r['uid']] = 'MESH_GRAY'
    elif fe_d > 15 and vl_d < 8:
        old_map[r['uid']] = 'WASH_HEAVY'
    elif fe_d > 15 and vl_d < 15:
        old_map[r['uid']] = 'WASH_MILD'
    elif fe_d > 15 and vl_d > 15:
        old_map[r['uid']] = 'NORMAL_COLOR'
    else:
        old_map[r['uid']] = 'AMBIGUOUS'

print("\n=== Confusion matrix: OLD class -> NEW class ===")
mat = {}
for r in records:
    o = old_map.get(r['uid'], 'MISS')
    n = r['cls']
    mat[(o,n)] = mat.get((o,n), 0) + 1
old_classes = ['MESH_GRAY','NORMAL_COLOR','AMBIGUOUS','WASH_HEAVY','WASH_MILD']
new_classes = sorted(cls_counts.keys())
hdr = 'OLD | NEW'
print(f"  {hdr:<18}", '  '.join(f"{c[:11]:>11}" for c in new_classes))
for o in old_classes:
    row = [mat.get((o,n), 0) for n in new_classes]
    print(f"  {o:<18}", '  '.join(f"{v:>11}" for v in row), f" sum={sum(row)}")

# Save results
with open('/home/z50057756/tmp/phase2_smoke/color_audit_v2.json', 'w') as f:
    json.dump({'records': records, 'class_counts': dict(cls_counts)}, f, indent=2)
print(f"\nSaved -> /home/z50057756/tmp/phase2_smoke/color_audit_v2.json")
