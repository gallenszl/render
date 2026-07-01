"""For each of 80 vanilla+Layers tars + matching FE tar, compute fg RGB stats.
Classify per §8.8 method: mesh-gray vs render-wash vs normal-color.
"""
import os, glob, tarfile, io, json
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

VL_DIR  = '/home/z50057756/tmp/phase2_smoke/render_out_vanilla_layers'
FE_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/FluffyElephant_tar'


def color_stats_one(args):
    """Compute fg-pixel RGB stats from one tar."""
    src, tar_path = args
    if not os.path.exists(tar_path):
        return None
    uid = os.path.basename(tar_path)[:-4]
    Rs, Gs, Bs, eq_counts, fg_counts = [], [], [], 0, 0
    try:
        with tarfile.open(tar_path) as tf:
            prefix = None
            members = tf.getmembers()
            for m in members:
                if m.name.endswith('.png') and 'depth' not in m.name:
                    if '/' in m.name:
                        prefix = m.name.rsplit('/',1)[0] + '/'
                    else:
                        prefix = './' if m.name.startswith('./') else ''
                    if m.name.startswith('./') and prefix != './':
                        prefix = './'
                    break
            if prefix is None: return None
            for i in range(40):
                try:
                    f = tf.extractfile(f"{prefix}{i:03d}.png")
                    arr = np.array(Image.open(io.BytesIO(f.read())))
                    if arr.ndim != 3 or arr.shape[2] < 4: continue
                    fg_mask = arr[..., 3] > 127
                    if fg_mask.sum() < 100: continue
                    rgb = arr[..., :3][fg_mask].astype(np.int32)
                    Rs.append(rgb[:,0].mean())
                    Gs.append(rgb[:,1].mean())
                    Bs.append(rgb[:,2].mean())
                    diff_per_px = np.maximum.reduce([
                        np.abs(rgb[:,0]-rgb[:,1]),
                        np.abs(rgb[:,1]-rgb[:,2]),
                        np.abs(rgb[:,0]-rgb[:,2])
                    ])
                    eq_counts += int((diff_per_px < 3).sum())
                    fg_counts += int(fg_mask.sum())
                except KeyError: break
                except Exception: continue
    except Exception:
        return None
    if not Rs:
        return None
    R, G, B = np.mean(Rs), np.mean(Gs), np.mean(Bs)
    diff = max(abs(R-G), abs(G-B), abs(R-B))
    eq_pct = 100.0 * eq_counts / max(fg_counts, 1)
    return {
        'uid': uid, 'src': src,
        'R': float(R), 'G': float(G), 'B': float(B),
        'diff': float(diff), 'eq_pct': float(eq_pct),
    }


vl_tars = sorted(glob.glob(f'{VL_DIR}/*.tar'))
vl_uids = [os.path.basename(t)[:-4] for t in vl_tars]

# Index FE shards
fe_index = {}
for sub in sorted(os.listdir(FE_BASE)):
    sp = os.path.join(FE_BASE, sub)
    if not os.path.isdir(sp): continue
    for fn in os.listdir(sp):
        if fn.endswith('.tar'):
            fe_index[fn[:-4]] = os.path.join(sp, fn)
fe_paths = {u: fe_index[u] for u in vl_uids if u in fe_index}

jobs = []
for t in vl_tars: jobs.append(('VL', t))
for u, p in fe_paths.items(): jobs.append(('FE', p))
print(f"Scanning {len(jobs)} tars …")
with ProcessPoolExecutor(max_workers=24) as pool:
    results = [r for r in pool.map(color_stats_one, jobs) if r is not None]

VL = {r['uid']: r for r in results if r['src']=='VL'}
FE = {r['uid']: r for r in results if r['src']=='FE'}
print(f"VL: {len(VL)}  FE: {len(FE)}  paired: {len(set(VL) & set(FE))}")

paired = sorted(set(VL) & set(FE))

# Classify per §8.8 thresholds:
# diff = max channel diff of mean RGB; eq_pct = % of fg pixels where R≈G≈B (within 3)
# diff < 8: visually gray; eq_pct > 80%: heavy gray
THRESH_GRAY_DIFF = 8.0
THRESH_COLOR_DIFF = 15.0

categories = {
    'MESH_GRAY (FE+VL both gray)': [],
    'WASH_HEAVY (FE color, VL gray)': [],
    'WASH_MILD  (FE color, VL mid)': [],
    'NORMAL_COLOR (FE color, VL color)': [],
    'AMBIGUOUS': [],
}
for u in paired:
    vl, fe = VL[u], FE[u]
    if fe['diff'] < THRESH_GRAY_DIFF and vl['diff'] < THRESH_GRAY_DIFF:
        categories['MESH_GRAY (FE+VL both gray)'].append((u, vl, fe))
    elif fe['diff'] > THRESH_COLOR_DIFF and vl['diff'] < THRESH_GRAY_DIFF:
        categories['WASH_HEAVY (FE color, VL gray)'].append((u, vl, fe))
    elif fe['diff'] > THRESH_COLOR_DIFF and vl['diff'] < THRESH_COLOR_DIFF:
        categories['WASH_MILD  (FE color, VL mid)'].append((u, vl, fe))
    elif fe['diff'] > THRESH_COLOR_DIFF and vl['diff'] > THRESH_COLOR_DIFF:
        categories['NORMAL_COLOR (FE color, VL color)'].append((u, vl, fe))
    else:
        categories['AMBIGUOUS'].append((u, vl, fe))

print()
print("=" * 100)
print(f"COLOR CLASSIFICATION ({len(paired)} paired obj, threshold diff<{THRESH_GRAY_DIFF}=gray, diff>{THRESH_COLOR_DIFF}=color)")
print("=" * 100)
for cat, items in categories.items():
    print(f"\n{cat}: {len(items)} / {len(paired)}")
    if not items: continue
    items.sort(key=lambda x: -x[2]['diff'])
    for u, vl, fe in items[:8]:
        print(f"  {u[:14]}  FE diff={fe['diff']:>5.1f} (RGB={fe['R']:.0f},{fe['G']:.0f},{fe['B']:.0f}) | "
              f"VL diff={vl['diff']:>5.1f} (RGB={vl['R']:.0f},{vl['G']:.0f},{vl['B']:.0f})  "
              f"VL eq%={vl['eq_pct']:.0f}%")
    if len(items) > 8:
        print(f"  ... and {len(items)-8} more")

# Save full table
with open('/home/z50057756/tmp/phase2_smoke/color_audit_table.json', 'w') as f:
    out = []
    for u in paired:
        vl, fe = VL[u], FE[u]
        out.append({'uid': u,
                    'VL_R': vl['R'], 'VL_G': vl['G'], 'VL_B': vl['B'],
                    'VL_diff': vl['diff'], 'VL_eq_pct': vl['eq_pct'],
                    'FE_R': fe['R'], 'FE_G': fe['G'], 'FE_B': fe['B'],
                    'FE_diff': fe['diff'], 'FE_eq_pct': fe['eq_pct']})
    json.dump(out, f, indent=2)
print(f"\nFull table → /home/z50057756/tmp/phase2_smoke/color_audit_table.json")

print()
print("=" * 100)
print("VL-only stats (80 obj):")
print("=" * 100)
diffs = np.array([r['diff'] for r in VL.values()])
eqs   = np.array([r['eq_pct'] for r in VL.values()])
print(f"  diff: mean={diffs.mean():.1f}  median={np.median(diffs):.1f}  p10={np.percentile(diffs,10):.1f}  p90={np.percentile(diffs,90):.1f}")
print(f"  eq%:  mean={eqs.mean():.1f}%  median={np.median(eqs):.1f}%  p10={np.percentile(eqs,10):.1f}%  p90={np.percentile(eqs,90):.1f}%")
print(f"  obj with diff<{THRESH_GRAY_DIFF} (visually gray):    {(diffs < THRESH_GRAY_DIFF).sum()}/{len(diffs)}")
print(f"  obj with diff>{THRESH_COLOR_DIFF} (visually color):    {(diffs > THRESH_COLOR_DIFF).sum()}/{len(diffs)}")
print(f"  obj with eq>80% (heavy gray):     {(eqs > 80).sum()}/{len(eqs)}")
