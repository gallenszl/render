"""Final verification: targeted paired comparison.
Only opens the .tar files for the 80 UIDs present in vanilla+Layers.
"""
import os, glob, tarfile, io, json
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

VL_DIR   = '/home/z50057756/tmp/phase2_smoke/render_out_vanilla_layers'
OLD_DIR  = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/data/objaverse_renders_44798'
FE_BASE  = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/FluffyElephant_tar'


def scan_one(args):
    src, tar_path = args
    if not os.path.exists(tar_path):
        return None
    uid = os.path.basename(tar_path)[:-4]
    fgs, clips = [], []
    k_zoom, a_ext, fov_deg = None, None, None
    try:
        with tarfile.open(tar_path) as tf:
            prefix = None
            members = tf.getmembers()
            for m in members:
                if 'phases.json' in m.name:
                    ph = json.loads(tf.extractfile(m).read())
                    k_zoom = ph.get('k_zoom'); a_ext = ph.get('a_ext')
                    prefix = m.name.rsplit('/',1)[0] + '/' if '/' in m.name else ''
                    break
            if prefix is None:
                # Fallback: detect from first .png path
                for m in members:
                    if m.name.endswith('.png'):
                        prefix = m.name.rsplit('/',1)[0] + '/' if '/' in m.name else ''
                        # FE uses './000.png' → prefix='./'
                        if not prefix and m.name.startswith('./'):
                            prefix = './'
                        break
            if prefix is None:
                return None
            try:
                tj = json.loads(tf.extractfile(f"{prefix}transforms.json").read())
                if tj.get('frames'):
                    fov_deg = float(np.degrees(tj['frames'][0].get('fov', 0)))
            except Exception:
                pass
            # FE has 25 frames, our renders have 40
            for i in range(40):
                try:
                    f = tf.extractfile(f"{prefix}{i:03d}.png")
                    arr = np.array(Image.open(io.BytesIO(f.read())))
                    if arr.ndim != 3 or arr.shape[2] < 4: continue
                    a = arr[..., 3] > 127
                    if a.sum() < 10: continue
                    fgs.append(float(a.mean()))
                    clips.append(int(a[0,:].any() or a[-1,:].any() or a[:,0].any() or a[:,-1].any()))
                except KeyError:
                    break
                except Exception:
                    continue
    except Exception:
        return None
    if not fgs:
        return None
    return {
        'uid': uid, 'src': src,
        'fg_mean': float(np.mean(fgs)),
        'fg_p50':  float(np.median(fgs)),
        'n_views': len(fgs),
        'clip_rate': float(np.mean(clips)) * 100,
        'k_zoom': k_zoom, 'a_ext': a_ext, 'fov_deg': fov_deg,
    }


# Step 1: scan VL
vl_tars = sorted(glob.glob(f'{VL_DIR}/*.tar'))
vl_uids = [os.path.basename(t)[:-4] for t in vl_tars]
print(f"vanilla+Layers tars: {len(vl_tars)}")

# Step 2: locate the corresponding old r44 + FE tar paths
old_paths = {u: f'{OLD_DIR}/{u}.tar' for u in vl_uids if os.path.exists(f'{OLD_DIR}/{u}.tar')}
print(f"old r44 hits:        {len(old_paths)}")

# FE is sharded into subdirs 000-XXX/<uid>.tar — find each
fe_paths = {}
# Build subdir map once
fe_sub_index = {}
for sub in sorted(os.listdir(FE_BASE)):
    sub_path = os.path.join(FE_BASE, sub)
    if not os.path.isdir(sub_path): continue
    for fn in os.listdir(sub_path):
        if fn.endswith('.tar'):
            fe_sub_index[fn[:-4]] = os.path.join(sub_path, fn)
for u in vl_uids:
    if u in fe_sub_index:
        fe_paths[u] = fe_sub_index[u]
print(f"FE hits:             {len(fe_paths)}")

# Step 3: scan all in parallel
jobs = []
for t in vl_tars:        jobs.append(('VL',  t))
for p in old_paths.values(): jobs.append(('OLD', p))
for p in fe_paths.values():  jobs.append(('FE',  p))
print(f"\nScanning {len(jobs)} tars (parallel) …")

with ProcessPoolExecutor(max_workers=24) as pool:
    results = [r for r in pool.map(scan_one, jobs) if r is not None]

VL  = {r['uid']: r for r in results if r['src']=='VL'}
OLD = {r['uid']: r for r in results if r['src']=='OLD'}
FE  = {r['uid']: r for r in results if r['src']=='FE'}
print(f"Scanned: VL={len(VL)}  OLD={len(OLD)}  FE={len(FE)}")

# Paired set
paired = sorted(set(VL.keys()) & set(OLD.keys()) & set(FE.keys()))
only_vl_not_old = sorted(set(VL.keys()) - set(OLD.keys()))
only_vl_not_fe  = sorted(set(VL.keys()) - set(FE.keys()))
print(f"\nVL: {len(VL)}   VL∩OLD: {len(VL.keys() & OLD.keys())}   VL∩FE: {len(VL.keys() & FE.keys())}   VL∩OLD∩FE: {len(paired)}")
print(f"In VL but NOT old: {len(only_vl_not_old)}")
print(f"In VL but NOT FE:  {len(only_vl_not_fe)}")
for u in only_vl_not_old: print(f"  not in old: {u}")
for u in only_vl_not_fe:  print(f"  not in FE:  {u}")

# Paired stats
vl_a  = np.array([VL[u]['fg_mean']  for u in paired])
vl_m  = np.array([VL[u]['fg_p50']   for u in paired])
old_a = np.array([OLD[u]['fg_mean'] for u in paired])
old_m = np.array([OLD[u]['fg_p50']  for u in paired])
fe_a  = np.array([FE[u]['fg_mean']  for u in paired])
fe_m  = np.array([FE[u]['fg_p50']   for u in paired])

print()
print("=" * 95)
print(f"PAIRED comparison on {len(paired)} UIDs (same obj across vanilla+Layers / old r44 / FE):")
print("=" * 95)
def col(arr, q=None):
    return f"{(np.percentile(arr, q) if q is not None else arr.mean()):.4f}"
print(f"  {'':<24} {'vanilla+Layers':>15}  {'old r44':>10}  {'FE':>10}")
print(f"  {'mean(fg_mean)':<24} {col(vl_a):>15}  {col(old_a):>10}  {col(fe_a):>10}")
print(f"  {'p50(fg_mean)':<24} {col(vl_a,50):>15}  {col(old_a,50):>10}  {col(fe_a,50):>10}")
print(f"  {'p10(fg_mean)':<24} {col(vl_a,10):>15}  {col(old_a,10):>10}  {col(fe_a,10):>10}")
print(f"  {'p90(fg_mean)':<24} {col(vl_a,90):>15}  {col(old_a,90):>10}  {col(fe_a,90):>10}")

print()
print(f"  {'':<24} {'VL/FE ratio':>15}  {'old/FE ratio':>13}")
print(f"  {'mean(fg_mean) ratio':<24} {vl_a.mean()/fe_a.mean():>15.4f}  {old_a.mean()/fe_a.mean():>13.4f}")
print(f"  {'p50(fg_mean) ratio':<24} {np.percentile(vl_a,50)/np.percentile(fe_a,50):>15.4f}  {np.percentile(old_a,50)/np.percentile(fe_a,50):>13.4f}")

# Per-obj ratios
ratios_vl  = vl_a / fe_a
ratios_old = old_a / fe_a
print()
print("Per-obj fg ratio vs FE:")
print(f"  VL  vs FE:   mean={ratios_vl.mean():.3f}  median={np.median(ratios_vl):.3f}  p10={np.percentile(ratios_vl,10):.3f}  p90={np.percentile(ratios_vl,90):.3f}")
print(f"  OLD vs FE:   mean={ratios_old.mean():.3f}  median={np.median(ratios_old):.3f}  p10={np.percentile(ratios_old,10):.3f}  p90={np.percentile(ratios_old,90):.3f}")

# Clip rates from VL
print()
print("=" * 95)
print("CLIP RATE (vanilla+Layers, all 80 obj):")
print("=" * 95)
all_vl = list(VL.values())
trig   = [r for r in all_vl if r['k_zoom'] is not None and r['k_zoom'] < 1.0]
normal = [r for r in all_vl if r['k_zoom'] is None or  r['k_zoom'] >= 1.0]
def wclip(rs):
    num = sum(r['clip_rate']*r['n_views'] for r in rs)
    den = sum(r['n_views'] for r in rs)
    return num/den if den else 0
print(f"  Total 80 obj                              view-weighted clip = {wclip(all_vl):>5.2f}%")
print(f"  Normal obj  (k=1.0)        {len(normal):>2d}             view-weighted clip = {wclip(normal):>5.2f}%")
print(f"  Trig   obj  (Strategy E)   {len(trig):>2d}             view-weighted clip = {wclip(trig):>5.2f}%")

print()
print("Trig obj detail (sorted by clip%):")
for r in sorted(trig, key=lambda x: -x['clip_rate']):
    print(f"  {r['uid'][:14]}  k={r['k_zoom']:.3f} a_ext={r['a_ext']:.2f} fg={r['fg_mean']:.3f} clip={r['clip_rate']:5.1f}% fov={r['fov_deg']:.1f}°")

# FoV verification
fovs = np.array([r['fov_deg'] for r in all_vl if r['fov_deg'] is not None])
print()
print(f"FoV (vanilla+Layers): n={len(fovs)}  min={fovs.min():.1f}°  max={fovs.max():.1f}°  mean={fovs.mean():.1f}°")
print(f"  Expected U(40°, 75°): {'PASS' if fovs.min() >= 40 and fovs.max() <= 75 else 'FAIL'}")
