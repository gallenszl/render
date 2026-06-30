"""Full audit of foreground (alpha > 127) ratio across all 44,798 obj × 40 views.

Per-obj record: {uid, fg_per_view[40], fg_mean, fg_min, fg_max, fg_std, fov_deg}
Per-view fg = (alpha > 127).mean() — fraction of canvas pixels covered by object.

Designed to be I/O-bound; CPU compute is just alpha threshold + reduce.
Scales linearly with workers up to JuiceFS read bandwidth.
"""
import os
import sys
import json
import time
import tarfile
import io
import math
import multiprocessing as mp
import numpy as np
from PIL import Image

TAR_DIR = '/home/z50057756/data/objaverse_renders_44798'
OUT_PATH = '/home/z50057756/tmp/fg_audit/fg_audit_full.json'


def scan_one(uid):
    try:
        tar_path = os.path.join(TAR_DIR, f"{uid}.tar")
        if not os.path.exists(tar_path):
            return {'uid': uid, 'error': 'tar_missing'}
        fg_per_view = []
        fov_deg = None
        with tarfile.open(tar_path, 'r:') as t:
            # find prefix (same logic as dataset_objaverse P5a)
            prefix = None
            for m in t.getmembers():
                if m.name.endswith('.png') and 'depth' not in m.name:
                    if '/' in m.name:
                        prefix = m.name.rsplit('/', 1)[0] + '/'
                    else:
                        prefix = ''
                    break
            if prefix is None:
                return {'uid': uid, 'error': 'no_pngs'}
            # read transforms.json for FoV (per-obj since A2)
            try:
                f = t.extractfile(prefix + 'transforms.json')
                if f is not None:
                    tj = json.loads(f.read())
                    frame0 = tj.get('frames', [{}])[0]
                    fov_rad = frame0.get('fov', frame0.get('camera_angle_x', None))
                    if fov_rad:
                        fov_deg = math.degrees(fov_rad)
            except Exception:
                pass
            for i in range(40):
                member_name = f"{prefix}{i:03d}.png"
                try:
                    fileobj = t.extractfile(member_name)
                    if fileobj is None:
                        return {'uid': uid, 'error': f'view_{i}_missing'}
                    img = Image.open(io.BytesIO(fileobj.read()))
                    arr = np.array(img, copy=False)
                    if arr.ndim != 3 or arr.shape[2] < 4:
                        return {'uid': uid, 'error': f'view_{i}_no_alpha'}
                    alpha = arr[..., 3]
                    fg = (alpha > 127).mean()
                    fg_per_view.append(float(fg))
                except Exception as e:
                    return {'uid': uid, 'error': f'view_{i}_{type(e).__name__}'}
        arr = np.asarray(fg_per_view)
        return {
            'uid': uid,
            'fg_per_view': fg_per_view,
            'fg_mean': float(arr.mean()),
            'fg_min': float(arr.min()),
            'fg_max': float(arr.max()),
            'fg_std': float(arr.std()),
            'fov_deg': fov_deg,
        }
    except Exception as e:
        return {'uid': uid, 'error': type(e).__name__}


def main():
    uids = sorted([f[:-4] for f in os.listdir(TAR_DIR) if f.endswith('.tar')])
    print(f"[{time.strftime('%H:%M:%S')}] found {len(uids)} tars in {TAR_DIR}", flush=True)
    if not uids:
        sys.exit(1)

    n_workers = int(os.environ.get('AUDIT_WORKERS', '96'))
    print(f"[{time.strftime('%H:%M:%S')}] starting Pool of {n_workers} workers", flush=True)

    t0 = time.time()
    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_one, uids, chunksize=20)):
            results.append(r)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(uids) - (i + 1)) / rate
                print(f"  {i + 1}/{len(uids)} done ({rate:.1f} obj/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] all {len(results)} processed in {elapsed:.0f}s "
          f"({len(results)/elapsed:.1f} obj/s)", flush=True)

    n_errors = sum(1 for r in results if 'error' in r)
    print(f"[{time.strftime('%H:%M:%S')}] {n_errors} obj had errors", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump({
            'n_uids': len(uids),
            'workers': n_workers,
            'elapsed_sec': elapsed,
            'n_errors': n_errors,
            'results': results,
        }, f)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
