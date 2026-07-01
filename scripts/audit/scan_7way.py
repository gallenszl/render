"""7-way scan with multiprocessing."""
import json, io, os, tarfile, glob
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def scan_one_tar(t):
    uid = os.path.basename(t)[:-4]
    fg_views, clip, n = [], 0, 0
    k_zoom = None
    try:
        with tarfile.open(t) as tf:
            prefix = None
            for m in tf.getmembers():
                if 'phases.json' in m.name:
                    phases = json.loads(tf.extractfile(m).read())
                    k_zoom = phases.get('k_zoom')
                    prefix = m.name.rsplit('/', 1)[0] + '/' if '/' in m.name else ''
                    break
            for i in range(40):
                try:
                    f = tf.extractfile(f"{prefix}{i:03d}.png")
                    arr = np.array(Image.open(io.BytesIO(f.read())))
                    if arr.ndim != 3 or arr.shape[2] < 4: continue
                    a = arr[..., 3] > 127
                    if a.sum() < 10: continue
                    n += 1
                    fg_views.append(a.mean())
                    if a[0,:].any() or a[-1,:].any() or a[:,0].any() or a[:,-1].any():
                        clip += 1
                except Exception: continue
    except Exception: return None
    if fg_views:
        return {'uid': uid, 'fg': float(np.mean(fg_views)),
                'clip': clip, 'n': n, 'k_zoom': k_zoom}
    return None


def scan_dir(dirpath, pool):
    tars = sorted(glob.glob(f'{dirpath}/*.tar'))
    results = [r for r in pool.map(scan_one_tar, tars) if r is not None]
    quar = []
    qpath = f'{dirpath}/quarantine.txt'
    if os.path.exists(qpath):
        quar = [l.strip().split('\t') for l in open(qpath) if l.strip()]
    return results, quar


if __name__ == '__main__':
    with open('/home/z50057756/tmp/fluffy_audit/fluffy_audit_full.json') as f:
        fe_data = json.load(f)
    fe_map = {r['entry'].split('/')[-1]: r['fg_mean'] for r in fe_data['results'] if 'error' not in r}

    datasets = [
        ('NO robust (vanilla)', '/home/z50057756/tmp/phase2_smoke/render_out_r062'),
        ('Percentile 99.5',     '/home/z50057756/tmp/phase2_smoke/render_out_r062v2'),
        ('MAD N=5',             '/home/z50057756/tmp/phase2_smoke/render_out_mad5'),
        ('MAD N=7',             '/home/z50057756/tmp/phase2_smoke/render_out_mad7'),
        ('MAD N=10',            '/home/z50057756/tmp/phase2_smoke/render_out_mad10'),
        ('MAD N=15',            '/home/z50057756/tmp/phase2_smoke/render_out_mad15'),
        ('MAD N=20',            '/home/z50057756/tmp/phase2_smoke/render_out_mad20'),
        ('MAD N=20 + plan C',   '/home/z50057756/tmp/phase2_smoke/render_out_mad20c'),
        ('MAD N=30 + plan C',   '/home/z50057756/tmp/phase2_smoke/render_out_mad30c'),
        ('MAD N=40 + plan C',   '/home/z50057756/tmp/phase2_smoke/render_out_mad40c'),
        ('MAD N=50 + plan C',   '/home/z50057756/tmp/phase2_smoke/render_out_mad50c'),
        ('vanilla+Layers (CHOSEN)', '/home/z50057756/tmp/phase2_smoke/render_out_vanilla_layers'),
    ]
    problem_uids = ['43da08b666', '615e79f062', '48a915e2c4', '2d89f86c5a',
                    '62b00923dd', '5d8ee3d4f1', '895ce9cbb6', '43e53a57f0']

    # The 7 problem UIDs that vanilla leaked as fg < 0.005 (plus the 8th
    # — 43da08b666... — that vanilla failed to render at all). All 8 are
    # known-broken obj; exclude to compare main distribution only.
    RESCUED_5 = {
        '615e79f062844f22b24c57d0d776825b',
        '48a915e2c4ce44239f0be46b5a3fafc4',
        '2d89f86c5af845abacec24b6203c873a',
        '62b00923dd2d4ee4b9321b4cebe68460',
        '5d8ee3d4f1f247a4a1e6193b037ba66c',
        '895ce9cbb60548348225b64818716c46',
        '43e53a57f0244288a5cd7196ffd05366',
    }

    rows = []
    all_uid_fg = {}
    all_quar = {}
    with ProcessPoolExecutor(max_workers=24) as pool:
        for label, dirpath in datasets:
            results, quar = scan_dir(dirpath, pool)
            if not results: continue
            fg_arr = np.array([r['fg'] for r in results])
            n_leak = (fg_arr < 0.005).sum()
            common = [r for r in results if r['uid'] in fe_map]
            fg_p = np.array([r['fg'] for r in common])
            fe_p = np.array([fe_map[r['uid']] for r in common])

            # Excluding the 5 rescued UIDs (truly main-distribution only)
            common_clean = [r for r in common if r['uid'] not in RESCUED_5]
            fg_pc = np.array([r['fg'] for r in common_clean])
            fe_pc = np.array([fe_map[r['uid']] for r in common_clean])

            trig = [r for r in results if r['k_zoom'] is not None and r['k_zoom'] < 1.0]
            norm = [r for r in results if r['k_zoom'] is None or r['k_zoom'] >= 1.0]
            trig_c = [r for r in trig if r['uid'] not in RESCUED_5]
            norm_c = [r for r in norm if r['uid'] not in RESCUED_5]
            trig_clip = sum(r['clip'] for r in trig); trig_n = sum(r['n'] for r in trig)
            norm_clip = sum(r['clip'] for r in norm); norm_n = sum(r['n'] for r in norm)
            trig_clip_c = sum(r['clip'] for r in trig_c); trig_n_c = sum(r['n'] for r in trig_c)
            norm_clip_c = sum(r['clip'] for r in norm_c); norm_n_c = sum(r['n'] for r in norm_c)

            n_sal, n_qu = 0, 0
            for puid in problem_uids:
                if any(r['uid'].startswith(puid) for r in results): n_sal += 1
                elif any(qx[0].startswith(puid) for qx in quar): n_qu += 1
            all_uid_fg[label] = {r['uid']: r['fg'] for r in results}
            all_quar[label] = quar
            rows.append({
                'label': label, 'tarred': len(results), 'quar': len(quar), 'leak': n_leak,
                'fg_mean_ratio': fg_p.mean()/fe_p.mean(),
                'fg_p50_ratio': np.percentile(fg_p, 50)/np.percentile(fe_p, 50),
                'norm_obj': len(norm), 'norm_clip': norm_clip/max(norm_n,1)*100 if norm_n else 0,
                'trig_obj': len(trig), 'trig_clip': trig_clip/max(trig_n,1)*100 if trig_n else 0,
                # Excluding 5 rescued
                'fg_mean_ratio_clean': fg_pc.mean()/fe_pc.mean() if len(fg_pc) else 0,
                'fg_p50_ratio_clean': np.percentile(fg_pc, 50)/np.percentile(fe_pc, 50) if len(fg_pc) else 0,
                'norm_obj_c': len(norm_c),
                'norm_clip_c': norm_clip_c/max(norm_n_c, 1)*100 if norm_n_c else 0,
                'trig_obj_c': len(trig_c),
                'trig_clip_c': trig_clip_c/max(trig_n_c, 1)*100 if trig_n_c else 0,
                'salvaged': n_sal, 'quarantined_problem': n_qu,
            })

    print("="*135)
    print("FULL (all tarred obj):")
    print("="*135)
    print(f"{'setting':<22} {'tar':>4} {'quar':>5} {'leak':>5} | {'fg_mn/FE':>9} {'p50/FE':>8} | {'norm':>5} {'n_clip':>8} | {'trig':>5} {'t_clip':>8} | {'rescue':>6} {'P-quar':>7}")
    print("-"*135)
    for r in rows:
        print(f"{r['label']:<22} {r['tarred']:>4} {r['quar']:>5} {r['leak']:>5} | "
              f"{r['fg_mean_ratio']:>9.3f} {r['fg_p50_ratio']:>8.3f} | "
              f"{r['norm_obj']:>5} {r['norm_clip']:>7.1f}% | "
              f"{r['trig_obj']:>5} {r['trig_clip']:>7.1f}% | "
              f"{r['salvaged']:>6} {r['quarantined_problem']:>7}")

    print()
    print("="*135)
    print("EXCLUDING 7 broken UIDs (615e79 / 48a915 / 2d89f8 / 62b00 / 5d8ee3 / 895ce9 / 43e53a — vanilla leak set) — pure main-distribution comparison:")
    print("="*135)
    print(f"{'setting':<22} {'tar_c':>5} | {'fg_mn/FE':>9} {'p50/FE':>8} | {'norm_c':>7} {'n_clip_c':>9} | {'trig_c':>7} {'t_clip_c':>9}")
    print("-"*135)
    for r in rows:
        tar_c = r['tarred'] - sum(1 for u in RESCUED_5 if any(u == k for k in all_uid_fg[r['label']]))
        print(f"{r['label']:<22} {tar_c:>5} | "
              f"{r['fg_mean_ratio_clean']:>9.3f} {r['fg_p50_ratio_clean']:>8.3f} | "
              f"{r['norm_obj_c']:>7} {r['norm_clip_c']:>8.1f}% | "
              f"{r['trig_obj_c']:>7} {r['trig_clip_c']:>8.1f}%")

    print()
    print("="*112)
    print("Per-obj fg ratio vs NO-robust baseline (filter fg>0.005, common-good obj only):")
    print("="*112)
    labels = [r['label'] for r in rows]
    baseline_map = all_uid_fg.get('NO robust (vanilla)', {})
    for label in labels[1:]:
        cmp_map = all_uid_fg.get(label, {})
        common = set(baseline_map) & set(cmp_map)
        if not common: continue
        ratios = [cmp_map[u]/baseline_map[u] for u in common if baseline_map[u] > 0.005]
        enlarged = sum(1 for u in common if baseline_map[u] > 0.005 and cmp_map[u]/baseline_map[u] > 1.15)
        unchanged = sum(1 for u in common if baseline_map[u] > 0.005 and abs(cmp_map[u]/baseline_map[u] - 1.0) < 0.02)
        print(f"  {label:<22}: ratio mean={np.mean(ratios):.3f}  median={np.median(ratios):.3f}  "
              f"p90={np.percentile(ratios, 90):.3f}  enlarged>1.15: {enlarged:>3}  "
              f"unchanged(±2%): {unchanged}/{len(ratios)}")

    print()
    print("=== 8 problem obj 在各方法下的去向 ===")
    short = {l: l.replace(' (vanilla)','').replace(' robust','').replace('Percentile ','Pct').replace(' N=','=') for l in labels}
    header = '  '.join(f"{short[l][:9]:<9}" for l in labels)
    print(f"{'uid':<12} {'old r44 fg':>11}  {header}")
    with open('/mnt/data-alpha-sg-01/team-camera/home/z50057756/data/objaverse_renders_44798/fg_audit_full.json') as f:
        old_audit = json.load(f)
    old_fg = {r['uid']: r['fg_mean'] for r in old_audit['results'] if 'error' not in r}
    for puid in problem_uids:
        full_uids = [u for u in old_fg if u.startswith(puid)]
        if not full_uids: continue
        old = old_fg[full_uids[0]]
        vals = []
        for label in labels:
            rmap = all_uid_fg.get(label, {})
            quar_l = all_quar.get(label, [])
            hit = next((v for k, v in rmap.items() if k.startswith(puid)), None)
            if hit is not None:
                vals.append(f"fg={hit:.3f}")
            elif any(qx[0].startswith(puid) for qx in quar_l):
                vals.append("QUAR")
            else:
                vals.append("MISS")
        print(f"  {puid:<10}  {old:>11.6f}  " + "  ".join(f"{v:<9}" for v in vals))
