"""Per-obj 3x3 montage for MAD N=30 + plan C smoke.
For each rendered obj: sample 9 random frames out of 40, composite alpha on white,
arrange in 3x3 grid, save to OUT_DIR/<uid>.png with overlay (uid, fg, k_zoom, a_ext).
"""
import io, json, os, random, tarfile, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

R30C_DIR = '/home/z50057756/tmp/phase2_smoke/render_out_vanilla_layers'
OUT_DIR = '/home/z50057756/tmp/phase2_smoke_viz_vanilla_layers'
QUAR_FILE = f'{R30C_DIR}/quarantine.txt'
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 200
GRID = 3
LABEL_H = 28
W = CELL * GRID
H = LABEL_H + CELL * GRID

try:
    font   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except IOError:
    font = font_s = ImageFont.load_default()


def build_montage(tar_path):
    uid = os.path.basename(tar_path)[:-4]
    out_path = f"{OUT_DIR}/{uid}.png"
    try:
        with tarfile.open(tar_path) as tf:
            # Find prefix + phases.json metadata
            prefix = None
            k_zoom = None; a_ext = None; aspect_long = None
            for m in tf.getmembers():
                if 'phases.json' in m.name:
                    ph = json.loads(tf.extractfile(m).read())
                    k_zoom = ph.get('k_zoom'); a_ext = ph.get('a_ext')
                    aspect_long = ph.get('aspect_long')
                    prefix = m.name.rsplit('/', 1)[0] + '/' if '/' in m.name else ''
                    break

            # Get fg stats
            fgs = []
            frame_arrs = {}
            for i in range(40):
                try:
                    f = tf.extractfile(f"{prefix}{i:03d}.png")
                    arr = np.array(Image.open(io.BytesIO(f.read())))
                    if arr.ndim != 3 or arr.shape[2] < 4: continue
                    a = arr[..., 3] > 127
                    if a.sum() < 10: continue
                    fgs.append(a.mean())
                    frame_arrs[i] = arr
                except Exception: continue

            if not fgs: return None
            fg_mean = float(np.mean(fgs))

            # Sample 9 random frames (seeded by uid for stability)
            random.seed(int(uid[:8], 16))
            avail = list(frame_arrs.keys())
            sample = sorted(random.sample(avail, min(9, len(avail))))

            canvas = Image.new('RGB', (W, H), (240, 240, 240))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 0, W, LABEL_H], fill=(40, 40, 80))
            kz_s = f"{k_zoom:.3f}" if k_zoom is not None else "?"
            ax_s = f"{a_ext:.2f}" if a_ext is not None else "?"
            label = f"{uid[:18]}  fg={fg_mean:.3f}  k={kz_s}  a_ext={ax_s}"
            draw.text((5, 6), label, fill=(255, 255, 200), font=font)

            for idx, f_idx in enumerate(sample):
                col = idx % GRID
                row = idx // GRID
                arr = frame_arrs[f_idx]
                a = arr[..., 3:4].astype(np.float32) / 255
                rgb = (arr[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
                im = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
                canvas.paste(im, (col * CELL, LABEL_H + row * CELL))
                draw.text((col * CELL + 3, LABEL_H + row * CELL + 3),
                          f"f{f_idx:02d}", fill=(255, 0, 0), font=font_s)

            canvas.save(out_path)
            return (uid, fg_mean, k_zoom, a_ext)
    except Exception as e:
        return None


def make_quarantine_card(quar_entry):
    uid, reason = quar_entry
    out_path = f"{OUT_DIR}/QUAR_{uid}.png"
    canvas = Image.new('RGB', (W, H), (50, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W, LABEL_H], fill=(120, 30, 30))
    draw.text((5, 6), f"QUARANTINED  {uid[:18]}", fill=(255, 230, 200), font=font)
    msg = f"reason: {reason}\n\nThis mesh was rejected\nby Layer 1a or 1c.\nNo tar produced."
    draw.multiline_text((20, LABEL_H + 30), msg, fill=(255, 240, 240), font=font, spacing=8)
    canvas.save(out_path)
    return uid


tars = sorted(glob.glob(f'{R30C_DIR}/*.tar'))
print(f"Building {len(tars)} montages from {R30C_DIR} ...")

with ProcessPoolExecutor(max_workers=24) as pool:
    results = [r for r in pool.map(build_montage, tars) if r is not None]

# Quarantined obj — render placeholder card so user has 100 total
if os.path.exists(QUAR_FILE):
    quar = [l.strip().split('\t') for l in open(QUAR_FILE) if l.strip()]
    print(f"Quarantined: {len(quar)}")
    for q in quar:
        make_quarantine_card(q)
else:
    quar = []

print(f"\nMontages saved to: {OUT_DIR}/")
print(f"  rendered:    {len(results)}")
print(f"  quarantined: {len(quar)}")
print(f"  total cards: {len(results) + len(quar)}")

# Stats
fgs = [r[1] for r in results]
print(f"\nfg distribution (vanilla+Layers):")
print(f"  n      = {len(fgs)}")
print(f"  mean   = {np.mean(fgs):.4f}")
print(f"  median = {np.median(fgs):.4f}")
print(f"  p10    = {np.percentile(fgs, 10):.4f}")
print(f"  p90    = {np.percentile(fgs, 90):.4f}")

# Show top + bottom 5 by fg
results_sorted = sorted(results, key=lambda x: x[1])
print(f"\nLowest 5 fg:")
for u, fg, k, a in results_sorted[:5]:
    print(f"  {u[:14]}  fg={fg:.4f}  k={k}  a_ext={a}")
print(f"\nHighest 5 fg:")
for u, fg, k, a in results_sorted[-5:]:
    print(f"  {u[:14]}  fg={fg:.4f}  k={k}  a_ext={a}")
