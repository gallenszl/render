"""Sample WASH_INVERSE obj, build VL vs FE side-by-side montages for visual verify."""
import os, json, tarfile, io, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

OUT_DIR = '/home/z50057756/tmp/wash_inverse_check'
os.makedirs(OUT_DIR, exist_ok=True)

with open('/home/z50057756/tmp/phase2_smoke/color_audit_v2.json') as f:
    data = json.load(f)
inverse = [r for r in data['records'] if r['cls'] == 'WASH_INVERSE']
print(f"WASH_INVERSE: {len(inverse)} obj")
# Sort by VL_S desc (most extreme cases first)
inverse.sort(key=lambda x: -x['VL_S'])

# Build FE index
FE_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/FluffyElephant_tar'
fe_index = {}
for sub in sorted(os.listdir(FE_BASE)):
    sp = os.path.join(FE_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.tar'):
                fe_index[fn[:-4]] = os.path.join(sp, fn)


def get_views(tar_path, view_indices, fe=False):
    """Return list of (idx, composed_rgb)."""
    out = []
    with tarfile.open(tar_path) as tf:
        prefix = None
        for m in tf.getmembers():
            if m.name.endswith('.png') and 'depth' not in m.name:
                if '/' in m.name: prefix = m.name.rsplit('/',1)[0] + '/'
                elif m.name.startswith('./'): prefix = './'
                else: prefix = ''
                break
        for i in view_indices:
            try:
                f = tf.extractfile(f"{prefix}{i:03d}.png")
                arr = np.array(Image.open(io.BytesIO(f.read())))
                if arr.ndim < 3 or arr.shape[2] < 4: continue
                a = arr[..., 3:4].astype(np.float32) / 255
                rgb = (arr[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
                out.append((i, rgb))
            except Exception: continue
    return out


CELL = 200
GRID_W = CELL * 5
LABEL_H = 28
H = LABEL_H * 2 + CELL * 2
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except IOError:
    font = font_s = ImageFont.load_default()


def build_montage(record):
    uid = record['uid']
    vl_tar = f'/home/z50057756/tmp/phase2_smoke/render_out_2000/{uid}.tar'
    fe_tar = fe_index.get(uid)
    if not fe_tar: return None
    vl_views = get_views(vl_tar, [0, 8, 16, 24, 32])
    fe_views = get_views(fe_tar, [0, 5, 10, 15, 20], fe=True)
    if not vl_views or not fe_views: return None

    canvas = Image.new('RGB', (GRID_W, H), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    # VL header
    draw.rectangle([0, 0, GRID_W, LABEL_H], fill=(40, 60, 80))
    draw.text((5, 6), f"{uid[:18]}  VL (ours)  S_pct={record['VL_S']:.0f}%  chroma={record['VL_chroma']:.1f}",
              fill=(255, 240, 200), font=font)
    for idx, (i, rgb) in enumerate(vl_views[:5]):
        im = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
        canvas.paste(im, (idx*CELL, LABEL_H))
        draw.text((idx*CELL + 3, LABEL_H + 3), f"v{i}", fill=(255, 0, 0), font=font_s)
    # FE header
    fe_y = LABEL_H + CELL
    draw.rectangle([0, fe_y, GRID_W, fe_y + LABEL_H], fill=(80, 40, 40))
    draw.text((5, fe_y + 6), f"FE (TRELLIS orig)  S_pct={record['FE_S']:.0f}%  chroma={record['FE_chroma']:.1f}",
              fill=(255, 220, 220), font=font)
    for idx, (i, rgb) in enumerate(fe_views[:5]):
        im = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
        canvas.paste(im, (idx*CELL, fe_y + LABEL_H))
        draw.text((idx*CELL + 3, fe_y + LABEL_H + 3), f"v{i}", fill=(255, 0, 0), font=font_s)
    out_path = f"{OUT_DIR}/{uid}.png"
    canvas.save(out_path)
    return uid


# Pick top 15 most extreme WASH_INVERSE + 5 borderline
sample = inverse[:15]  # most extreme (high VL_S)
sample += inverse[len(inverse)//2:len(inverse)//2 + 5]  # borderline

results = []
with ProcessPoolExecutor(max_workers=8) as pool:
    for r in pool.map(build_montage, sample):
        if r: results.append(r)

print(f"\nBuilt {len(results)} montages -> {OUT_DIR}/")
print("\nTop WASH_INVERSE samples (VL has color, FE doesn't):")
for r in sample[:10]:
    print(f"  {r['uid'][:14]}  FE: S={r['FE_S']:.0f}% chroma={r['FE_chroma']:.1f}  |  "
          f"VL: S={r['VL_S']:.0f}% chroma={r['VL_chroma']:.1f}")
