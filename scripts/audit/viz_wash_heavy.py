"""Visualize what WASH_HEAVY meshes actually look like — are they really gray or measurement artifact?"""
import os, json, tarfile, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

OUT = '/home/z50057756/tmp/wash_heavy_check'
os.makedirs(OUT, exist_ok=True)

with open('/home/z50057756/tmp/phase2_smoke/color_audit_v2.json') as f:
    data = json.load(f)
wh = [r for r in data['records'] if r['cls'] == 'WASH_HEAVY']
print(f"WASH_HEAVY: {len(wh)}")
# Sort by VL_S asc (most "washed" first)
wh.sort(key=lambda x: x['VL_S'])

FE_BASE = '/mnt/data-alpha-sg-01/team-camera/home/z50057756/FluffyElephant_tar'
fe_index = {}
for sub in sorted(os.listdir(FE_BASE)):
    sp = os.path.join(FE_BASE, sub)
    if os.path.isdir(sp):
        for fn in os.listdir(sp):
            if fn.endswith('.tar'):
                fe_index[fn[:-4]] = os.path.join(sp, fn)


def get_views(tar, indices, fe=False):
    out = []
    with tarfile.open(tar) as tf:
        prefix = None
        for m in tf.getmembers():
            if m.name.endswith('.png') and 'depth' not in m.name:
                if '/' in m.name: prefix = m.name.rsplit('/',1)[0]+'/'
                elif m.name.startswith('./'): prefix = './'
                else: prefix = ''
                break
        for i in indices:
            try:
                f = tf.extractfile(f"{prefix}{i:03d}.png")
                arr = np.array(Image.open(io.BytesIO(f.read())))
                if arr.ndim < 3 or arr.shape[2] < 4: continue
                a = arr[..., 3:4].astype(np.float32)/255
                rgb = (arr[..., :3] * a + 255*(1-a)).astype(np.uint8)
                # Per-view RGB
                fg = arr[..., 3] > 127
                if fg.sum() < 50:
                    mean_rgb = (0,0,0)
                else:
                    mean_rgb = tuple(int(x) for x in arr[..., :3][fg].mean(axis=0))
                out.append((i, rgb, mean_rgb))
            except Exception: continue
    return out


CELL = 180
GRID = 5
LABEL_H = 28
H = LABEL_H * 2 + CELL * 2
W = CELL * GRID
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
except IOError:
    font = font_s = ImageFont.load_default()


def build_one(rec):
    uid = rec['uid']
    vl_tar = f'/home/z50057756/tmp/phase2_smoke/render_out_2000/{uid}.tar'
    fe_tar = fe_index.get(uid)
    if not fe_tar: return None
    vl_views = get_views(vl_tar, [0, 8, 16, 24, 32])
    fe_views = get_views(fe_tar, [0, 5, 10, 15, 20])
    if not vl_views or not fe_views: return None
    canvas = Image.new('RGB', (W, H), (240,240,240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0,0,W,LABEL_H], fill=(40,60,80))
    draw.text((5, 6), f"{uid[:18]}  VL  FE_S={rec['FE_S']:.0f}% chroma={rec['FE_chroma']:.0f}  VL_S={rec['VL_S']:.0f}% chroma={rec['VL_chroma']:.0f}",
              fill=(255,240,200), font=font)
    for idx, (i, rgb, mean) in enumerate(vl_views[:5]):
        im = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
        canvas.paste(im, (idx*CELL, LABEL_H))
        draw.text((idx*CELL+3, LABEL_H+3), f"v{i} RGB={mean}", fill=(255,0,0), font=font_s)
    fy = LABEL_H + CELL
    draw.rectangle([0, fy, W, fy+LABEL_H], fill=(80,40,40))
    draw.text((5, fy+6), "FE (TRELLIS orig)", fill=(255,220,220), font=font)
    for idx, (i, rgb, mean) in enumerate(fe_views[:5]):
        im = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
        canvas.paste(im, (idx*CELL, fy+LABEL_H))
        draw.text((idx*CELL+3, fy+LABEL_H+3), f"v{i} RGB={mean}", fill=(255,0,0), font=font_s)
    out = f"{OUT}/{uid}.png"
    canvas.save(out)
    return uid


# Pick top 10 most extreme + 5 borderline
SAMPLES = wh[:10] + wh[len(wh)//2:len(wh)//2+5]
with ProcessPoolExecutor(max_workers=8) as pool:
    res = [r for r in pool.map(build_one, SAMPLES) if r]
print(f"Built {len(res)} montages -> {OUT}/")
for r in SAMPLES[:10]:
    print(f"  {r['uid'][:14]}  FE_S={r['FE_S']:.0f}% FE_chroma={r['FE_chroma']:.0f}  VL_S={r['VL_S']:.0f}% VL_chroma={r['VL_chroma']:.0f}")
