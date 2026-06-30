"""Comprehensive integrity check for $HOME/data/objaverse_renders_44798/*.tar.
Parallelized for CPU sbatch (16 workers)."""
import os, glob, json, tarfile, random, time
from collections import Counter, defaultdict
from multiprocessing import Pool

DATA = os.path.expanduser("~/data/objaverse_renders_44798")
LIST = os.path.join(DATA, "obj_list_44798.txt")
NUM_VIEWS_EXPECTED = 40
SAMPLE_DECODE_N = 200

t0 = time.perf_counter()

with open(LIST) as f:
    expected_uids = {os.path.splitext(os.path.basename(l.strip()))[0]
                     for l in f if l.strip()}
print(f"Expected uids in obj_list: {len(expected_uids)}", flush=True)

tars = sorted(glob.glob(os.path.join(DATA, "*.tar")))
actual_uids = {os.path.splitext(os.path.basename(t))[0] for t in tars}
print(f"Actual .tar files:          {len(actual_uids)}", flush=True)

missing = expected_uids - actual_uids
extra = actual_uids - expected_uids
print(f"Missing (.tar not produced): {len(missing)}")
if missing:
    print(f"  first 10: {sorted(missing)[:10]}")
print(f"Extra (.tar not in obj_list): {len(extra)}")
if extra:
    print(f"  first 10: {sorted(extra)[:10]}")

tmps = glob.glob(os.path.join(DATA, ".*.tar.tmp"))
print(f"Orphan .tar.tmp residue:    {len(tmps)}", flush=True)


def check_tar(t):
    uid = os.path.splitext(os.path.basename(t))[0]
    try:
        with tarfile.open(t) as tf:
            names = tf.getnames()
    except Exception as e:
        return uid, f"unreadable:{type(e).__name__}", 0
    rels = {n.split("/", 1)[-1] for n in names if "/" in n}
    rgb = sum(1 for n in rels if n.endswith(".png") and "_depth" not in n)
    dep = sum(1 for n in rels if n.endswith("_depth.png"))
    flag = []
    if "phases.json" not in rels: flag.append("no_phases")
    if "transforms.json" not in rels: flag.append("no_transforms")
    if rgb != NUM_VIEWS_EXPECTED: flag.append(f"rgb={rgb}")
    if dep != NUM_VIEWS_EXPECTED: flag.append(f"depth={dep}")
    return uid, "OK" if not flag else ",".join(flag), len(names)


print(f"\nScanning {len(tars)} tars in parallel (16 workers)...", flush=True)
with Pool(16) as p:
    results = p.map(check_tar, tars, chunksize=200)

counts = Counter(r[2] for r in results)
status = Counter(r[1] for r in results)
problems = defaultdict(list)
for uid, st, nfiles in results:
    if st != "OK":
        problems[st].append(uid)

print(f"\n=== STRUCTURAL CHECK ({time.perf_counter()-t0:.1f}s) ===")
print("status:")
for st, c in status.most_common():
    print(f"  {st}: {c}  ({100*c/len(results):.2f}%)")
print("\nmember count distribution:")
for n, c in sorted(counts.items()):
    print(f"  {n} files: {c} tars  ({100*c/len(results):.2f}%)")
if problems:
    print("\nproblem samples (first 5 each):")
    for cat, uids in problems.items():
        print(f"  {cat}: {len(uids)} tars  e.g. {uids[:5]}")


def deep_check(t):
    uid = os.path.splitext(os.path.basename(t))[0]
    errs = []
    try:
        with tarfile.open(t) as tf:
            try:
                d = json.load(tf.extractfile(f"{uid}/transforms.json"))
                if len(d["frames"]) != NUM_VIEWS_EXPECTED:
                    errs.append(f"frames={len(d['frames'])}")
                req = {"file_path","fov","yaw","pitch","radius","transform_matrix"}
                if not req.issubset(d["frames"][0].keys()):
                    errs.append("transforms_missing_keys")
            except Exception as e:
                errs.append(f"transforms_err:{type(e).__name__}")
            try:
                png = tf.extractfile(f"{uid}/000.png").read(16)
                if png[:8] != b"\x89PNG\r\n\x1a\n":
                    errs.append("png_bad_header")
            except Exception as e:
                errs.append(f"png_err:{type(e).__name__}")
            try:
                dpng = tf.extractfile(f"{uid}/000_depth.png").read(16)
                if dpng[:8] != b"\x89PNG\r\n\x1a\n":
                    errs.append("depth_bad_header")
            except Exception as e:
                errs.append(f"depth_err:{type(e).__name__}")
    except Exception as e:
        errs.append(f"open:{type(e).__name__}")
    return uid, errs


print(f"\n=== DEEP SAMPLE CHECK ({SAMPLE_DECODE_N} random tars) ===", flush=True)
random.seed(42)
sample = random.sample(tars, min(SAMPLE_DECODE_N, len(tars)))
with Pool(16) as p:
    deep = p.map(deep_check, sample)
deep_errs = Counter()
for uid, errs in deep:
    for e in errs:
        deep_errs[e] += 1
if not deep_errs:
    print(f"  All {len(sample)}/{SAMPLE_DECODE_N} sampled tars: "
          f"transforms.json + PNG headers all valid OK")
else:
    for e, c in deep_errs.most_common():
        print(f"  {e}: {c} / {len(sample)}")

print(f"\nTotal scan time: {time.perf_counter()-t0:.1f}s")
