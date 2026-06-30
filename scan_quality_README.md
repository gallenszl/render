# scan_quality.py — Post-render foreground audit

Audits per-obj alpha foreground ratio across all rendered tars in a dataset
directory. Produces `fg_audit_full.json` next to the data, which downstream
tooling uses to generate a `blacklist.txt`.

## Schema

For each `<uid>.tar` (containing `<uid>/000.png ... 039.png` RGBA 8-bit):

```json
{
  "uid": "<32-hex>",
  "fg_per_view": [0.0..1.0, x40],  // (alpha > 127).mean() per frame
  "fg_mean": float,
  "fg_min": float,
  "fg_max": float,
  "fg_std": float,
  "fov_deg": float | null            // from transforms.json frame[0].fov
}
```

Errors are reported as `{"uid", "error": "<reason>"}` instead.

## Run (CPU partition, 96 worker, ~16 min wall for 44,798 obj)

Edit `TAR_DIR` / `OUT_PATH` at the top of scan_quality.py, then:

```bash
sbatch --partition=cpu --cpus-per-task=96 --mem=64G --time=00:30:00 \
       --wrap "AUDIT_WORKERS=\$SLURM_CPUS_PER_TASK \
               /home/z50057756/conda/envs/rng-fa3/bin/python \
               /home/z50057756/code/render_pipeline/scan_quality.py"
```

## Generate blacklist.txt from output

```python
import json
THRESHOLD = 0.005  # fg_mean < this -> blacklist
with open('fg_audit_full.json') as f:
    data = json.load(f)
ok = [r for r in data['results'] if 'error' not in r]
bl = sorted([r['uid'] for r in ok if r['fg_mean'] < THRESHOLD])
with open('blacklist.txt', 'w') as f:
    f.write(f"# threshold fg_mean < {THRESHOLD}\n")
    for uid in bl: f.write(uid + '\n')
```

The dataset loader (`data/dataset_objaverse.py`) auto-detects
`<data_root>/blacklist.txt` and filters UIDs at `__init__` time.

## Role in pipeline (post-Phase 2)

Once Phase 2 Layer 1 (in-render bbox / alpha gate) ships, broken obj will be
rejected at render time and `blacklist.txt` should be nearly empty.
This script remains useful as a post-build sanity audit.
