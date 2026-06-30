"""Orchestrator: generate per-view camera parameters (TRELLIS Hammersley
base + our patches A1/A2/R2) and invoke Blender to render one or many objects.

Patches vs TRELLIS original `dataset_toolkits/render.py`:
- A1: per-view 2D lookAt offset on the object plane (off-center rendering)
- A2: per-OBJECT FoV (one FoV per obj, all 40 views share; radius derived to
      keep object framing constant via r_frame = 0.684 / sin(fov/2))
- A3: TRELLIS-original pitch/yaw (sphere_hammersley_sequence untouched —
      already 75/25 upper-hemisphere biased with sphere-surface-correct
      density; NOT clipped)
- R2: per-view radius jitter U(0.95, 1.15) around the FoV-derived base radius
      (asymmetric; biases toward farther to avoid clip)
- Accepts --obj_list (text file of paths) or --object (single GLB)
- Persistent mode: passes the whole obj list to one Blender invocation
  (handled by blender_script/render.py)

All other TRELLIS render settings (128 spp, bounces 1/1/3/3, denoise, 3-light
studio, normalize_scene, RGBA + depth) are preserved by the Blender script.
"""
import argparse
import json
import math
import os
import sys
import time
from subprocess import call

import numpy as np

# Make `utils.py` importable when this script is run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import sphere_hammersley_sequence  # noqa: E402


def _patched_views(num_views: int,
                   fov_min_deg: float = 40.0,
                   fov_max_deg: float = 75.0,
                   offset_max: float = 0.15,
                   offset_zero_prob: float = 0.20,
                   radius_min_factor: float = 0.95,
                   radius_max_factor: float = 1.10,
                   r_frame: float = 0.74,
                   seed: int = None):
    """Generate `num_views` view dicts.

    Patches vs TRELLIS:
      A1: per-view 2D lookAt offset on camera plane (off-center rendering)
      A2 (revised): one FoV per object — matches real single-session capture
      A3 (final = TRELLIS original): sphere_hammersley_sequence pitch/yaw
          untouched (TRELLIS already biases 75% upper hemisphere with
          sphere-surface-correct density)
      R2 (asymmetric): per-view radius jitter in [radius_min_factor,
          radius_max_factor] around the FoV-derived base radius. Default
          [0.95, 1.10] = mean 1.025 (tighter than original [0.95, 1.15]
          which had mean 1.05). Tightening recovers ~5% fg ratio.

    Framing: radius coupled to FoV so r_frame sphere fits the frustum at
    any FoV. r_frame = 0.74 (was TRELLIS-aligned 0.684 = 2·sin(20°)). The
    increase aligns our fg distribution to FluffyElephant (FE p50 = 0.190,
    we were 0.134, gap fully explained by R2+other-render-detail effects
    of A1/A2 sim2real patches, not by mesh distribution — verified by 1:1
    UID-paired comparison on 44,527 shared meshes).

    Each view dict: {yaw, pitch, radius, fov, lookat_offset}
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    sphere_offset = (rng.random(), rng.random())

    # A2 (per-object FoV): single sample for the whole object.
    fov = math.radians(rng.uniform(fov_min_deg, fov_max_deg))
    # r_frame sphere inscribed in frustum at any FoV. r_frame = 0.74 chosen
    # to align fg distribution with FluffyElephant baseline (see docstring).
    # Original TRELLIS r_frame = 2·sin(20°) = 0.684.
    radius_base = r_frame / math.sin(fov / 2.0)

    views = []
    for i in range(num_views):
        # A3 (TRELLIS original): leave yaw/pitch as-is from sphere_hammersley.
        yaw, pitch = sphere_hammersley_sequence(i, num_views, sphere_offset)

        # R2: asymmetric radius jitter (mostly farther to avoid clip)
        radius = radius_base * rng.uniform(radius_min_factor, radius_max_factor)

        # A1: 2D lookAt offset on camera plane, offset_zero_prob chance zero-offset
        if rng.random() < offset_zero_prob:
            offset_xy = (0.0, 0.0)
        else:
            ratio = rng.uniform(0.0, offset_max)
            half_img = radius * math.tan(fov / 2.0)  # world half-extent at z=0
            theta = rng.uniform(0.0, 2 * math.pi)
            r2 = math.sqrt(rng.uniform(0.0, 1.0)) * ratio
            offset_xy = (r2 * half_img * math.cos(theta),
                         r2 * half_img * math.sin(theta))
        views.append({
            "yaw": yaw, "pitch": pitch, "radius": radius,
            "fov": fov, "lookat_offset": [offset_xy[0], offset_xy[1]],
        })
    return views


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--object", type=str,
                     help="Path to a single .glb/.obj/.fbx mesh")
    src.add_argument("--obj_list", type=str,
                     help="Text file with one mesh path per line")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Base output directory (per-obj subdir created inside)")
    ap.add_argument("--num_views", type=int, default=40)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    ap.add_argument("--samples", type=int, default=128,
                    help="Cycles spp (TRELLIS default 128)")
    ap.add_argument("--offset_max", type=float, default=0.15,
                    help="A1: max lookAt offset ratio (fraction of frame half-extent)")
    ap.add_argument("--offset_zero_prob", type=float, default=0.20,
                    help="A1: probability of zero-offset (preserves centered samples)")
    ap.add_argument("--fov_min_deg", type=float, default=40.0,
                    help="A2: min FoV in degrees (sampled once per object). "
                         "Default 40° = phone 2-3× telephoto / DSLR 50mm.")
    ap.add_argument("--fov_max_deg", type=float, default=75.0,
                    help="A2: max FoV in degrees. Default 75° = phone 1× wide.")
    ap.add_argument("--radius_min_factor", type=float, default=0.95,
                    help="R2: min radius factor (allows some closer; default 0.95)")
    ap.add_argument("--radius_max_factor", type=float, default=1.10,
                    help="R2: max radius factor (asymmetric farther; default 1.10)")
    ap.add_argument("--r_frame", type=float, default=0.74,
                    help="Framing sphere radius inscribed in frustum (default 0.74; "
                         "TRELLIS original was 2*sin(20°)=0.684)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blender", type=str,
                    default=os.path.expanduser(
                        "~/tools/blender-4.2.9-linux-x64/blender"))
    ap.add_argument("--persistent", action="store_true",
                    help="Invoke one Blender process for the whole obj_list "
                         "(blender script reads list and loops internally)")
    ap.add_argument("--debug_overlay", action="store_true",
                    help="DEBUG: draw red cross at image center + orange dot at "
                         "projected object center on each rendered PNG. Use this "
                         "to visually validate max_offset_ratio.")
    ap.add_argument("--scratch_dir", default=None,
                    help="If set, render each obj to scratch_dir/<uid> first "
                         "(local SSD), then rsync to output_dir/<uid> after "
                         "Blender exits. Avoids slow JuiceFS per-frame writes.")
    ap.add_argument("--tar_output", action="store_true",
                    help="Tar each obj's render dir into <output_dir>/<uid>.tar "
                         "instead of rsync-ing as a directory tree. Requires "
                         "--scratch_dir. Skips uids whose .tar already exists "
                         "in output_dir (restart-safe). Reduces total file "
                         "count ~82x vs loose files.")
    ap.add_argument("--tar_batch_size", type=int, default=100,
                    help="In persistent mode, blender_script flushes tar+move "
                         "every N completed objs, freeing scratch as we go. "
                         "Bounds /tmp use and caps preempt-loss to N objs. "
                         "Default 100 = preempt-resilient for lowest QOS; "
                         "increase to 500 for stable QOS to reduce overhead.")
    args = ap.parse_args()

    # Build object list
    if args.object:
        obj_list = [os.path.expanduser(args.object)]
    else:
        with open(args.obj_list) as f:
            obj_list = [ln.strip() for ln in f if ln.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    # Clean orphan .<uid>.tar.tmp files from previous crashed jobs (R4/R7 cleanup)
    if args.tar_output:
        import glob as _glob
        orphans = _glob.glob(os.path.join(args.output_dir, ".*.tar.tmp"))
        for o in orphans:
            try:
                os.remove(o)
            except OSError:
                pass
        if orphans:
            print(f"[orchestrator] cleaned {len(orphans)} orphan .tar.tmp files",
                  flush=True)

    # tar_output: skip uids whose .tar already exists in output_dir (restart-safe)
    if args.tar_output:
        orig_count = len(obj_list)
        obj_list = [p for p in obj_list
                    if not os.path.exists(os.path.join(
                        args.output_dir,
                        os.path.splitext(os.path.basename(p))[0] + ".tar"))]
        skipped = orig_count - len(obj_list)
        print(f"[orchestrator] tar_output: {len(obj_list)} objs need rendering "
              f"(skipped {skipped} already-done .tar)", flush=True)
        if not obj_list:
            print("[orchestrator] nothing to do, exiting", flush=True)
            return
    # When --scratch_dir set, each obj is rendered to scratch_dir/<uid>, then
    # rsync'd to output_dir/<uid> after Blender exits. Avoids slow per-frame
    # JuiceFS writes — bulk transfer at end is much faster.
    out_base = args.scratch_dir if args.scratch_dir else args.output_dir
    if args.scratch_dir:
        os.makedirs(args.scratch_dir, exist_ok=True)
        print(f"[orchestrator] scratch mode: renders → {args.scratch_dir}, "
              f"rsync → {args.output_dir}", flush=True)
    blender_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "blender_script", "render.py")

    # Each object gets its own views (with shared seed-derived sub-seed)
    rng = np.random.default_rng(args.seed)

    # Build a single "job spec" JSON file with everything per obj
    jobs = []
    for i, mesh_path in enumerate(obj_list):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        views = _patched_views(
            num_views=args.num_views,
            fov_min_deg=args.fov_min_deg, fov_max_deg=args.fov_max_deg,
            offset_max=args.offset_max,
            offset_zero_prob=args.offset_zero_prob,
            radius_min_factor=args.radius_min_factor,
            radius_max_factor=args.radius_max_factor,
            r_frame=args.r_frame,
            seed=sub_seed,
        )
        # uid derivation: usually basename without ext (e.g. "<obj_uid>.glb" -> "<obj_uid>").
        # Exception: GSO-style nested layout "<scene>/meshes/model.obj" all give uid="model"
        # and clobber each other. Detect that and use the scene dir name instead.
        _basename = os.path.basename(mesh_path)
        if os.path.splitext(_basename)[0] == "model":
            # parent's parent dir → scene name (e.g. "3D_Dollhouse_Sofa")
            uid = os.path.basename(os.path.dirname(os.path.dirname(mesh_path)))
        else:
            uid = os.path.splitext(_basename)[0]
        jobs.append({
            "mesh_path": mesh_path,
            "uid": uid,
            "output_dir": os.path.join(out_base, uid),
            "views": views,
        })

    jobs_file = os.path.join(out_base, "_jobs.json")
    with open(jobs_file, "w") as f:
        json.dump(jobs, f)

    if args.persistent:
        # One Blender invocation processes all objects
        cmd = [
            args.blender, "--background", "--python", blender_script, "--",
            "--jobs_file", jobs_file,
            "--resolution", str(args.resolution),
            "--device", args.device,
            "--samples", str(args.samples),
        ] + (["--debug_overlay"] if args.debug_overlay else [])
        # When tar_output + scratch_dir, let blender_script do per-batch tar
        if args.tar_output and args.scratch_dir:
            cmd += [
                "--scratch_dir", args.scratch_dir,
                "--final_output_dir", args.output_dir,
                "--tar_batch_size", str(args.tar_batch_size),
            ]
        print(f"[orchestrator] persistent mode: 1 Blender for {len(jobs)} objs",
              flush=True)
        t0 = time.perf_counter()
        rc = call(cmd)
        t1 = time.perf_counter()
        print(f"[orchestrator] done in {t1-t0:.1f}s  rc={rc}", flush=True)
    else:
        for j in jobs:
            cmd = [
                args.blender, "--background", "--python", blender_script, "--",
                "--jobs_file", jobs_file,
                "--only_uid", j["uid"],
                "--resolution", str(args.resolution),
                "--device", args.device,
                "--samples", str(args.samples),
            ]
            print(f"[orchestrator] rendering {j['uid']} ...", flush=True)
            t0 = time.perf_counter()
            call(cmd)
            t1 = time.perf_counter()
            print(f"[orchestrator] {j['uid']} done in {t1-t0:.1f}s", flush=True)

    # Post-render: ship from scratch → final output_dir.
    if args.scratch_dir and args.tar_output:
        # Tar each obj locally (NVMe), then move single .tar across filesystems.
        import shutil
        t0 = time.perf_counter()
        n_packed = 0
        for j in jobs:
            scratch_obj = os.path.join(args.scratch_dir, j["uid"])
            if not os.path.isdir(scratch_obj):
                continue
            # B1: skip tar for partially-rendered objs. phases.json is the
            # LAST file blender_script writes, so its presence = complete render.
            if not os.path.exists(os.path.join(scratch_obj, "phases.json")):
                print(f"[tar] WARN incomplete render (no phases.json) for "
                      f"{j['uid']}, skip tar so restart can re-render", flush=True)
                continue
            tar_local = os.path.join(args.scratch_dir, j["uid"] + ".tar")
            tar_final = os.path.join(args.output_dir, j["uid"] + ".tar")
            # -cf no compression: PNGs already compressed
            rc = call(["tar", "-cf", tar_local, "-C", args.scratch_dir, j["uid"]])
            if rc != 0:
                print(f"[tar] WARN tar rc={rc} for {j['uid']}", flush=True)
                continue
            shutil.move(tar_local, tar_final)
            shutil.rmtree(scratch_obj, ignore_errors=True)
            n_packed += 1
        jf = os.path.join(args.scratch_dir, "_jobs.json")
        if os.path.exists(jf):
            os.remove(jf)
        t1 = time.perf_counter()
        print(f"[tar] packed {n_packed} objs in {t1-t0:.1f}s "
              f"({(t1-t0)/max(n_packed,1):.2f}s/obj)", flush=True)
    elif args.scratch_dir:
        # Legacy: rsync each obj from scratch → final output_dir as dir tree
        import shutil
        t0 = time.perf_counter()
        n_synced = 0
        for j in jobs:
            scratch_obj = os.path.join(args.scratch_dir, j["uid"])
            final_obj = os.path.join(args.output_dir, j["uid"])
            if not os.path.isdir(scratch_obj):
                continue
            os.makedirs(args.output_dir, exist_ok=True)
            rc = call(["rsync", "-a", scratch_obj + "/", final_obj + "/"])
            if rc == 0:
                shutil.rmtree(scratch_obj, ignore_errors=True)
                n_synced += 1
            else:
                print(f"[rsync] WARN rsync rc={rc} for {j['uid']}", flush=True)
        jf = os.path.join(args.scratch_dir, "_jobs.json")
        if os.path.exists(jf):
            os.remove(jf)
        t1 = time.perf_counter()
        print(f"[rsync] synced {n_synced} objs in {t1-t0:.1f}s "
              f"({(t1-t0)/max(n_synced,1):.2f}s/obj)", flush=True)


if __name__ == "__main__":
    main()
