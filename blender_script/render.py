"""Patched TRELLIS blender_script/render.py for RnG_appearance sim-to-real.

Forked from microsoft/TRELLIS:dataset_toolkits/blender_script/render.py.

Sim-to-real patches:
  A1: per-view 2D lookAt offset (cam_empty.location moves on camera plane)
  A4: per-frame fov + lookat_offset written to transforms.json

Engineering changes:
  - Persistent mode: --jobs_file points to a JSON with [{mesh_path, output_dir, views, ...}]
                     One Blender process loops over all jobs (skip Python+plugin startup)
  - --device GPU/CPU explicitly picks Cycles backend (OPTIX/OPENIMAGEDENOISE)
  - scene.render.use_persistent_data = True (BVH shared between frames)
  - Per-view depth min/max written to transforms.json (TRELLIS already does this)

Everything else identical to TRELLIS: init_render (128 spp, bounces 1/1/3/3,
denoise, RGBA, film_transparent), init_lighting (key+top+bottom), normalize_scene,
init_nodes depth as 16-bit PNG via MapRange.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Callable, Dict, Tuple

import bpy
import numpy as np
from mathutils import Vector


# ============================== TRELLIS PRESERVED ==============================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.wm.stl_import if hasattr(bpy.ops.wm, "stl_import") else bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.wm.ply_import if hasattr(bpy.ops.wm, "ply_import") else bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    "PNG": "png", "JPEG": "jpg", "OPEN_EXR": "exr",
    "TIFF": "tiff", "BMP": "bmp", "HDR": "hdr", "TARGA": "tga",
}


def init_render(engine="CYCLES", resolution=512, device="GPU", samples=128,
                geo_mode=False):
    """TRELLIS settings, with device picker and persistent_data added."""
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.film_transparent = True
    # WASH BUG FIX: use_persistent_data=True + init_scene per-obj cleanup causes
    # Cycles BVH/texture cache to leak across render() calls, producing stochastic
    # texture wash (~6.4% of meshes rendered gray/wrong-colored in persistent mode).
    # Verified via wash_persistent_test.py: same mesh renders different colors across
    # iterations with True; deterministic correct color with False. Set False to
    # force clean state per render (small perf cost — BVH rebuild dominated by ray-trace).
    bpy.context.scene.render.use_persistent_data = False

    bpy.context.scene.cycles.samples = samples if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = "BOX"
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True

    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "GPU":
        prefs.compute_device_type = "OPTIX"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.denoiser = "OPTIX"
        prefs.refresh_devices()
        for d in prefs.devices:
            d.use = d.type in ("OPTIX", "CUDA")
    else:
        prefs.compute_device_type = "NONE"
        bpy.context.scene.cycles.device = "CPU"
        bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"
    print(f"[init_render] engine={engine} device={device} samples={samples} "
          f"compute={prefs.compute_device_type}", flush=True)


def init_nodes(save_depth=True, base_path=""):
    """Compositor depth pass → 16-bit PNG via MapRange (TRELLIS approach).

    Blender 4.2 quirk: `OutputFile.file_slots[0].path` is ALWAYS resolved
    relative to `base_path` (if absolute it gets concatenated to CWD as if
    relative). Fix: set base_path to the per-object output_dir, and use a
    bare filename in file_slots[0].path each frame.
    """
    outputs = {}
    spec_nodes = {}
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[0].use_pass_z = save_depth

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)

    render_layers = nodes.new("CompositorNodeRLayers")

    if save_depth:
        depth_file_output = nodes.new("CompositorNodeOutputFile")
        depth_file_output.base_path = base_path
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = "PNG"
        depth_file_output.format.color_depth = "16"
        depth_file_output.format.color_mode = "BW"
        depth_map = nodes.new(type="CompositorNodeMapRange")
        depth_map.inputs[1].default_value = 0
        depth_map.inputs[2].default_value = 10
        depth_map.inputs[3].default_value = 0
        depth_map.inputs[4].default_value = 1
        links.new(render_layers.outputs["Depth"], depth_map.inputs[0])
        links.new(depth_map.outputs[0], depth_file_output.inputs[0])
        outputs["depth"] = depth_file_output
        spec_nodes["depth_map"] = depth_map
    return outputs, spec_nodes


def init_scene():
    # 1. Explicitly remove top-level objects + named datablocks
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
    # 2. Purge ALL orphan datablocks (meshes, lights, cameras, curves,
    #    armatures, actions, node_groups, collections, ...). Without this,
    #    `bpy.data.objects.remove()` leaves the underlying ObjectData behind,
    #    accumulating ~10-50 MB per obj → 65-70 GB after a few thousand objs
    #    (see plan section 15 for the OOM autopsy of job 67884).
    bpy.ops.outliner.orphans_purge(do_local_ids=True,
                                   do_linked_ids=True,
                                   do_recursive=True)


def init_camera():
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam, cam_empty   # return both so we can move target per-frame (A1)


def init_lighting():
    """TRELLIS 3-light setup (key/top/bottom)."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    key = bpy.data.objects.new("Default_Light",
                               bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(key)
    key.data.energy = 1000
    key.location = (4, 1, 6)

    top = bpy.data.objects.new("Top_Light",
                               bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top)
    top.data.energy = 10000
    top.location = (0, 0, 10)
    top.scale = (100, 100, 100)

    bot = bpy.data.objects.new("Bottom_Light",
                               bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bot)
    bot.data.energy = 1000
    bot.location = (0, 0, -10)
    return {"key": key, "top": top, "bottom": bot}


def load_object(object_path: str):
    ext = object_path.split(".")[-1].lower()
    import_function = IMPORT_FUNCTIONS[ext]
    if ext == "blend":
        import_function(directory=object_path, link=False)
    elif ext in ("glb", "gltf"):
        import_function(filepath=object_path, merge_vertices=True,
                        import_shading="NORMALS")
    else:
        import_function(filepath=object_path)


def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


class SkipRender(Exception):
    """Signal from Layer 1a/1c to render_one_object: skip this mesh."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _robust_vertex_extent(n_mad=None):
    """Collect all mesh vertex positions in world frame, return robust
    bbox (lo, hi) using MAD-based outlier filtering.

    MAD (Median Absolute Deviation) is robust to outliers — unlike fixed
    percentile cutoffs that always trim 0.5-1% of vertices regardless of
    distribution, MAD adapts: clean meshes get 0 vertices trimmed, meshes
    with rogue outliers get exactly those points trimmed.

    Algorithm:
      1. Collect all vertex world positions
      2. median_pos = 3D median of all verts
      3. distances = norm(vert - median_pos)
      4. median_dist = median(distances)
      5. mad = median(|distances - median_dist|)
      6. inliers = verts with distance < median_dist + n_mad × mad

    n_mad controls how aggressive: 3 = strict (1% cut on Gaussian),
    5 = balanced (0.01% cut), 7 = lenient (almost zero cut on real distributions).
    Default (env MAD_N) = 5.
    """
    if n_mad is None:
        n_mad = float(os.environ.get('MAD_N', '5.0'))

    coords = []
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        m = obj.matrix_world
        for v in obj.data.vertices:
            wv = m @ v.co
            coords.append((wv.x, wv.y, wv.z))
    if not coords:
        return None, None
    arr = np.asarray(coords, dtype=np.float64)

    median_pos = np.median(arr, axis=0)
    distances = np.linalg.norm(arr - median_pos, axis=1)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))

    if mad < 1e-9:
        # All verts effectively at same position — no outliers
        inliers = arr
    else:
        threshold = median_dist + n_mad * mad
        inliers = arr[distances < threshold]
        if len(inliers) < max(10, 0.5 * len(arr)):
            # Sanity: don't trim more than half the verts — fall back to all
            inliers = arr

    lo = inliers.min(axis=0)
    hi = inliers.max(axis=0)
    return Vector(lo.tolist()), Vector(hi.tolist())


def normalize_scene() -> Tuple[float, Vector]:
    """Scene fits unit cube, bbox center → origin.

    Phase 2 Layer 1a (degenerate-mesh gate, no robust outlier handling):
      - SkipRender if bbox non-finite or empty (catches truly broken meshes).
      - SkipRender if shortest axis < 1e-4 × longest axis (planar / collapsed).
      - Use the raw bbox extent for scale (vanilla TRELLIS behavior). Meshes
        with rogue outlier vertices that blow up the scale are accepted but
        rely on Layer 1c (post-render alpha check) to filter empty renders.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values()
                          if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    ext = bbox_max - bbox_min
    if not all(math.isfinite(x) for x in (ext[0], ext[1], ext[2])):
        raise SkipRender("non_finite_bbox")
    ext_max = max(ext[0], ext[1], ext[2])
    ext_min = min(ext[0], ext[1], ext[2])
    if ext_max < 1e-6:
        raise SkipRender("empty_bbox")
    if ext_min < 1e-4 * ext_max:
        raise SkipRender("degenerate_planar_mesh")

    scale = 1 / ext_max
    scene.scale = scene.scale * scale
    bpy.context.view_layer.update()
    bbox_min2, bbox_max2 = scene_bbox()
    offset = -(bbox_min2 + bbox_max2) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def get_transform_matrix(obj):
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = [rt[ii][jj] for jj in range(3)]
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


def _project_object_center(cam, resolution):
    """Project world origin (0,0,0) onto image pixel coords using current cam.
    Returns (u_pix, v_pix) or None if behind camera / out of frame."""
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    co_2d = world_to_camera_view(bpy.context.scene, cam, Vector((0.0, 0.0, 0.0)))
    if co_2d.z <= 0:
        return None
    u_pix = int(co_2d.x * resolution)
    v_pix = int((1.0 - co_2d.y) * resolution)  # flip y to image convention
    return u_pix, v_pix


def _draw_markers(draw, cx, cy, u_pix, v_pix, R):
    """Draw red(center) + orange(object) + yellow connector on an ImageDraw."""
    rad = max(4, R // 100)
    if u_pix is not None and 0 <= u_pix < R and 0 <= v_pix < R:
        draw.line([(cx, cy), (u_pix, v_pix)],
                  fill=(255, 255, 0, 200), width=max(1, R // 256))
        draw.ellipse([u_pix - rad, v_pix - rad, u_pix + rad, v_pix + rad],
                     fill=(255, 128, 0, 255),
                     outline=(255, 255, 255, 255), width=1)
    # red dot last so it's always visible
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                 fill=(255, 0, 0, 255),
                 outline=(255, 255, 255, 255), width=1)


def draw_debug_overlay(png_path: str, cam, resolution: int,
                      depth_png_path: str = None,
                      depth_viz_path: str = None):
    """DEBUG: overlay markers on rendered RGB PNG; if depth files given,
    also produce a colorized depth visualization PNG with same markers.

    Saves:
      - {png_path}: RGB with markers (in place)
      - {depth_viz_path}: colorized depth + markers (NEW file, doesn't touch
        the original {depth_png_path} which is used for training)

    Marker scheme:
      - Red filled circle at image center
      - Orange filled circle at projected world origin (object center)
      - Yellow line connecting them
    """
    from PIL import Image, ImageDraw
    import numpy as np

    R = resolution
    cx, cy = R // 2, R // 2
    proj = _project_object_center(cam, R)
    u_pix, v_pix = (proj if proj else (None, None))

    # --- RGB overlay ---
    img = Image.open(png_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    _draw_markers(draw, cx, cy, u_pix, v_pix, R)
    img.save(png_path)

    # --- Depth visualization with markers ---
    if depth_png_path and depth_viz_path and os.path.exists(depth_png_path):
        try:
            dep = np.array(Image.open(depth_png_path))
        except Exception as e:
            print(f"[overlay] depth load failed: {e}", flush=True)
            return

        # Foreground / background mask
        bg_mask = dep >= 65534
        fg_vals = dep[~bg_mask]
        if fg_vals.size > 0:
            # Normalize fg to [0, 1] then apply viridis-like colormap
            d_min, d_max = fg_vals.min(), fg_vals.max()
            if d_max > d_min:
                norm = (dep.astype(np.float32) - d_min) / (d_max - d_min)
            else:
                norm = np.zeros_like(dep, dtype=np.float32)
            norm = np.clip(norm, 0.0, 1.0)
            # Hand-rolled viridis-like colormap (BGR-friendly via RGB cubic mix)
            # Using a simple turbo-like: blue → green → yellow → red
            r_ch = np.clip(1.5 * (norm - 0.0), 0, 1)
            g_ch = np.clip(1.5 - 1.5 * np.abs(norm - 0.5), 0, 1)
            b_ch = np.clip(1.5 * (1.0 - norm), 0, 1)
            rgb = np.stack([(r_ch * 255).astype(np.uint8),
                             (g_ch * 255).astype(np.uint8),
                             (b_ch * 255).astype(np.uint8)], axis=-1)
            rgb[bg_mask] = (255, 255, 255)  # bg = white
        else:
            rgb = np.full((R, R, 3), 255, dtype=np.uint8)

        viz = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        # Resize if needed (depth may not match resolution after compositor)
        if viz.size != (R, R):
            viz = viz.resize((R, R), Image.NEAREST)
        draw2 = ImageDraw.Draw(viz)
        _draw_markers(draw2, cx, cy, u_pix, v_pix, R)
        viz.save(depth_viz_path)


# ============================== PER-JOB RENDER ==============================

def render_one_object(mesh_path: str, output_dir: str, views: list,
                      args: argparse.Namespace):
    """Render one object's N views. Assumes init_render/init_nodes already done.

    Phase 2 layers (skip-on-broken paths):
      - Layer 1a: normalize_scene raises SkipRender for degenerate bbox →
        write phases.json with status=skipped_<reason>, return early.
      - Layer 1b: thin-obj adaptive zoom — compute k from bbox aspect after
        normalize_scene; apply k to all 40 view radii.
      - Layer 1c: post-render alpha check on view 0 — if first frame is
        all-empty, write phases.json status=skipped_empty_alpha, return.
    """
    os.makedirs(output_dir, exist_ok=True)
    init_scene()
    load_object(mesh_path)

    # Layer 1a: degenerate-mesh gate
    try:
        scale, offset = normalize_scene()
    except SkipRender as e:
        print(f"[SKIP 1a] {mesh_path}: {e.reason}", flush=True)
        phases = {"mesh_path": mesh_path, "status": f"skipped_{e.reason}",
                  "num_views": 0}
        with open(os.path.join(output_dir, "phases.json"), "w") as f:
            json.dump(phases, f, indent=2)
        return

    cam, cam_empty = init_camera()
    init_lighting()

    # Layer 1b: Strategy E adaptive zoom — geometric target.
    # Set ENABLE_STRATEGY_E=0 to disable.
    ENABLE_STRATEGY_E = os.environ.get('ENABLE_STRATEGY_E', '1') == '1'
    R_FRAME = float(os.environ.get('R_FRAME', '0.684'))
    #
    # Geometric formula (replaces old fg_target-based formula):
    #   silhouette extent at k=1 = 2·atan(0.5·sin(fov/2)/r_frame) / fov  (canvas fraction)
    #   want long-axis silhouette to fill TARGET_SILHOUETTE of canvas
    #   → k_zoom = silhouette_at_k1 / TARGET_SILHOUETTE
    #
    # Thin rod's max possible fg ≈ 0.05 (1×0.05 silhouette / canvas) when long
    # axis fills canvas — bounded by geometry, NOT by fg_target. Old formula
    # tried to force fg=0.20 → camera too close → 99% clip on triggered.
    #
    # Aspect filter (skip flat plates):
    #   pen  1×0.05×0.05:  aspect_long=20  aspect_second=20  → trigger
    #   plate 1×1×0.01:    aspect_long=100 aspect_second=1   → skip
    #   chair 1×0.5×0.4:   aspect_long=2.5 aspect_second=2   → skip
    ASPECT_LONG_THRESH = 5.0
    ASPECT_SECOND_THRESH = 3.0
    TARGET_SILHOUETTE = 0.95   # fill 95% of canvas axis-aligned (5% margin)
    bbox_min2, bbox_max2 = scene_bbox()
    ext2 = bbox_max2 - bbox_min2
    ext2_sorted = sorted([ext2[0], ext2[1], ext2[2]], reverse=True)
    a_ext = ext2_sorted[0]
    b_ext = max(ext2_sorted[1], 1e-3)
    c_ext = max(ext2_sorted[2], 1e-3)
    aspect_long = a_ext / c_ext
    aspect_second = a_ext / b_ext
    k_zoom = 1.0
    if not ENABLE_STRATEGY_E:
        print(f"[Strategy E OFF] (ENABLE_STRATEGY_E=0)", flush=True)
    elif aspect_long > ASPECT_LONG_THRESH and aspect_second > ASPECT_SECOND_THRESH:
        # Use this obj's FoV (all 40 views share via A2)
        fov_rad = views[0]["fov"]
        # silhouette extent at k=1 in canvas fraction.
        # Plan C: use actual long axis a_ext (NOT hard-coded 0.5).
        # For vanilla case raw bbox == robust bbox → a_ext = 1.0, formula
        # collapses to the original `0.5 * sin(fov/2) / R_FRAME`.
        # For robust-normalize cases where MAD cut outliers, robust scale
        # > vanilla scale, so the raw bbox in normalized world reaches
        # a_ext = max(raw)/max(robust) > 1. Plugging a_ext/2 into the
        # arctan tells the formula "main body is already larger than unit"
        # → silhouette_k1 saturates → k_raw clamps to 1.0 → no double-zoom.
        silhouette_k1 = 2.0 * math.atan((a_ext / 2) * math.sin(fov_rad/2) / R_FRAME) / fov_rad
        k_raw = silhouette_k1 / TARGET_SILHOUETTE
        k_zoom = max(0.5, min(1.0, k_raw))
        print(f"[Strategy E] a_ext={a_ext:.2f} aspect_long={aspect_long:.2f} aspect_second={aspect_second:.2f} "
              f"fov={math.degrees(fov_rad):.1f}° silhouette_k1={silhouette_k1:.3f} k={k_zoom:.3f}",
              flush=True)
        for v in views:
            v["radius"] *= k_zoom
    elif aspect_long > ASPECT_LONG_THRESH:
        print(f"[Strategy E SKIP plate-like] aspect_long={aspect_long:.2f} "
              f"aspect_second={aspect_second:.2f}", flush=True)

    # Re-create nodes with per-object base_path so depth files land in output_dir
    # (B1 fix: Blender treats absolute file_slots paths as relative-to-CWD)
    outputs, spec_nodes = init_nodes(save_depth=True, base_path=output_dir)

    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": [],
    }

    per_view_times = []
    per_view_phases = []  # 4-stage timing: setup, render, post, meta
    for i, view in enumerate(views):
        # ---- Phase A: setup (camera, lookAt, FoV, depth range, output paths)
        t_a0 = time.perf_counter()
        radius = view["radius"]
        yaw = view["yaw"]
        pitch = view["pitch"]
        fov = view["fov"]
        dx_img, dy_img = view["lookat_offset"]  # 2D offset in camera-plane

        # Camera position on sphere
        cx = radius * math.cos(yaw) * math.cos(pitch)
        cy = radius * math.sin(yaw) * math.cos(pitch)
        cz = radius * math.sin(pitch)
        cam.location = (cx, cy, cz)

        # A1: project (dx, dy) onto camera's right/up plane in world frame
        if dx_img == 0.0 and dy_img == 0.0:
            cam_empty.location = (0.0, 0.0, 0.0)
        else:
            view_dir = -Vector((cx, cy, cz)).normalized()
            world_up = Vector((0.0, 0.0, 1.0))
            right = view_dir.cross(world_up)
            if right.length < 1e-6:
                right = Vector((1.0, 0.0, 0.0))
            right.normalize()
            up = right.cross(view_dir).normalized()
            cam_empty.location = right * dx_img + up * dy_img

        # FoV → focal length (sensor_width=32, so lens = 16 / tan(fov/2))
        cam.data.lens = 16.0 / math.tan(fov / 2.0)

        # Depth MapRange — depth bounds per frame (TRELLIS formula adapted to per-view radius)
        spec_nodes["depth_map"].inputs[1].default_value = radius - 0.5 * math.sqrt(3)
        spec_nodes["depth_map"].inputs[2].default_value = radius + 0.5 * math.sqrt(3)

        # Output paths
        # RGB: absolute path works for scene.render.filepath
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"{i:03d}.png")
        # Depth: RELATIVE filename — gets joined to depth_file_output.base_path (=output_dir)
        for name, out in outputs.items():
            out.file_slots[0].path = f"{i:03d}_{name}"

        bpy.context.view_layer.update()
        t_a1 = time.perf_counter()

        # ---- Phase B: render (Cycles + denoise + compositor file writes)
        bpy.ops.render.render(write_still=True)
        t_b1 = time.perf_counter()
        per_view_times.append(t_b1 - t_a1)

        # Layer 1c: post-render fg ratio check on first view only.
        #
        # r062 smoke found: alpha.max() < 3 is too lenient — 8 broken obj
        # (43da08b6, 615e79f0, 48a915e2, 2d89f86c, 62b00923, 5d8ee3d4,
        # 895ce9cb, 43e53a57) had alpha_max=255 in view 0 (one fully opaque
        # pixel) but the visible silhouette was < 0.1% of canvas — useless
        # for training. The robust-normalize_scene fix above SHOULD prevent
        # this from happening; this Layer 1c is the safety net using the
        # same fg_ratio metric (> 127 alpha) as the audit (threshold 0.005).
        if i == 0:
            png_path = os.path.join(output_dir, f"{i:03d}.png")
            try:
                from PIL import Image
                arr = np.array(Image.open(png_path))
                if arr.ndim == 3 and arr.shape[2] >= 4:
                    fg_ratio = float((arr[..., 3] > 127).mean())
                    if fg_ratio < 0.005:  # < 0.5% canvas covered = useless
                        print(f"[SKIP 1c] {mesh_path}: empty_alpha "
                              f"(view 0 fg_ratio={fg_ratio:.5f})", flush=True)
                        phases = {
                            "mesh_path": mesh_path,
                            "status": "skipped_empty_alpha",
                            "num_views": 1,
                            "fg_ratio_view0": fg_ratio,
                        }
                        with open(os.path.join(output_dir, "phases.json"), "w") as f:
                            json.dump(phases, f, indent=2)
                        return
            except Exception as e:
                print(f"[WARN] Layer 1c fg_ratio check failed: "
                      f"{type(e).__name__}: {e}", flush=True)

        # ---- Phase C: post-process (rename _depth0001 → _depth, optional overlay)
        for name, out in outputs.items():
            ext = EXT[out.format.file_format]
            target = os.path.join(output_dir, f"{i:03d}_{name}.{ext}")
            matches = sorted(
                glob.glob(os.path.join(output_dir, f"{i:03d}_{name}*.{ext}"))
            )
            src = None
            for m in matches:
                if m != target:
                    src = m
                    break
            if src is not None:
                os.replace(src, target)
            elif not os.path.exists(target):
                print(f"[WARN] view {i:03d}: no {name} file matched glob "
                      f"{i:03d}_{name}*.{ext} in {output_dir}", flush=True)

        if getattr(args, "debug_overlay", False):
            try:
                draw_debug_overlay(
                    png_path=os.path.join(output_dir, f"{i:03d}.png"),
                    cam=cam,
                    resolution=args.resolution,
                    depth_png_path=os.path.join(output_dir, f"{i:03d}_depth.png"),
                    depth_viz_path=os.path.join(output_dir, f"{i:03d}_depth_viz.png"),
                )
            except Exception as e:
                print(f"[WARN] view {i:03d}: debug_overlay failed: "
                      f"{type(e).__name__}: {e}", flush=True)
        t_c1 = time.perf_counter()

        # ---- Phase D: metadata
        metadata = {
            "file_path": f"{i:03d}.png",
            "camera_angle_x": fov,
            "fov": fov,
            "yaw": yaw,
            "pitch": pitch,
            "radius": radius,
            "lookat_offset": [dx_img, dy_img, 0.0],
            "transform_matrix": get_transform_matrix(cam),
            "depth": {
                "min": radius - 0.5 * math.sqrt(3),
                "max": radius + 0.5 * math.sqrt(3),
            },
            "render_seconds": t_b1 - t_a1,
        }
        to_export["frames"].append(metadata)
        t_d1 = time.perf_counter()

        per_view_phases.append({
            "setup": t_a1 - t_a0,
            "render": t_b1 - t_a1,
            "post": t_c1 - t_b1,
            "meta": t_d1 - t_c1,
            "total": t_d1 - t_a0,
        })
        if i < 3 or i == len(views) - 1:
            print(f"  view {i:03d}: fov={math.degrees(fov):.1f}° "
                  f"r={radius:.2f} offset=({dx_img:+.3f},{dy_img:+.3f}) "
                  f"setup={t_a1-t_a0:.3f}s render={t_b1-t_a1:.3f}s "
                  f"post={t_c1-t_b1:.3f}s meta={t_d1-t_c1:.3f}s",
                  flush=True)

    # Save transforms.json
    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump(to_export, f, indent=2)

    # Per-object benchmark
    bench = {
        "mesh_path": mesh_path,
        "num_views": len(views),
        "samples": args.samples,
        "resolution": args.resolution,
        "device": args.device,
        "per_view_seconds": per_view_times,
        "per_view_mean": float(np.mean(per_view_times)),
        "per_view_std": float(np.std(per_view_times)),
        "total_render_sec": float(np.sum(per_view_times)),
    }
    with open(os.path.join(output_dir, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=2)

    # 4-stage per-view timing profile
    stage_totals = {
        k: float(sum(p[k] for p in per_view_phases))
        for k in ("setup", "render", "post", "meta", "total")
    }
    stage_means = {
        k: float(np.mean([p[k] for p in per_view_phases]))
        for k in ("setup", "render", "post", "meta", "total")
    }
    phases = {
        "mesh_path": mesh_path,
        "status": "ok",          # Phase 2 Layer 1a/1c skip with "skipped_<reason>"
        "num_views": len(views),
        "k_zoom": k_zoom,        # Strategy E adaptive zoom applied (1.0 if no zoom)
        "a_ext": float(a_ext),   # raw long axis after robust normalize (=1.0 if no outliers)
        "aspect_long": float(aspect_long),
        "aspect_second": float(aspect_second),
        "per_view": per_view_phases,
        "total_sec": stage_totals,
        "mean_sec": stage_means,
    }
    with open(os.path.join(output_dir, "phases.json"), "w") as f:
        json.dump(phases, f, indent=2)


# ================================== main ==================================

def _rss_gb():
    """Read process RSS in GB from /proc (no psutil dep)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return -1.0


def _flush_tar_batch(uids, scratch_dir, final_output_dir):
    """Tar each uid in scratch_dir, atomically move to final_output_dir/<uid>.tar.
    Skips uids without phases.json (incomplete). Uses .<uid>.tar.tmp staging +
    os.rename for crash-safe atomic publish (R4, R7 mitigations). Returns count
    of obj successfully published."""
    import shutil
    import subprocess as _sp
    if not (scratch_dir and final_output_dir):
        return 0
    n_done = 0
    n_skipped = 0
    for uid in uids:
        scratch_obj = os.path.join(scratch_dir, uid)
        if not os.path.isdir(scratch_obj):
            continue
        phases_path = os.path.join(scratch_obj, "phases.json")
        if not os.path.exists(phases_path):
            print(f"[tar] WARN no phases.json for {uid}, skip", flush=True)
            continue
        # Phase 2: Layer 1a/1c may have marked this obj as skipped.
        # Don't tar broken obj; record to quarantine.txt for audit.
        try:
            with open(phases_path) as f:
                phases = json.load(f)
            status = phases.get("status", "ok")
        except Exception:
            status = "ok"
        if status != "ok":
            print(f"[tar] SKIP {uid}: {status}", flush=True)
            quarantine_path = os.path.join(final_output_dir, "quarantine.txt")
            try:
                with open(quarantine_path, "a") as f:
                    f.write(f"{uid}\t{status}\n")
            except Exception as e:
                print(f"[tar] WARN quarantine.txt write failed: {e}", flush=True)
            shutil.rmtree(scratch_obj, ignore_errors=True)
            n_skipped += 1
            continue
        tar_local = os.path.join(scratch_dir, uid + ".tar")
        tar_tmp = os.path.join(final_output_dir, "." + uid + ".tar.tmp")
        tar_final = os.path.join(final_output_dir, uid + ".tar")
        if os.path.exists(tar_final):
            shutil.rmtree(scratch_obj, ignore_errors=True)
            continue
        rc = _sp.call(["tar", "-cf", tar_local, "-C", scratch_dir, uid])
        if rc != 0:
            print(f"[tar] WARN tar rc={rc} for {uid}", flush=True)
            continue
        try:
            shutil.move(tar_local, tar_tmp)
            os.rename(tar_tmp, tar_final)
        except Exception as e:
            print(f"[tar] WARN move/rename failed for {uid}: {e}", flush=True)
            for p in (tar_tmp, tar_local):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            continue
        shutil.rmtree(scratch_obj, ignore_errors=True)
        n_done += 1
    if n_skipped > 0:
        print(f"[tar] batch done: {n_done} tarred, {n_skipped} quarantined "
              f"(Layer 1a/1c)", flush=True)
    return n_done


def main(args):
    # Initialize render engine ONCE for the entire Blender invocation
    init_render(engine=args.engine, resolution=args.resolution,
                device=args.device, samples=args.samples)

    # Load jobs
    with open(args.jobs_file) as f:
        jobs = json.load(f)
    if args.only_uid is not None:
        jobs = [j for j in jobs if j["uid"] == args.only_uid]
    print(f"[main] {len(jobs)} jobs to render", flush=True)

    t0 = time.perf_counter()
    to_tar = []  # uids completed but not yet flushed this batch (R1/R2 mitigation)
    for k, j in enumerate(jobs):
        print(f"\n[main] === job {k+1}/{len(jobs)}: {j['uid']} ===", flush=True)
        t_obj = time.perf_counter()
        try:
            render_one_object(j["mesh_path"], j["output_dir"], j["views"], args)
        except Exception as e:
            print(f"[main] FAILED {j['uid']}: {type(e).__name__}: {e}",
                  flush=True)
            import traceback; traceback.print_exc()
            continue
        to_tar.append(j["uid"])
        # Memory health probe: meshes/lights/cameras count should stay near 1-3
        # (current obj only); RSS should plateau, not grow. If RSS climbs past
        # ~20 GB across many objs, orphans_purge is failing.
        print(f"[main] {j['uid']} done in {time.perf_counter()-t_obj:.1f}s  "
              f"[mem] meshes={len(bpy.data.meshes)} "
              f"lights={len(bpy.data.lights)} cameras={len(bpy.data.cameras)} "
              f"images={len(bpy.data.images)} rss={_rss_gb():.2f}GB",
              flush=True)
        # Per-batch tar flush: every N obj, write tars to final OUTPUT_DIR and
        # free scratch. Bounds /tmp use and limits preemption-loss to N objs.
        is_last = (k == len(jobs) - 1)
        if (args.scratch_dir and args.final_output_dir
                and (len(to_tar) >= args.tar_batch_size or is_last)):
            t_b = time.perf_counter()
            n_packed = _flush_tar_batch(
                to_tar, args.scratch_dir, args.final_output_dir)
            print(f"[tar-batch] flushed {n_packed}/{len(to_tar)} objs in "
                  f"{time.perf_counter()-t_b:.1f}s", flush=True)
            to_tar = []
    print(f"\n[main] all done in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs_file", required=True,
                        help="JSON file with [{mesh_path, uid, output_dir, views}, ...]")
    parser.add_argument("--only_uid", default=None,
                        help="If set, render only this UID from the jobs file")
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--debug_overlay", action="store_true",
                        help="Draw red cross at image center + orange dot at "
                             "projected object center on each rendered PNG.")
    # Per-batch tar flush args. If both --scratch_dir and --final_output_dir
    # are set, main loop tars + moves completed objs every --tar_batch_size
    # objs, freeing scratch (R1) and capping preempt-loss to batch_size (R2).
    parser.add_argument("--scratch_dir", default=None)
    parser.add_argument("--final_output_dir", default=None)
    parser.add_argument("--tar_batch_size", type=int, default=500)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(parser.parse_args(argv))
