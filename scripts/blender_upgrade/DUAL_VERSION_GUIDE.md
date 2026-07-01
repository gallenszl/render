# Blender 版本切换指南 — 4.2.9 ↔ 5.2 双版本共存

## 概述

`blender-5-upgrade` branch 的 `blender_script/render.py` 通过 **runtime API detection** 同时支持:
- **Blender 4.2.9 LTS**(当前生产,已知 wash bug,用 toggle 修复)
- **Blender 5.2 beta**(root fix wash + 快 43%)

**同一份代码,两个 Blender binary,输出格式完全一致**。

## 输出验证(2026-07-01)

同 UID `8fbfab629a474631aba7f80d81609302` 用两版本各渲一次,40 view × 262144 pixel:

| 维度 | 结果 |
|---|---|
| Pearson correlation | mean 0.999994, min 0.999922 |
| Mean \|diff\| (uint16 单位) | 0.0 |
| p95 / p99 diff | 0 / 1 |
| fg 平均 depth | 4.2=22419, 5.2=22419 |
| bg 平均 depth | 都 65535 |
| 物理 bg > fg check | 40/40 通过 |

**结论**: 5.2 depth 跟 4.2.9 numeric-equivalent(仅有边界 pixel 的单位 rounding 差),下游 dataloader 无需任何改动。

## 使用方式

### 用 4.2.9(老 production)
```bash
export BLENDER=/mnt/data-alpha-sg-01/team-camera/home/z50057756/tools/blender-4.2.9-linux-x64/blender
export BLPY=/mnt/data-alpha-sg-01/team-camera/home/z50057756/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11

$BLPY render.py --obj_list list.txt --output_dir out/ ...  --blender "$BLENDER"
```

- 自动路径:`use_persistent_data` toggle each obj(wash workaround,+11s/obj 开销)
- 使用 `scene.cycles.filter_type`
- 使用 `scene.node_tree`(compositor)
- 使用 `CompositorNodeMapRange` for depth normalize
- 深度直出 16-bit BW PNG

### 用 5.2 beta(新)
```bash
export BLENDER=/home/z50057756/tools/blender-5.2-beta/blender
export BLPY=/home/z50057756/tools/blender-5.2-beta/5.2/python/bin/python3.13

$BLPY render.py --obj_list list.txt --output_dir out/ ...  --blender "$BLENDER"
```

- 自动路径:**无 toggle**(5.2 native fix wash)
- 使用 `scene.cycles.pixel_filter_type`(renamed)
- 使用 `scene.compositing_node_group`(replaces node_tree)
- 使用 `ShaderNodeMath`(CompositorNodeMath 被删)
- depth 出 EXR → Python 后处理转 16-bit BW PNG(绕过 5.2 beta 已知 `override_node_format` bug)

## Runtime detection 位置

`blender_script/render.py` 里的 4 处版本检测:

| 位置 | 检测方式 | 作用 |
|---|---|---|
| `init_render()` | `hasattr(cycles, 'pixel_filter_type')` | filter_type API rename |
| `init_nodes()` | `bpy.app.version >= (5, 2, 0)` | node_tree → compositing_node_group,file_slots → file_output_items,MapRange → Math chain |
| `render_one_object()` | `bpy.app.version < (5, 2, 0)` | 4.2/4.5 需要 use_persistent_data toggle,5.2 不需要 |
| 视图循环 depth post-process | `spec_nodes.get("depth_is_v52")` | 5.2 EXR → PNG 转换 |

## 各版本依赖(bundled Python)

### Blender 4.2.9 (Python 3.11)
- `Pillow` `numpy` — 通常 pre-installed 或者:
  ```bash
  /path/to/blender-4.2.9-linux-x64/4.2/python/bin/python3.11 -m pip install Pillow numpy
  ```

### Blender 5.2 beta (Python 3.13)
- `Pillow` `numpy` — 需要 pip 装
- `OpenEXR` — depth 后处理读 EXR
  ```bash
  /home/z50057756/tools/blender-5.2-beta/5.2/python/bin/python3.13 -m pip install Pillow numpy OpenEXR
  ```

## 已知 5.2 beta bug (upstream)

`CompositorNodeOutputFile` 的 `override_node_format=True` + `itm.format='PNG'` 被 silently 忽略,深度始终存为 OPEN_EXR_MULTILAYER 而不是 PNG。

**我们的 workaround**: `_depth_exr_to_16bit_png()` 每 view 读 EXR + 归一化 + 存 16-bit PNG + 删 EXR。开销 ~10-30ms/view(总占 render 时间 5-7%)。

5.2 LTS 正式版(2026-07-14 计划发布)可能修好这个 bug。若修好,可以删掉 workaround,回到 compositor 直出 PNG。

## Metric 对比(2000-obj scale extrapolated from 100-obj)

| Metric | 4.2.9 + toggle(prod) | Blender 5.2 |
|---|---|---|
| Wall 100 obj (4 procs 1 GPU) | ~13 min | 7:21 |
| Wall 44,798 obj (4 GPU × 4 procs) | ~10 h | ~5.5 h |
| Wash rate | ~0% (toggle 修) | 0% (native 修) |
| fg vs FE per-obj median | 0.909 | 0.909 |
| depth format | 16-bit BW PNG | 16-bit BW PNG(相关性 >99.99%)|
| tar structure | 82 files | 82 files |

## 何时切换到 5.2

**推荐**: 5.2 LTS 官方 release(2026-07-14)之后,切主 branch 用 5.2。

在此之前 branch `blender-5-upgrade` 可用于:
- 新 Phase 2 全量渲染(想要 5.5h 完成 vs 10h)
- A/B 对比测试
