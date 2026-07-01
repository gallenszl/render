# RnG_appearance 渲染管线 — 最终版

## Context

**项目**：CVPR'26 RnG 论文(arxiv 2603.01194)的延续工作。模型 `VGGT4LVSM` 输入 4 个 posed views,输出 RGB + camera pose + 3D point map。

**目标**:在 TRELLIS 主渲染管线之上做最小化改造,模拟手持相机拍桌面物体的真实分布,缩小 sim-to-real gap,同时保持训练数据分布与 TRELLIS / FluffyElephant 严格对齐。

**部署假设**:
- 真实拍摄时用户可构造白底或后处理 mask 为白色 → 不需要 HDR/复杂背景
- 真实拍摄物体不会精准对齐相机主光轴 → mis-alignment 是主要 gap

**当前 status (2026-06-22)**:**Phase 2 数据 100% 完成** (44,798 obj × 40 view rendered, 375 GB),**Dataloader patches P1-P5 完成并验证通过** (render44798-dataset-patches branch),下一步进入 Phase 1 端到端训练验证。

---

## 1. 最终渲染规格

### 1.1 Sim-to-real 改造 (5 patches in render + 1 in train)

| ID | Patch | 实现 | 解决的 gap |
|---|---|---|---|
| **A1** | LookAt offset | per-view 2D 偏移,max_ratio=**0.15**,**20%** 概率零偏 | 真实手持拍摄物体不精确居中 |
| **A2** | Per-object FoV | 每 obj 采一次 FoV ∈ [25°, 70°],所有 view 共用 | 真实单次会话焦距一致 |
| **A3** | Pitch 分布 | **完全用 TRELLIS 原版** `sphere_hammersley_sequence` (自带 75/25 上半球 bias) | 真实拍摄主要俯视 |
| **A4** | metadata 扩展 | 每帧写 `fov/yaw/pitch/radius/lookat_offset` 进 transforms.json | dataloader 启用 per-frame fov |
| **R2** | Radius jitter | per-view × `U(0.95, 1.15)` 非对称(主往远走避免 clip) | 手臂自然伸缩 |
| **R1** | Camera roll | **不在 render 端做**,留训练时 augment(每 view 独立 ±10°) | 手抖 roll |

### 1.2 严格保留 TRELLIS 原版(确保分布对齐 FluffyElephant)

| 参数 | 值 |
|---|---|
| Cycles `samples` | 128 |
| bounces (diffuse/glossy/transmission/transparent_max) | 1/1/3/3 |
| `filter_type` / `filter_width` | BOX / 1 |
| `use_denoising` | True (OPTIX on GPU / OpenImageDenoise on CPU) |
| `film_transparent` | True (alpha 通道保留) |
| Lighting | 3 灯 studio: key (point) + top (area, 10000W) + bottom (area, 1000W) |
| `normalize_scene` | bbox center → origin, longest axis = 1 |
| TRACK_TO constraint | 用(仅每帧改 target 位置) |
| Resolution | 512×512 |
| 输出 | RGBA PNG (8-bit) + 16-bit depth PNG, 40 views/obj |

### 1.3 Framing 公式(关键)

```python
# Per-object FoV (A2)
fov = math.radians(uniform(25, 70))

# Keep TRELLIS's r_frame = 2·sin(20°) = 0.684 inscribed in frustum
radius_base = 2.0 * math.sin(math.radians(40)/2.0) / math.sin(fov/2.0)
#           = 0.684 / sin(fov/2)

# Per-view R2 jitter
radius_view = radius_base * uniform(0.95, 1.15)
```

**含义**:任意 FoV 下,"框选球"半径 r_frame=0.684 都内切视椎台 → 物体在画面里占比 ~70%,跟 TRELLIS / FluffyElephant 训练数据分布对齐。

**Trade-off 实测**:单位立方体 corner-on 角度会轻微 clip,100 obj × 40 view 中 **3.15% 触边**,74/100 obj 完全无 clip。可接受 — TRELLIS 主渲染本来就接受这个取舍。

### 1.4 验证通过项(早期 100-obj smoke,2026-05)

- ✅ **A1 offset**: max\|dy\| 22.7%, mean ~7-8%
- ✅ **A2 FoV**: 每 obj 40 帧 spread = 0°(严格 per-obj 常数),跨 obj 覆盖 [25°, 70°]
- ✅ **A3 pitch**: 实测 [-84°, +87°],75% 在上半球(TRELLIS 分布)
- ✅ **R2 radius**: 每 obj 抖动 ±8.6%-9.4%(U(0.95, 1.15) 设计中心 1.05,spread ±10%)
- ✅ **Edge-touch**: 3.15%(126/4000 view),74/100 obj 完全无 clip
- ✅ **Pose convention 5/5**: first view normalize 1.11e-16, rotation 正交 1.86e-15, pose encoding 范围合理, alpha mask 偏心检测, multi-view point cloud 拼合无重影
- ✅ **Bbox extent in normalized world frame**: 0.15-0.57(FluffyElephant 0.20-0.90)

### 1.5 训练侧后续工作 — 已部分完成 (P1-P5 见 §6)

| 项 | 实施位置 | 状态 |
|---|---|---|
| 用 per-frame `fov` 算 focal_length | `dataset_objaverse.py:138` | ✅ P1 done |
| Roll augment | `dataset_objaverse.py:preprocess_frames` | ✅ P2 done |
| `total_frames_per_obj: 40` | 新 config yaml | ✅ P5b done |
| `exclude_bg` 改用 alpha mask | `loss.py:179-184` | ✅ P4 done |
| (可选) view_selector 局部 orbit | `dataset_objaverse.py:view_selector` | TODO — 缓解"4 view 跨度过大"与真实分布的 gap |

---

## 2. 工程优化(全部已落地)

| ID | 优化 | 实现 |
|---|---|---|
| C1 | Persistent BVH between frames | `scene.render.use_persistent_data = True` |
| C2 | OPTIX kernel cache 持久化 | `OPTIX_CACHE_PATH=$HOME/.cache/OptixCache` |
| C3 | 持久 Blender 进程 | orchestrator `--persistent`,1 个 Blender 内循环 |
| C4 | Depth 16-bit PNG | compositor MapRange 节点 (避 OpenEXR 依赖) |
| C5 | Slurm 数组并行 | sbatch `--array=0-N` 切 obj_list |
| C6 | X11 用户态库 | `LD_LIBRARY_PATH=$HOME/tools/x11_libs` (compute 节点缺 libSM) |
| C7 | Blender bundled Python | compute 节点无 system numpy |
| **C8** | **/tmp scratch + 末尾 batch rsync + atomic tar rename** | -17% to -29% 总耗时,防止半个 .tar |
| **C9** | **N=4 procs/GPU × 4 GPU 单 job** | 4-GPU 同节点测出最优 (见 §4.2) |

---

## 3. 关键文件

### Render pipeline (`$HOME/code/render_pipeline/`)
| 文件 | 用途 |
|---|---|
| `render.py` | Orchestrator: 采样 per-obj views (A1/A2/A3/R2), 调 Blender, tar+atomic move |
| `blender_script/render.py` | Blender 端: TRELLIS init + 每帧 lookAt offset + 写 metadata + per-batch flush |
| `utils.py` | TRELLIS `sphere_hammersley_sequence` |
| `sbatch_gpu_multi.sh` | 4-GPU + N=4 procs/GPU production sbatch |
| `sbatch_gpu_array.sh` | 单 GPU array fallback |

### 验证脚本

| 文件 | 用途 |
|---|---|
| `$HOME/tmp/render_bench_100/verify_format.py` | 文件计数 + RGB/depth dtype + transforms.json schema + A2/R2 ranges |
| `$HOME/tmp/render_bench_100/verify_edge_touch.py` | 计算 alpha mask 触画面边缘的 view 比例(clip rate) |
| `$HOME/tmp/render_bench_100/validate_pose.py` | 5 项 pose convention 检查 + 反投影生成 pcd_all.npz |
| `$HOME/tmp/render_bench_100/view_pcds.py` 等 viser 脚本 | 8123/8124/8125 端口可视化点云 |
| `$HOME/code/RnG_feature_allignment/scripts/verify_render44798/` | Dataloader patches L0-L4.5 验证(见 §6.3) |

### Render 命令参考
```bash
# 单 obj (debug)
$BLPY render.py --object mesh.glb --output_dir out/ --num_views 40 \
    --resolution 512 --device GPU --samples 128 \
    --offset_max 0.15 --radius_min_factor 0.95 --radius_max_factor 1.15 \
    --fov_min_deg 25 --fov_max_deg 70 --persistent

# 4-GPU production (全量 44,798)
OBJ_LIST=$HOME/data/objaverse_renders_44798/obj_list_44798.txt \
OUTPUT_DIR=$HOME/data/objaverse_renders_44798/ \
NUM_VIEWS=40 N_GPUS=4 N_PROCS_PER_GPU=4 \
sbatch --time=20:00:00 --cpus-per-task=96 --mem=1000G sbatch_gpu_multi.sh

# 调试可视化(带 RGB + depth overlay)
$BLPY render.py --obj_list debug_5obj.txt --output_dir debug_out/ \
    --num_views 10 --debug_overlay --persistent
```

---

## 4. 关键 benchmark 结论(已固化为 production 配置)

### 4.1 单 GPU multi-Blender 扫描 (job 67853, 100 obj × 10 view, B mode)

| N_PROCS/GPU | 1 | 2 | 4 | **8** |
|---|---|---|---|---|
| Per-GPU vps | 3.03 | 4.57 | 5.62 | **7.21** |

**结论**: 单 GPU 上 N=8 最优 (2.38× over N=1)。

### 4.2 4-GPU 同节点扫描 (job 67882)

| N_PROCS/GPU | 1 | 2 | **4** | 8 |
|---|---|---|---|---|
| 总 VPS | 18.81 | 26.82 | **29.32** | 26.97 |
| Per-GPU VPS | 4.70 | 6.70 | **7.33** | 6.74 |

**结论**: **4-GPU 同节点 N=4 最优**,不是 N=8 (32 procs 把 PCIe/DRAM 抢爆)。Per-GPU vps 7.33 ≈ 单 GPU bench 7.21 → **scaling 101.7% 线性**。

### 4.3 Per-stage profile (job 67817, 5 obj × 10 view, B mode)

| Stage | mean/view | 占比 |
|---|---|---|
| **render** | **0.226 s** | **99.7%** |
| setup, post, meta | < 0.001 s 合计 | 0.3% |

**结论**: render = 99.7% 时间 → 在"无质量风险"约束下基本无可优化空间。剩下的 OPTIX cache (1.5s 节省) / pipeline rsync (~1%) 都是 marginal。

### 4.4 Production 失败教训 + 修复(已落地)

| 现象 | 根因 | 修复 |
|---|---|---|
| Job 67884 跑 11h43m 后被 OOM kill(渲了 13,298/44,798) | 400G mem 不够 16 procs × 大 mesh BVH | sbatch --mem 升到 1000G |
| Job 内存逐步泄漏 | `init_scene()` 没真清孤儿 datablock(`bpy.data.meshes/materials/images/textures` 未 `do_unlink=True; remove()`) | 显式调 `bpy.data.X.remove(item, do_unlink=True)` 清 4 个集合 |
| Trap/timeout 期间损失整 chunk(1400 obj) | 渲完所有 obj 才 tar+move | per-batch flush:每渲 500 obj 立刻 tar+move,scratch 立刻清 |
| Restart 后看见损坏的 `.tar` | tar 当时正在写就被 trap kill | **atomic rename**: 先 `tar -cf <uid>.tar.tmp`,完成才 `mv .tmp → .tar`,中断时 `.tar` 仍是空 |
| 每 job 重新 JIT 编译 OPTIX kernel (1.5s 损失) | `OPTIX_CACHE_PATH` 没设 | sbatch `export OPTIX_CACHE_PATH=$HOME/.cache/OptixCache` |
| **Bad GLB → Blender segfault** | 少数 obj 触发 Cycles 崩溃 | per-batch tar 限损失 ≤ 500 obj。**反复出现的 uid 加 obj_list 黑名单**(整集 ≤ 0.1%) |

---

## 5. Production 完成 — 44,798 Objaverse meshes

**状态**: ✅ **100% (2026-05-20 ~ 05-21)**

| 项 | 值 |
|---|---|
| 源 | `$HOME/.objaverse/hf-objaverse-v1/glbs/000-{000..159}/` (160 子目录, 326 GB) |
| 输出 | `$HOME/data/objaverse_renders_44798/<uid>.tar` × **44,798**, 总 **375 GB** |
| 每 .tar | 40 RGB + 40 depth + transforms.json + phases.json (82 files, ~8 MB) |
| 关键设计 | per-obj .tar 避免 inode 撞限 (3.7M 文件 → 44,798 文件,降 82×) |

**.tar layout** (P5a 自动检测的 prefix 之一):
```
<uid>/000.png       # RGBA 512x512 8-bit
<uid>/000_depth.png # 16-bit grayscale
<uid>/transforms.json
<uid>/phases.json
...
```

`transforms.json` 包含: `aabb / scale / offset / frames[i].{file_path, camera_angle_x, fov, yaw, pitch, radius, lookat_offset, transform_matrix, depth.{min,max}, render_seconds}`

---

## 6. Dataloader patches (RnG_feature_allignment 新 branch)

**branch**: `render44798-dataset-patches` (off `fa3-dtype-match-rng`),**已 commit 全套 P1-P5 + 验证脚本**,**未 merge 到主 branch**。

### 6.1 P1-P5 改动

| # | Patch | 文件 | 状态 |
|---|---|---|---|
| **P1** | per-frame fov (A2 启用) | [`data/dataset_objaverse.py:138`](code/RnG_feature_allignment/data/dataset_objaverse.py) | ✅ commit `0a05707` |
| **P5a** | tar prefix auto-detect (兼容 `./` 和 `<uid>/`) | `data/dataset_objaverse.py:108` | ✅ commit `1e1844e` + fix `1aa1bea` |
| **P4a** | dataset emit alpha_mask | `data/dataset_objaverse.py` | ✅ commit `b3281c6` |
| **P2** | per-view roll augment (R1 启用) | `data/dataset_objaverse.py:preprocess_frames` | ✅ commit `2da6130` |
| **P4b** | `loss.py` 接收 `target_alpha_mask` kwarg | `model/loss.py` | ✅ commit `5f864f5` |
| **P4c** | 模型 forward plumb alpha_mask | `model/vggt/models/RnG.py` + `model/LVSM_obj_decoder_only.py` | ✅ commit `0f7d337` |
| **P5b** | 新 config + obj list | `configs/RnGUP_obj_448_bf16_15k_render44798.yaml`, `data/objaverse_44798.txt` | ✅ commit `e7f9b3d` |

**全部 patches 100% 后向兼容**: roll_augment_max_deg=0 默认关,alpha_mask=None 时 loss 回退白底法,P5a auto-detect 兼容 FluffyElephant_tar (`./` 前缀)。

### 6.2 P2 roll augment 关键设计

每 view 独立采 roll_deg,**image + depth + alpha + c2w 用同一个角度同步旋转**:
- image: PIL rotate BILINEAR, fillcolor=white
- depth: PIL rotate NEAREST, fillcolor=65535 (preserve invalid mask)
- alpha: PIL rotate NEAREST, fillcolor=0
- c2w: 后乘 `Rz_cam(roll_rad)` **在 first-view normalize 之前**

### 6.3 验证结果 (L1 + L3 + L4.5)

**所有 5 个 patch 验收过**:

| Level | 验证内容 | 结果 |
|---|---|---|
| L1.P1 | 7 view 的 focal 来自 transforms.json per-frame fov,非硬编码 40° | ✅ PASS |
| L1.P5a | 44,798 (`<uid>/`) + FluffyElephant_tar (`./`) 都加载成功 | ✅ PASS |
| L1.P4a | alpha_mask shape `[v,1,h,w]`, fg ratio 0.03-0.42 | ✅ PASS |
| L1.P4b | exclude_bg + alpha_mask 的 l2 跟白底 fallback 不同 | ✅ PASS (diff=0.00005) |
| L1.P2 | image/alpha/c2w 三者随 roll 一起改变 | ✅ PASS |
| L3 stats (20 batch) | fov ∈ [25.8°, 68.9°], alpha fg mean 0.143, point_map valid = alpha 一致 | ✅ PASS |
| **L4.5 viser CD** | 见下表 | ✅ PASS |

### 6.4 L4.5 多视图反投影 CD 验证

```
obj       raw_CD   aligned_CD  rotation°
000074a3  0.0032   0.00221     1.8°
0000ecca  0.0110   0.04304     178.6°  (← Kabsch 在近对称物体上找错对称轴,raw CD 已说明结构正确)
00010d96  0.0106   0.00300     10.1°
```

**重要洞察:per-view 独立 roll + first-view normalize 会让 world frame 全局旋转**(不是 bug 是设计):

- dataset 把所有 c2w 表达成"相对第 0 view"的坐标系
- 第 0 view 自己也被 roll → 整个世界 frame 跟着 view 0 的随机 roll 转
- 数学:`world_pt_roll_on = R_world @ world_pt_roll_off`,`R_world = target_first_view @ inv(Rz(roll_view0)) @ inv(target_first_view)`
- **场景结构 100% 保留,训练 loss 不受影响**(模型只看相对位置)

**通过判据**:
- raw CD < 0.015 (物体 size ~1 unit) ✓
- aligned CD < 0.01 (除了 Kabsch 在对称物体上的退化 case)
- rotation_deg ∈ [0°, 10°] (跟 view 0 的随机 roll_deg 一致) ✓

### 6.5 新 config 关键字段(`RnGUP_obj_448_bf16_15k_render44798.yaml`)

```yaml
training:
  dataset_name: data.dataset_objaverse.ObjaverseDataset
  dataset_path: data/objaverse_44798.txt
  root_path: /home/z50057756/data/objaverse_renders_44798
  use_tar: true
  tar_root_path: /home/z50057756/data/objaverse_renders_44798
  total_frames_per_obj: 40     # (FluffyElephant 是 25)
  num_views: 7                 # 4 input + 3 target
  roll_augment_max_deg: 10.0   # P2 启用
exp_name: RnGUP_obj_448_render44798
```

---

## 7. 数据存储

| 路径 | 内容 |
|---|---|
| `$HOME/data/objaverse_renders_44798/<uid>.tar` × 44,798 | **production 数据** (375 GB) |
| `$HOME/tmp/render_bench_100/output_gpu/<uid>/` × 100 | 早期 100-obj 验证(loose files,边触检测 + pose 验证 已跑过) |
| `$HOME/tmp/render_bench_100/output_gpu/edge_touch.json` | 4000 view 触边 per-view border_px |
| `$HOME/tmp/render_bench_100/output_gpu/<uid>/pcd_all.npz` | 反投影 3D 点云(1M 上限) |
| `$HOME/tmp/multi_gpu_scaling_test/` | benchmark sweep 结果 (job 67882) |
| `$HOME/.objaverse/hf-objaverse-v1/glbs/` | 源 GLB (326 GB) |

**Mesh 来源**: Sketchfab UUID 子集,aesthetic_score ≥ 6.0,`random.seed(42)` 采样;来自 `TRELLIS-500K ∩ Step1X-3D-obj-data` overlap (双 SOTA 团队都背书的高质量集)。

### Viser 可视化端口(早期 smoke 用,需要时重启)
| Port | 数据 | 脚本 |
|---|---|---|
| 8123 | 新渲染 100-obj 前 4 个 | `$HOME/tmp/render_bench_100/view_pcds.py` |
| 8124 | FluffyElephant 训练集前 4 个(对比基准) | `view_fluffy_pcds.py` |
| 8125 | 最严重 clip 的 4 个 obj | `view_clipped_pcds.py` |
| 8126 / 8127 | Dataloader L4.5 验证(roll-off vs roll-on) | `verify_l4p5_viser.py`, `verify_l4p5_roll_compare.py` |

---

## 8. 下一阶段路线

| Phase | 数据规模 | 当前状态 |
|---|---|---|
| **Phase 0: 渲染管线 + dataloader** | — | ✅ 完成 |
| **Phase 1a: 全量 44,798 训练 run(15K step)** | 44,798 obj × 40v | 🟢 **本节启动** |
| Phase 1b: 真实测试集 + baseline vs +A1/A2/R1/R2 对比 | — | 需 10-30 真手机多视角 |
| Phase 2: 中等规模(若 1b 显示有 sim2real 提升) | 50,000 obj | 阻塞 1b |
| Phase 3: 全量 | 330,000 obj | 阻塞 Phase 2 |

---

## 8.1 Phase 1a 启动:render44798 全量训练

### Context

Render pipeline + dataloader patches 已经验证完(§4-§6),5 个 sim2real patches 全启用,**现在跑全量 15K step 训练**。

- 数据: 44,798 obj × 40 view tar (`$HOME/data/objaverse_renders_44798/`)
- Config: `configs/RnGUP_obj_448_bf16_15k_render44798.yaml` (§6.5)
- Branch: `render44798-dataset-patches`(未 push,先本地提)
- 资源: 8 H200 同节点(qos=normal, 2 day)
- 预估 wall: 15K step × ~3-5 s/step ≈ **12-21 h**(看 dataloader / 40-view 加载是否成 bottleneck)

### 8.1.1 新 sbatch:`scripts/rngUP448_render44798_8h200.sbatch`

**不改原** `rngUP448Bf16_15k_obj_train_4h200.sbatch`(保留作 FluffyElephant baseline 参照),复制一份新文件,只改 4 处:

```bash
#SBATCH --job-name=rng448-render44798-8h200    # 新名字
# (其它 SBATCH 行 + conda 激活 + env vars 完全沿用原版)

torchrun --nproc_per_node=8 --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29513 \
  train.py --config configs/RnGUP_obj_448_bf16_15k_render44798.yaml
  # 没有 cmdline override — 新 config 里已经写好
  # use_tar=true / tar_root_path / roll_augment_max_deg / num_views=40 / total_frames=40
```

**为什么这么简单**: 我们 P5b 写新 config 时把所有需要的 override 都内嵌进去了(§6.5),sbatch 不需要再 cmdline 覆盖。

### 8.1.2 资源 / QoS

| 项 | 值 | 备注 |
|---|---|---|
| `partition` | `gpu` | 唯一可选 |
| `qos` | `normal` | MaxWall 48h,priority 高于 low,default qos(已经在原 sbatch 里) |
| `gres` | `gpu:h200:8` | 同节点 8 卡 |
| `nodes` / `ntasks` | 1 / 1 | torchrun 内自己起 8 个 rank |
| `cpus-per-task` | 208 | 整节点 CPU,dataloader 多 worker 用得上 |
| `mem` | 1500G | 接近整节点 |
| `time` | 2-00:00:00 | 48h(QoS=normal 上限) |
| `requeue` | 是 | 节点故障自动 resubmit |
| `exclude` | `lrc-alpha-sg-gpu05` | 历史坏节点;实测现在 idle 了但保持排除以防万一 |

### 8.1.3 集群当前状态(2026-06-22 提交前的 snapshot)

可用整节点: **gpu01, 04, 06, 07 完全 idle**(每节点 0 used GPU + 208 free CPU)。gpu05 也 idle 但被 exclude。提交后应该立刻调度上,**不会排队**。

### 8.1.4 提交命令

```bash
cd $HOME/code/RnG_feature_allignment
sbatch scripts/rngUP448_render44798_8h200.sbatch
```

(如果想 override exp_name 区分多次跑,在 torchrun 后加 `exp_name=RnGUP_obj_448_render44798_v2`)

### 8.1.5 监控清单

**前 100 step**(出问题最早会显形):
- ❌ loss/PSNR NaN/Inf → check no recent unintended yaml/code change
- ❌ dataloader stall(GPU util < 50%)→ 提高 num_workers(已 8)/ prefetch_factor(已 4)/ 降 num_views 40→25
- ❌ Step OOM → 降 batch_size_per_gpu 8→6
- ✅ `wandb` 里 lr warmup 起来,loss 下降

**前 1k step**:
- ✅ alpha-mask exclude_bg 真的在用:fg PSNR > full PSNR(说明 P4 path 启用了)
- ✅ per-frame fov 多样性:wandb 看 fxfycxcy 分布
- ✅ roll 真的生效:可视化第 0 view 的 image 在 batch 间有旋转

如果 1k step 一切正常,**直接放任跑完 15K**。

### 8.1.6 回退方案

| 触发条件 | 应对 |
|---|---|
| dataloader 是 bottleneck | 改 `num_views: 40 → 25`(显存省 60%, target 候选少但仍 25 个,可接受);bump `prefetch_factor: 4 → 8` |
| 显存 OOM | `batch_size_per_gpu: 8 → 6`,或 `grad_accum_steps: 1 → 2` |
| 长 loss 反常(明显高于 FluffyElephant baseline) | A. 关 roll (`roll_augment_max_deg: 0`) 排除 P2 副作用<br>B. 关 alpha-mask exclude_bg(loss 改用 cmdline force `target_alpha_mask` 不传) |
| 节点故障/抢占 | `requeue` 自动重 submit;`reset_training_state=false`(默认)继续从 ckpt 接续 |

### 8.1.7 输出位置

- Checkpoints: `./experiments/checkpoints/RnGUP_obj_448_render44798/ckpt_*.pt`(每 2000 step)
- Slurm logs: `./slurm_logs/rng448-render44798-8h200-<JobID>.{out,err}`
- Wandb: project `LVSM_obj`,run name `RnGUP_obj_448_render44798`

### 8.1.8 提交前 git 整理(强烈建议)

当前 branch `render44798-dataset-patches` 在本地,共 **15 commits**(P1-P5 + verify scripts + fixes)。**还没 push remote**。

```bash
cd $HOME/code/RnG_feature_allignment
git push -u origin render44798-dataset-patches
```

这样:
- 训练用的代码状态有 remote 备份
- 节点故障 requeue 后还能找到 commit
- 实验 wandb run 里可以记 commit hash,便于复现

### 8.1.9 不在 Phase 1a scope

- Phase 1b 的真实手机测试集采集(可以并行)
- baseline FluffyElephant 对照 run(如果想训完后做对比,可以在 1a 跑完 / 跑一半时再起一个)
- 任何 model 或 loss 的算法 ablation

---

## 8.2 81043 Iter time 诊断 + 修复(2026-06-22 实测)

### 8.2.1 现象

Job 81043 在 8 H200 gpu01 上跑,前 50 step **iter time = 18-24s**,LR warmup 期 step 1 = 46s (worker 启动),稳定后 22-24s。

| run | num_views | roll | iter time | 倍数 |
|---|---|---|---|---|
| baseline `rng448-sft-repal7-8h200-80665`(FluffyElephant) | 25 | 关 | **7.14s, 7.69s** | 1.0× |
| 本次 81043(render44798) | **40** | **±10°** | **18-24s** | **3.2×** |

**外推**: 24s × 15K = 100h,**超 48h time limit ≈ 2× → 会 TIMEOUT 在 ~7K step**。

### 8.2.2 根因

dataloader bottleneck。`preprocess_frames` 在 `for v_idx in range(num_views)` 循环里**对每一个 view 都完整跑完**: image PIL composite + resize + roll(BILINEAR)+ alpha capture/resize/roll(NEAREST × 2)+ depth load/resize/roll/normalize + per-view point_map 反投影。

但 `ProcessData.fetch_views` 后续**只取出 4 input + 3 target = 7 views** 给模型 forward。**剩下 33 views 的所有工作完全浪费**。

成本拆解(每 view):
- image PIL.rotate BILINEAR 512×512 ≈ 20-30 ms
- alpha PIL.rotate NEAREST 512×512 ≈ 5-10 ms
- depth PIL.rotate NEAREST 512×512 ≈ 5-10 ms
- 其它(composite/resize/intrinsic/c2w/point_map)≈ 50 ms
- **per view ≈ 80-100 ms**

40 view × 80-100ms = **3-4 s / sample** → batch 8 × 8 GPU = 64 samples → 即使 8 worker × 8 GPU = 64 个 worker 完全并行,也接近 4s 纯 dataloader。加上 forward/backward 3-4s + JuiceFS metadata 抖动 → 20-24s 解释得通。

40/25 = 1.6× 的 num_views 增长解释 1.6× 慢;**剩下 2× 来自 P2 roll(per-view 多 3 个 PIL.rotate)+ P4a alpha capture**。

### 8.2.3 修复方案 — **`num_views: 40 → 7`**

**核心思路**(用户提出): num_views = `num_input_views + num_target_views = 4 + 3 = 7`。`view_selector` 每次从 **`total_frames_per_obj=40`** 里随机采 7 个,`preprocess_frames` **只 load + augment 那 7 个**。跨多 epoch 同 obj 被采到不同的 7 view 子集,target distribution 经过多个 epoch **覆盖全部 40 view**。

```yaml
total_frames_per_obj: 40   # 数据有 40 view
num_views: 7               # 每个 __getitem__ 从 40 里随机抽 7,只 load 这 7 个
num_input_views: 4
num_target_views: 3
```

**实现已就绪,不需要改代码**:
- `view_selector` (line 51-56) 实现: `random.sample(range(0, total_frames_per_obj=40), num_views=7)` ✓
- `preprocess_frames` (line 132-) 循环 `for v_idx, img_idx in enumerate(image_indices)` 只跑 7 次 ✓
- `ProcessData.fetch_views` (data_utils.py:110-146) input=前 4, target=随机 3 from {0..6} ✓
- 每次 same obj re-enter `__getitem__`,`random.sample` 重新抽 7 → 跨 epoch 多样性自然来自抽样 ✓

**对比 num_views=40**:

| 项 | num_views=40 (现在) | num_views=7 (修复) |
|---|---|---|
| 每 step dataloader work / obj | 40 view 全做 | **7 view 只做** ← 5.7× 少 |
| 单 epoch target view 覆盖 | 一个 obj 一次见全 40 候选 | 一个 obj 一次见 7 候选 |
| **跨 epoch (~14 epoch in 15K step) target 覆盖** | 完全 | **完全** (累计抽到全 40) |
| 预估 iter time | 24s | **~7s**(跟 baseline 同等 dataloader work) |
| 预估 wall(15K) | 100h ❌ | **29h** ✓ 远在 48h 内 |

**为什么不是中间值(14 / 25)**:
- 单次 batch 给 14 个候选 target 跟给 7 个候选 target,**模型梯度信号一样有效**(都是 random sample 几个)
- 跨 epoch 看,7-view-shuffle 7 epoch 就能覆盖 40 view 期望
- 更小 num_views = 更少 dataloader 工作 = 更快 = 更早完成训练
- 没必要为单 batch 多样性付 dataloader 浪费

### 8.2.4 我之前 num_views=7 → 40 "fix" 反而错了

回顾我在 commit `f263858 fix(P5b): num_views=40` 里说 num_views=7 是 bug 是**误解**:
- 那时我担心 target 只在 7 个里采,多样性差
- **忽略了**: view_selector 每次 random sample,跨 epoch 多样性自然来自抽样
- **修复**: 改回 num_views=7,代码无需动,只 config 一行

### 8.2.5 P6 patch — `target_has_input_prob` 概率混合

**用户需求**:90% 概率 disjoint 4+3 split,10% 概率允许 target 跟 input overlap。

**调研发现**:
- `utils/data_utils.py:114-129` 现在用 bool `target_has_input` 二选一:
  - `True`: target = `random.sample(range(num_views), num_target_views)` 可重叠
  - `False`: target = `[num_views-1, num_views-2, ..., num_views-num_target_views]` 排序后 = last num_target_views,跟 input(first num_input_views)严格 disjoint
- `view_selector` (`random.sample(range(40), 7)`) 返回 unsorted 7 indices → input 是 random 4 of 40 (前 4 of 7), target_has_input=false 时 target 是另外 random 3 of 40 (后 3 of 7) → 完美 disjoint 且都是 random of 40

**改动**:`fetch_views` 加 per-sample 概率分支。

```python
# utils/data_utils.py:fetch_views — 在 line 113 之前加
overlap_prob = self.config.training.get("target_has_input_prob", None)
if overlap_prob is None:
    # 后向兼容: bool target_has_input → {0.0, 1.0}
    overlap_prob = 1.0 if target_has_input else 0.0

# 替换 line 114-129 二选一的逻辑为 per-item 概率
target_indices = []
for _ in range(bs):
    if random.random() < overlap_prob:
        # target_has_input=true 路径: random sample, 可重叠
        target_indices.append(random.sample(range(num_views), num_target_views))
    else:
        # target_has_input=false 路径: 最后 num_target_views, 跟 input disjoint
        target_indices.append(sorted([num_views - 1 - j for j in range(num_target_views)]))
index = torch.tensor(target_indices, dtype=torch.long, device=data_batch["image"].device)
```

**后向兼容**:
- 没设 `target_has_input_prob` → 继续读 `target_has_input` bool,行为完全跟现在一样
- 老 config (`RnGUP_obj_448_bf16_15k.yaml`, val_dataset_cfgs 等)不需要改
- 新 knob 只在我们 render44798 config 启用

**Val 完全不受影响**: `process_val_data` 用 `config.training.val_dataset_cfgs` 子树作为它的 config,我们在外层 `config.training.target_has_input_prob` 加新 knob → val 子树读不到 → 走原 bool false 路径 → 行为不变。

### 8.2.6 新 config 改动(`RnGUP_obj_448_bf16_15k_render44798.yaml`)

```yaml
training:
  total_frames_per_obj: 40
  num_views: 7                      # 改:40 → 7 (从 40 里随机抽 7,只 augment 这 7)
  num_input_views: 4
  num_target_views: 3
  target_has_input: true            # 保留作后向兼容 fallback(prob 未设时使用)
  target_has_input_prob: 0.1        # 新加:10% 概率 overlap,90% 概率 disjoint
```

### 8.2.7 执行步骤

1. **`scancel 81043`**(避免继续 wasted compute)
2. **Edit `utils/data_utils.py`**: 加 per-sample 概率分支(P6)
3. **Edit `configs/RnGUP_obj_448_bf16_15k_render44798.yaml`**: `num_views: 40 → 7` + 加 `target_has_input_prob: 0.1`
4. **Commit** 到 branch (2 commits: P6 code + config) + **push**
5. **重新 sbatch** `scripts/rngUP448_render44798_8h200.sbatch`(不需要改)
6. **验证 first 10 step iter time** ~7s,以及 wandb 看 90/10 target 分布是否对

### 8.2.8 预期 wall

- num_views=7 + P6 (本次修复): iter ~7s × 15K = **29h** → 远在 48h 内 ✓ ✓
- num_views=40 (现 81043): iter ~24s × 15K = 100h → ❌ TIMEOUT

### 8.2.9 长期 TODO

- **lazy augment**: P2 roll + P4a alpha + point_map 移到 ProcessData/model 内,只 augment 实际用的 7 view → 即使未来 want 更大 num_views 也不付代价
- 现在不做,P6 + num_views=7 已达到 baseline 同等 dataloader 工作量

---

## 8.3 81058 iter time 仍然 3× 慢(2026-06-22 二诊)— P2 roll 嫌疑最大

### 8.3.1 真 baseline 对比

之前对错了 baseline。**正确 baseline 是 `FA3_repa_dinov3_l7` 这次 pretrain run,log `rng448-8h200-fa3-tar-72209.out`**:

| run | data | roll | alpha (P4a) | iter median | p95 |
|---|---|---|---|---|---|
| **FA3_repa_dinov3_l7 (72209)** baseline | FluffyElephant_tar | **OFF**(P2 还没存在) | **OFF**(P4a 还没存在) | **6.87s** | 7.08s |
| 81058 (now) | render44798 | ON ±10° | ON | 16-22s | — |

**完全相同的**:model class (VGGT4LVSM)、feature_alignment.layer_idx=7 + DINOv3 target encoder、batch_size=8、lr/optim/loss weights/grad clip。

**唯一新增的**:P1-P5 dataloader patches。

### 8.3.2 GPU util 100% 但仍然慢 = dataloader 在偶尔阻塞 GPU

`srun --jobid=81058 nvidia-smi`: GPU 100% util,memory 73 GB/141 GB。

如果纯 compute bound,iter 应该跟 baseline 同(6.87s)。慢 3× 必然有 dataloader **间歇性卡 GPU**(就算瞬时 util 100%, 总体也被拖)。

### 8.3.3 嫌疑排序(按可能性)

| # | 嫌疑 | 量化估算 | 验证方法 |
|---|---|---|---|
| **A** | **P2 roll augment 太慢** | 3 PIL.rotate × ~25ms × 7 view = ~525ms / obj 额外 CPU 工作。Python GIL + PIL not multi-thread 友好,workers 真实串行 → 64 个 worker 跟不上 | **关 `roll_augment_max_deg=0` 重测**,iter time → 7s 就 confirm |
| B | P4a alpha 提取 | ~5ms × 7 = ~35ms 工作 + 多 [b,v,1,h,w] tensor 通过 ProcessData.fetch_views 的 torch.gather | 显著小于 A,但叠加可能放大 |
| C | render44798 tar 比 FluffyElephant_tar 更慢加载 | 不太可能 — 都是 JuiceFS,文件结构类似 | 用 FluffyElephant 跑 21 步看 iter time |
| D | P1/P5a 自检 + per-frame fov 读取 | 几个 us,不可能 | — |

**Strongest hypothesis: A (P2 roll)**。3 个 PIL.rotate per view 是新增的最大单项 CPU 工作。

### 8.3.4 诊断步骤(二分定位,不预判修复)

**目标**:定位真正拖慢的是 P2 roll、P4a alpha、render44798 数据访问、还是别的。**修复方案等诊断完再讨论**。

每个 Test 跑前 ~20 step 看 iter time,~10 min。

#### Test 1 — 排除 P2 roll
- scancel 81058
- config 临时改 `roll_augment_max_deg: 10.0 → 0.0`(只磁盘,不 commit;诊断完恢复)
- 重 sbatch
- 期望:
    - **iter ≈ 7s** → P2 roll 是主因,**停诊断,跟用户讨论修复**
    - **iter 仍 15-20s** → 走 Test 2

#### Test 2 — 排除 P4a alpha emission(在 Test 1 关 roll 的基础上叠加)
- scancel
- 临时在 `dataset_objaverse.py` 把 `"alpha_mask": alpha_masks` 注释掉(同时 __getitem__ 不 stack alpha_masks)
- 模型 fallback 走 white-bg(P4b 的 `target_alpha_mask=None` 路径)
- 重 sbatch
- 期望:
    - **iter ≈ 7s** → P4a alpha 是主因(roll 也有贡献,看 Test 1 多大改善)
    - **iter 仍 15-20s** → 走 Test 3

#### Test 3 — 排除 render44798 数据本身
- scancel
- 临时改 config 用 FluffyElephant 数据:
  ```yaml
  dataset_path: data/objaverse_v1_in_lvis_25v.txt
  tar_root_path: /home/z50057756/FluffyElephant_tar
  total_frames_per_obj: 25
  num_views: 7
  ```
- (P1-P5 patches 仍在,但 roll/alpha 已 Test 1/2 关闭)
- 重 sbatch
- 期望:
    - **iter ≈ 7s** → 我们 render44798 数据访问慢,JuiceFS / 8MB 大 tar / 16-bit depth 解析等需排查
    - **iter 仍 15-20s** → 患在我们 P1/P5a 代码本身的某处(per-frame fov 查 dict、prefix scan 等),或别的没想到的地方

### 8.3.5 诊断结果 — P4a 是单一 culprit

| Test | 配置 | iter median | Δ vs baseline 6.87s |
|---|---|---|---|
| baseline FA3 (72209) | FluffyElephant, no P1-P5 | 6.87s | 1.00× |
| 0 (81058) | render44798, P2 ON + P4a ON | 16-22s | 2.5×~3.2× ❌ |
| 1 (81063) | render44798, **P2 OFF**, P4a ON | **22.81s** | **3.3×** ❌ |
| **2 (81073)** | render44798, P2 OFF + **P4a OFF** | **6.40s** ✓ | **0.93× = 已对齐 baseline** |

- Test 1 → Test 2 删了 ~16s/iter, 而 Test 1 关 roll 完全无效 → **P2 roll 不影响,P4a alpha emission 是单一 16s slowdown 源**
- Test 3 不用做了(P4a 已定位)

### 8.3.6 根因 + 修复方案 (用户选定: 单次 PNG decode)

**根因**: `data/dataset_objaverse.py:152` 的 P4a 实现里:
```python
image_rgba = Image.open(...)
alpha_raw = np.array(image_rgba)[:, :, 3]   # ← 触发 PNG decode 1
image = Image.alpha_composite(white_bg, image_rgba).convert("RGB")  # ← 触发 PIL re-traverse (decode 2)
```

PIL 的 `alpha_composite` 在 numpy 强制 decode 后**没有完全复用 cached pixel buffer**,造成对 512×512 RGBA PNG 的**等效双遍历**(每遍 ~30-50ms × 7 view × 64 obj/step = 16s)。

**修复 (已在 disk 上,未 commit)**: 单次 `np.array(image_rgba)` decode,RGB 跟 alpha 共享同一份 numpy buffer,白底 composite **在 numpy 里向量化**做(替代 PIL alpha_composite):
```python
image_rgba = Image.open(...)
rgba_arr = np.array(image_rgba, copy=False)        # ← decode ONCE
alpha_raw = rgba_arr[..., 3]                       # view, no copy
alpha_f = (alpha_raw.astype(np.float32) / 255.0)[..., None]
rgb_arr = (rgba_arr[..., :3].astype(np.float32) * alpha_f
           + 255.0 * (1.0 - alpha_f)).astype(np.uint8)
image = Image.fromarray(rgb_arr)                   # ← PIL with already-decoded pixels
```

**功能保留 100%**: alpha_mask 仍照常 emit,P4b/P4c 链路不变,loss 仍走 alpha-mask 路径。smoke 加载验证 image shape [7,3,448,448], alpha_mask [7,1,448,448] fg ratio 0.073 正常。

### 8.3.7 执行步骤

1. **数据等价性验证**(优化版 vs 原版逐像素对比):
   - 从 1 个 obj 取 5 帧 PNG
   - 跑老路径(`np.array(image_rgba)[:,:,3]` + `PIL.alpha_composite`)拿 alpha_raw, rgb
   - 跑新路径(单 `np.array` + numpy composite)拿 alpha_raw_new, rgb_new
   - 断言: `alpha_raw == alpha_raw_new`(完全相等)
   - 断言: `|rgb - rgb_new| ≤ 1` per pixel(uint8 浮点舍入容差)
   - 通过才进 Test 4

2. **Test 4**: 用 P4a 优化版 + roll=10(完整 P1-P5)重新 sbatch,**验证 first 20 step iter time ~7s**(对齐 baseline 6.87s)

3. 通过 → **commit + push** P4a 优化到 branch(数据等价 + 速度对齐 才 commit)

4. **重新提全量训练** sbatch(同一份脚本,无需改)

5. **监控 first 1k step** loss 趋势、alpha mask 启用、roll augment 生效

### 8.3.8 预期 wall (修后)

- iter ~7s × 15K = **29h** → 远在 48h time limit 内 ✓
- 全 sim2real patches (A1/A2/A3/A4/R1/R2) 全启用

### 8.3.9 长期 TODO (不做)

- 进一步把 RGB composite 移到 GPU (torchvision) — 已经 numpy 向量化够快了,没必要
- view_selector 局部 orbit — 跟 dataloader 速度无关,留 phase 1b 之后

---

## 8.4 Test 4 失败 — P4a 单次 decode 优化无效,改走二次 bisect(2026-06-22 三诊)

### 8.4.1 Test 4 实测结果(不是修复)

| Test | 配置 | iter median | 备注 |
|---|---|---|---|
| baseline FA3 (72209) | FluffyElephant, no alpha | **6.87s** | 标杆 |
| 1 (81063) | render44798, roll OFF, **原版 PIL alpha_composite + alpha emit** | 22.81s | 慢 3.3× |
| **4 (81079)** | render44798, roll ON, **优化的单 np.array decode + alpha emit** | **22.40s** | **慢 3.3× — 优化无效** |
| 2 (81073) | render44798, roll OFF, **alpha 完全不 emit** | 6.40s | 对齐 baseline |

**数据等价性已验证**:Test 4 优化代码 vs Test 1 原版代码,alpha 完全相等(diff=0),RGB max diff ≤ 1(uint8 舍入容差,mean 0.002)。**功能等价 ✓,速度无差 ✗**。

### 8.4.2 §8.3 根因分析推翻

§8.3.6 假设"PIL alpha_composite 触发等效双遍 PNG decode"被 Test 4 证伪 — 单次 np.array decode + numpy 向量化跟 PIL alpha_composite 同样慢。

**真实结论**:16s/iter 的开销跟 `alpha = rgba_arr[..., 3]` + `Image.alpha_composite` 实现细节**无关**。开销来自 `alpha_mask` tensor 出现在 `__getitem__` 返回 dict 这件事**本身**触发的下游链路。

### 8.4.3 三 Explore agent 报告综合

| Agent | 发现 | 结论 |
|---|---|---|
| **A1** dataset 代码 | 每 view alpha 加工:`PIL.fromarray + resize NEAREST + np.array + threshold + torch.from_numpy + unsqueeze`,**~5-10ms/view**,7 view ~50ms/sample,64 worker 并行应该没压力 | 单纯 alpha 计算 cost 太小,理论不能解释 16s |
| **A2** loss/model/data_utils | loss alpha-path **反而比 white-bg fallback 少 2 ops**(threshold+expand vs broadcast+sum+threshold+expand);ProcessData.fetch_views 多一个 `torch.gather` (~16.8 MB GPU copy);model/RnG.py 只做 attr 查找无新计算 | loss 理论更快;gather 是真实 GPU 工作但毫秒级 |
| **A3** git history | 只有 alpha_mask 一个新 key;P1/P2/P5a/P5b/P6 都不改 dataloader 输出 shape;loss alpha 路径只在 `exclude_bg=True AND target_alpha_mask is not None` 双条件下触发 | 唯一变量就是 alpha_mask,缩小到三段:dataset emit / IPC+pin / loss path |

理论分析跟实测**严重对不上**。每段单看都不该慢 16s,加起来也不该 16s。说明某处有**未被建模的成本**(GIL 串行化 / numpy 临时 buffer 频繁 alloc/free / dtype mismatch / pin_memory 隐性 sync / kernel slow path 等)。

### 8.4.4 三 hypothesis × 三 bisect 测试设计

#### Hypothesis B1 — alpha_mask IPC/pin_memory size
- alpha_mask float32 `[7, 1, 448, 448]` = **5.6 MB/sample**, batch 8 × 8 worker = 360 MB/worker IPC/step
- pin_memory copies CPU→pinned-CPU + async H2D
- 改 bool/uint8 → 4-8× 缩 IPC

#### Hypothesis B2 — ProcessData.fetch_views 的 gather/slice 链
- 多处理一个 [b, v, 1, h, w] key → 多 1 个 `torch.gather` (16.8 MB GPU copy) + 1 个 slice
- 单看是 ms 级,叠加其他键的相同 pattern,GPU 可能出现 sync 等待

#### Hypothesis B3 — loss.py alpha-path 在实际 GPU 上比 white-bg 慢
- Agent 是静态分析,真实 boolean indexing 在 H200 上未必更快
- boolean mask gather 在某些 size 上 trigger 慢 kernel path

#### Bisect Test 设计(独立 3 个)

| Test | 改动 | 期望解读 |
|---|---|---|
| **A — dummy alpha** | dataset 里 alpha_masks 那 ~5 行 PIL 加工替换成 `alpha_masks.append(torch.zeros((1, resize_h, resize_w), dtype=torch.float32))`,dict 仍 emit `alpha_mask` 形状/dtype 不变 | A=7s ⟹ alpha **计算本身**慢(PIL 链)<br>A=22s ⟹ alpha **存在**触发下游(IPC/loss) |
| **B — disable loss path** | model/vggt/models/RnG.py 强制 `target_alpha_mask=None`,让 P4b 走 white-bg fallback;dataset 仍 emit 真 alpha | B=7s ⟹ **loss path** 是瓶颈<br>B=22s ⟹ **pre-loss**(IPC/ProcessData/model.forward)是瓶颈 |
| **C — bool dtype** | dataset emit alpha_mask 改 `torch.bool` `[v, 1, h, w]` (32× 缩小),loss.py 把 `> 0.5` 改成直接用 mask | C=7s ⟹ **IPC size** 主因<br>C=15s ⟹ IPC 有部分贡献<br>C=22s ⟹ 跟 size 无关 |

**每个 Test wall**:~10 min(等节点 + 跑 20 step)。3 个 Test 全跑完 ~30 min wall + 一些排队时间。

### 8.4.5 决策树 → 修复方向

| Test 结果 | 真因 | 修复方案 |
|---|---|---|
| A=7s, B=any, C=any | PIL resize/threshold 链慢(agent 静态分析错) | alpha 加工提前到 numpy(取消 PIL roundtrip),resize 用 cv2.resize 或 numpy stride 下采样 |
| A=22s, B=7s, C=any | loss.py alpha path | 改 loss path 算法(避免 boolean indexing,用 `mask * (rendering - target)` 纯算术替代) |
| A=22s, B=22s, C=7s | IPC/pin_memory size | alpha_mask 用 bool/uint8 dtype(立即生效,改动最小) |
| A=22s, B=22s, C=22s | ProcessData.fetch_views 多 gather 是问题 | fetch_views 跳过 alpha_mask 的 gather,直接在模型里按 input/target index 现算 |
| A=22s, B=22s, C=15s | IPC 是**部分**因,downstream 也是 | bool dtype + 配合 lazy gather |

### 8.4.6 执行顺序(顺序跑,不并发)+ 临时探针纪律

**核心原则**: §8.4 的所有 Test A/B/C 改动都是**临时诊断探针**,不 commit、不 push。每个 Test 跑完(无论结果)立刻 revert,确保下一个 Test 跟之前 Test 不交叉污染。诊断结论得到后,再单独写一次**正式修复 commit**(只该 commit 修复方案需要的最小改动),临时探针痕迹 0 入库。

**纪律保障**:
1. 在动手第一行 diagnostic edit 前,先 `git stash --include-untracked --message "WIP: pre-bisect baseline (clean state)"` 把当前未 commit 的 P4a numpy 优化也存起来(避免跟探针混)。
2. 每跑一个 Test:`git diff` 留指纹(贴 plan 里),sbatch 跑,scancel,**当场 `git checkout -- <touched files>`** 还原。
3. 三个 Test 全跑完,`git status` 必须是 clean(stash 还在)。
4. 拿到结论后再单独决定:从 `git stash pop` 拿回原优化基线 + 加新修复 → 这一笔才是 commit。

**Phase 1 (~15 min) — Test A**(改动最少):
- Edit `data/dataset_objaverse.py`: 把 alpha 加工那 5 行(`PIL.fromarray + resize + (rotate) + np.array > 127 + to_tensor`)替换为 `alpha_masks.append(torch.zeros((1, resize_h, resize_w), dtype=torch.float32))`
- sbatch + 等 20 step,记 iter median
- `scancel` + `git checkout -- data/dataset_objaverse.py`

**Phase 2 (~15 min) — 走 C 或 B**:
- A=7s → 直接走 §8.4.5 修复方案,**不**跑 B/C
- A=22s → Test C
  - Edit `data/dataset_objaverse.py`: alpha 加工最后 `.astype(np.float32)` 改成 `(np.array(alpha_resized_pil) > 127)` 直接 bool,`torch.from_numpy(...).bool().unsqueeze(0)`
  - Edit `model/loss.py`: alpha-mask 分支的 `target_alpha_mask > 0.5` 改成直接 `target_alpha_mask` 当 bool mask
  - 跑 + 记 + revert
- A=22s 且 C=22s → Test B
  - Edit `model/vggt/models/RnG.py`: `target_alpha_mask=getattr(target, 'alpha_mask', None)` 改成 `target_alpha_mask=None`
  - 跑 + 记 + revert

### 8.4.7 注意事项

- baseline 6.87s 是 FluffyElephant + DINOv3 + REPA layer_idx=7,跟当前 render44798 配置(除 alpha 外)完全等价 — 这个对比标准是干净的。
- 修复后必须复跑一次"Test 4 等价物"(完整 P1-P5,alpha emit 真开,roll=10):iter 真的 ~7s 才算修好。
- bool dtype Test C 改完 loss.py 后,**一定要 unit test alpha-mask 路径**仍然产出对的 mask([b*v, 3, h, w] bool with foreground True),不然 loss 算错。bool→loss 的语义对齐放进 Test C 的写代码 sanity check 里,不是 fast/slow 的二分本身。
- **诊断结束清理 checklist**:
  1. `git status` clean
  2. `git stash list` 看到诊断前 stash 还在,pop 回来恢复 P4a numpy 优化基线
  3. 在干净基线上 apply 正式修复(只改修复必须的那几行)
  4. 跑最终验证 sbatch:iter ~7s + alpha_mask emit 真启用
  5. 单独 commit + push 正式修复

### 8.4.8 时间预算

- Bisect 总 wall:~45 min(3 Test + 排队)
- 修复实施 + 验证:~30 min(改 1-2 文件 + sbatch + 等 20 step)
- 重新提全量 15K 训练:**~29h 内完成**(iter 7s × 15K)

### 8.4.9 Bisect 实测结果 + 根因(2026-06-22)

| Test | 配置 | iter median | 备注 |
|---|---|---|---|
| baseline FA3 (72209) | FluffyElephant, no alpha | 6.87s | 标杆 |
| Test 1 (81063) | render44798, real alpha emit, loss alpha-path ON | 22.81s ❌ | 慢 3.3× |
| Test 4 (81079) | render44798, optimized np.array decode, alpha emit, loss alpha-path ON | 22.40s ❌ | 优化无效 |
| **Test A (81218)** | render44798, **dummy zero alpha**(dataset 不计算),dict 仍 emit | **6.34s ✓** | 排除 IPC/ProcessData/dataset PIL |
| **Test B (81223)** | render44798, **real alpha emit**,但 RnG.py 强制 `target_alpha_mask=None` | **6.36s ✓** | 排除 dataset/IPC/ProcessData,孤立 loss path |
| Test 2 (81073) | render44798, P4a 完全不 emit | 6.40s | 基准对比 |

**结论**:Test A 和 Test B 两个都把 iter 拉回 ~6.35s,但配置完全不同:
- Test A:dataset 不算 alpha,loss 仍跑 alpha-path(空 mask → empty boolean indexing)
- Test B:dataset 全算 alpha 真值,loss 跳过 alpha-path 走 white-bg fallback

唯一两者共有的"快"现象:**loss boolean indexing 在 mask 是 trivial(空 / contiguous-repeat) 时快**。

读 `model/loss.py:188-201` 发现具体根因:

```python
# white-bg fallback (line 198-199, 快):
mask = torch.sum((target - white_bg).abs(), dim=1, keepdim=True) > 0
mask = repeat(mask, 'b 1 h w -> b 3 h w')  # ← einops repeat 产生 CONTIGUOUS [b*v, 3, h, w]

# P4a alpha-mask path (line 191, 慢):
mask = (target_alpha_mask > 0.5).expand(-1, 3, -1, -1)  # ← .expand() 是 stride-0 VIEW,非连续
```

后续 line 201 `rendering[mask]` boolean indexing:**PyTorch 在 stride-0 expand 出来的 non-contiguous bool mask 上走 slow path**(每次都要先 materialize 38 MB 连续 buffer,GPU 内 16s 额外开销;contiguous repeat 直接走 fast kernel)。

### 8.4.10 修复 (P4d patch): loss.py expand → repeat

**一行改动** in `model/loss.py:191`:

```python
# Before (慢):
mask = (target_alpha_mask > 0.5).expand(-1, 3, -1, -1)

# After (快):
mask = repeat(target_alpha_mask > 0.5, 'b 1 h w -> b 3 h w')
```

**为何 work**:`einops.repeat` 等价于 `.expand(...).contiguous()` 或 `.repeat(1, 3, 1, 1)`,产生一个真正的连续 [b*v, 3, h, w] bool tensor。PyTorch 的 boolean indexing 在连续 mask 上有 SIMD/CUDA fast kernel。

**数据等价**:`repeat` 产生的 mask 跟 `.expand` 在数值上 100% 一样,只是 stride 不同。loss 数值完全相同。

**为什么 baseline FA3 不慢**:baseline 没经过 alpha-mask path(`target_alpha_mask` 是 None),全部走 white-bg fallback。fallback 一开始就用 `repeat`,所以 baseline 不踩这个坑。

**为什么以前 dev 没发现**:P4a/P4b 在 L1 unit test 时只对比了 loss 数值差异(`diff=0.00005` < 容差),没对比 iter time。

### 8.4.11 执行步骤(终于正式修)

1. `git stash pop` 恢复 P4a numpy 优化基线(uncommitted 状态)
2. Edit `model/loss.py:191` 的 `.expand` → `repeat`
3. 全量 sbatch(完整 P1-P5,roll=10,alpha emit ON,loss alpha-path ON 但走 contiguous mask)
4. 等 20 step,**验证 iter median 真的 ~7s**
5. 通过 → 单独 commit + push:
   - 一笔 commit:`fix(P4d): loss alpha-mask .expand() → einops.repeat() to use fast boolean indexing kernel (16s/iter → ~0s overhead)`
   - 另一笔 commit:`refactor(P4a): single np.array PNG decode + numpy white composite (data-equivalent to PIL alpha_composite within uint8 rounding)`
6. 提全量 15K 训练 sbatch
7. 监控 first 1k step:loss/PSNR 趋势 + 确认 fg PSNR > full PSNR(alpha mask 真在用)

### 8.4.12 实际执行 + 提交记录(2026-06-22 完成)

- ✅ 跑 Test A: dataset 替换 alpha 加工成 `torch.zeros((1, resize_h, resize_w))` → **iter median 6.34s**(steps 2-21,grep 之前 line-wrap 问题已修)
- ✅ 跑 Test B: `target_alpha_mask=None` 在 `model/vggt/models/RnG.py:524` 强制传入 → **iter median 6.36s**
- ✅ Test A + Test B 两端 6.35s,**唯一共同因素 = loss 不走非平凡 alpha boolean indexing** → 根因锁定在 `model/loss.py` 的 alpha-path 用 `.expand()`,非 contiguous
- ✅ 改 `model/loss.py:191` (P4d) → 跑最终验证 job 81227:**iter median 6.50s** ≈ baseline,0 NaN,l2_loss 数值正常
- ✅ 已 push remote `render44798-dataset-patches` (本地 + remote in sync):
  - `1e2bc7f refactor(P4a): single np.array PNG decode + numpy white composite`
  - `fc3fa06 fix(P4d): loss alpha-mask .expand() -> einops.repeat() to use fast boolean indexing kernel`

---

## 8.5 三个 follow-up 问题:从根因到机制(2026-06-22)

用户提的三个问题,都先做根因分析、再设计**可机制化的**修复(不打临时补丁)。

### 8.5.1 三问题确认 + 表层判断

| 问题 | 表象 | 根因 | 是否独立 bug |
|---|---|---|---|
| Issue 1 | 669 obj 渲染近全空 alpha+depth | **render pipeline 没有 degenerate-mesh / empty-output 校验** | ✅ 独立 |
| Issue 2 | `loss.py:207` 空 mask 时 `(rendering[mask])**2.mean()` = NaN | 数学 corner case;**training 阶段 issue 1 的下游症状** | △ 80% 是 issue 1 的症状,20% 是独立 |
| Issue 3 | `render_pipeline/render.py:7` 顶层 doc 漂移(说 "pitch [-15°, 80°]" 但其实 TRELLIS untouched) | 注释没跟代码同步 | ✅ 独立(纯文档) |

### 8.5.2 Issue 1 根因(Explore agent 实测 6 个样本 UID)

**实测 6 个**:`all_empty_or_near` 3 个(`0153947acdd3...`, `018e6b7e03b3...`, `030da2b778...`)+ `mostly_low` 3 个。每个的发现:

- GLB **都存在**,18-183 MB,文件完整(不是坏文件)
- phases.json 显示 **Blender 跑完了**,render_seconds 正常(0.13-0.22s/帧)
- transforms.json 里的 `scale` / `offset` **严重异常**:
  - `0153947acdd3...`: scale=0.190, offset=[-3.29, 0.94, 2.54]
  - `030da2b778...`: scale=0.271, offset=[16.81, 20.48, 0.50] ← 物体被 normalize 推到离原点 20 米开外
- 所有帧 PNG alpha=0,depth=255(全背景)

**真因**:`render_pipeline/blender_script/render.py:237`:

```python
bbox_min, bbox_max = scene_bbox()
scale = 1 / max(bbox_max - bbox_min)
```

这两行有 4 个未处理 case:
1. **degenerate mesh**(顶点共点,bbox = 0)→ `max(...) ≈ 0` → `scale → ∞`,物体被推出画面
2. **极细 mesh**(1mm 厚的薄片,bbox 跨 1m)→ scale 巨大,物体只占 1 像素
3. **巨大 offset 的 sub-mesh**(bbox 算出来的中心远离物体真实"视觉中心")→ 推到画面外
4. **inverted normal / 透明 material** → mesh 几何存在但渲染不可见

**Pipeline 第二层缺陷**:
- blender_script 渲完一帧 / 一 obj **没校验** alpha 是否 > 0
- 编排器 render.py 渲完一 batch **没校验** tar 内 PNG 任何像素非透明
- 全管线唯一的"质量信号"是 phases.json 写入即认为成功 — **0 质量门**

### 8.5.3 Issue 2 跟 Issue 1 的关系

- 实测 NaN 触发条件:`mask = torch.zeros(...).bool()`,即整 batch × 所有 target view 都没有前景像素
- **健康数据**(blacklist 过滤后剩 44,129 obj):基本不可能(8 batch × 3 target view × 448² pixel 全 0 概率 ≈ 0)
- **broken 数据进 batch**:1 个 obj alpha 全 0,batch=8 中 1 个 → mask 不会 100% 空(其他 7 个有前景),不会 NaN
- **batch=1 + broken obj**(e.g. eval / debug):**直接 NaN**
- **batch=1 + 多 target view 中至少一个空**:可能 NaN

**结论**:NaN 不完全是 issue 1 的症状,batch=1 eval 路径会独立触发。但 issue 1 解决后训练 batch=8 路径几乎不会触发。

### 8.5.4 三层防御(prevent → detect → guard)

不是 "打 3 个补丁",而是 **3 个生效优先级不同的层次**:

#### Layer 1 — PREVENT:render pipeline 内嵌质量校验(机制化根因修)

**目标**:future render run 物理上不可能产出空 tar。

**改 `render_pipeline/blender_script/render.py`**(具体行号在实施时确认):

1. **加载 GLB 后**,在 normalize_scene 之前,断言 mesh 有 >0 个 polys + bbox 极差 ∈ [1e-4, 1e3]。失败 → 写 `phases.json {"status": "skipped", "reason": "degenerate_mesh", "bbox_extent": ...}` 并 `return`,**不渲 40 帧**(省 ~6s/obj × 几百个废 obj = 节省一两小时)。
2. **render 第一帧后**,读回 PNG alpha 通道。alpha_max < 0.01 → 写 `phases.json {"status": "skipped", "reason": "empty_alpha", ...}` 并 `return`,**不渲剩 39 帧**。
3. 没失败 → 继续走完,**phases.json 必须包含 `quality_check.alpha_max_per_view` 数组**(40 个数,供下游脚本扫描)。

**改 `render_pipeline/render.py`(编排器)**:
- 单 obj batch 结束、`tar -cf` 之前,scan phases.json `quality_check` 字段,所有 view alpha_max > 0.01 才进 tar;否则把 obj uid 写 `<output_dir>/quarantine.txt`,**tar 都不生成**(节省存储 + 自动天然 blacklist)。

**效果**:Phase 2/3 重渲数据时,broken obj 自动不进 .tar,**不需要事后扫**。

#### Layer 2 — DETECT:把 ad-hoc /tmp scan 升级成 pipeline artifact(机制化 detect)

**目标**:对于**已渲染好的旧数据**(Phase 1a 44,798 这批,sunk cost),不重渲、但要把过滤决策固化到仓库。

1. **把 user 写的 48-worker fastscan 脚本搬进 repo**:
   - 移到 `render_pipeline/scan_quality.py`(目前是 /tmp/...)
   - 加 README 说明它的角色 = "post-build quality gate"
2. **scan 输出标准化**:`<dataset_root>/quality_scan_<rev>.json` + 派生的 `<dataset_root>/blacklist.txt`(一行一 UID)
3. **dataset 加载方**(`data/dataset_objaverse.py`)对 blacklist 是"约定大于配置":
   - 默认在 `tar_root_path` 同级查找 `blacklist.txt`,存在就 filter,日志打印 "dropped N obj"
   - **不**需要在 config yaml 里硬编码路径
4. Layer 1 上线后,blacklist.txt 应该越来越空(理想情况下空文件)。

**好处**:
- 老数据仍能用(老 path 仍 work)
- 数据 + 拉黑同 dir,谁拿到 dataset 就拿到 blacklist,不会失联
- 跟 git 关系干净:repo 只放代码,数据 + blacklist 共住 dataset 目录(`/home/z50057756/data/objaverse_renders_44798/`)

#### Layer 3 — GUARD:训练 loss 防御性 NaN guard(机制化最终兜底)

**目标**:无论数据多脏 / 跑 batch=1 eval / 未来加任何新数据集,training loop 不能因为空 mask 死掉。

`model/loss.py:207` 加 `if mask.any()` 分支(具体改动见 §8.5.6)。

**为什么仍要做(虽然 Layer 1+2 几乎兜住)**:
- Layer 1 不溯及既往(老 .tar 已经写了)
- Layer 2 是统计性过滤(`alpha_mean < 0.005` 阈值),可能漏一些"前景 5 个像素"的边缘 obj — view_selector 抽出来后 + roll augment 旋转,剩余前景可能正好 0 像素
- eval / debug 路径 batch=1,空 mask 概率不为 0
- **完全不影响有前景的 happy path**

### 8.5.5 三层在 Phase 1a / Phase 2 / Phase 3 的部署节奏

| 阶段 | Layer 1 (prevent) | Layer 2 (detect) | Layer 3 (guard) |
|---|---|---|---|
| **Phase 1a (now)** | ❌ 不做(老数据不重渲) | ✅ 把 /tmp scan 升级到 repo + 数据目录 + dataset 自动读 blacklist | ✅ loss NaN guard |
| **Phase 2** (50K obj) | ✅ 渲新数据前加 Layer 1 | ✅ 重跑 scan 校验 Layer 1 真的兜住 | ✅(已上线) |
| **Phase 3** (330K obj) | ✅(已上线,直接受益) | ✅ 验证 blacklist.txt 几乎为空 | ✅(已上线) |

### 8.5.5b 全量 fg audit 实测(2026-06-22)— 更新拉黑阈值

旧 /tmp scan 只覆盖 669 flagged obj。我们重做全量审计(job 81239,sbatch cpu × 96 worker × 933s wall):

- 全部 44,798 obj × 40 view = **1,791,920 帧 alpha 扫**(0 错误)
- 输出:`/home/z50057756/tmp/fg_audit/fg_audit_full.json`(46 MB,per-obj `fg_per_view[40]`/`fg_mean`/`fg_min`/`fg_max`/`fg_std`/`fov_deg`)

**全量分布**:mean=0.1505,median=0.137,p10=0.040,p90=0.276。L3 §6.3 20-batch sample 报 0.143 ✓ **跟全量 mean 0.1505 一致** — 设计预期完全对齐。

**vs 设计预期 r_frame=0.684**:内切视椎台 → 单位 cube 球面积 ≈ 0.367,典型 Objaverse mesh silhouette 占球 ~40% → 预期 fg ≈ 0.15 ✓ 实测吻合

**per-FoV bucket 验证 A2**:FoV ∈ [25°, 70°] ✓,fg 随 FoV 单调降(25°→0.148,70°→0.118)— 跟"大 FoV 物体占比小"的设计一致 ✓

**直方图**(per-obj fg_mean):

| 区间 | obj 数 | 比例 | 视觉验证后判定 |
|---|---|---|---|
| [0.000, 0.001) | 266 | 0.59% | **铁定坏**(画面没东西) |
| [0.001, 0.005) | 147 | 0.33% | **大概率坏** |
| [0.005, 0.010) | 308 | 0.69% | 模糊带,视觉看是真细瘦物体 |
| [0.010, 0.050) | 5,287 | 11.80% | **真细瘦物体**(笔/章鱼/椅腿等),视觉 12 个全是真的 |
| [0.050, 0.150) | 19,378 | 43.26% | 主分布 |
| [0.150, 0.300) | 21,148 | 47.20% | 主分布(累计) |
| > 0.300 | 3,340 | 7.46% | 大物体 |

视觉验证 PNG:`/home/z50057756/tmp/fg_audit/viz_bucket_*.png` × 4(低 fg 各区间 12 个随机样本)

**用户决策(2026-06-22)**:**拉黑 fg < 0.005 = 413 obj**([0.000, 0.001) ∪ [0.001, 0.005)),保留 [0.005, 0.050) 是真细瘦物体(模型应该学这种)。

旧 scan 669 vs 新 audit 413 的关系:
- 旧 [all_empty + mostly_low] = 669 个 flagged
- 新 fg<0.005 严格 = 413,跟旧 669 重叠 398(旧 scan 漏掉 15 个,边界阈值差异)
- **以新 audit 413 为准**(扫得更全,数学定义干净)

### 8.5.6 Phase 1a 具体落地(就这一次)

#### Step 1 — Layer 2 落地(blacklist)

文件落位(数据 + blacklist 同住,跟代码 repo 解耦):
- `/home/z50057756/tmp/fg_audit/fg_audit_full.json` → 复制到 `/home/z50057756/data/objaverse_renders_44798/fg_audit_full.json`(跟数据同住 audit trail)
- 从中抽 fg_mean < 0.005 的 **413 个 UID**,写入 `/home/z50057756/data/objaverse_renders_44798/blacklist.txt`(413 行 UID,文件头加 `# threshold: fg_mean < 0.005 ...`)
- **scan 脚本**:`/home/z50057756/tmp/fg_audit/fg_audit_full.py` 搬进 `render_pipeline/scan_quality.py`,加简短 README(input/output schema 说明)

#### Step 2 — `dataset_objaverse.py` 自动读 blacklist

约定:在 `self.tar_root_path`(或 `self.root_path` 如果不用 tar)的同级找 `blacklist.txt`,自动 filter。**无需 config field**(零 config 改动)。

```python
# In ObjaverseDataset.__init__, after line 39:
# Layer 2: auto-detect a blacklist.txt next to the data.
#   Convention: <data_root>/blacklist.txt (one UID per line, comments with #).
#   Generated by render_pipeline/scan_quality.py post-render. Future render
#   pipelines (Layer 1) will quarantine broken obj at render time, so this
#   list shrinks over time.
data_root = self.tar_root_path if self.use_tar else self.root_path
blacklist_file = os.path.join(data_root, "blacklist.txt")
if os.path.exists(blacklist_file):
    with open(blacklist_file) as f:
        blacklisted = {l.strip() for l in f if l.strip() and not l.startswith("#")}
    before_n = len(self.all_object_list)
    self.all_object_list = [u for u in self.all_object_list if u not in blacklisted]
    print(f"[ObjaverseDataset] applied {blacklist_file}: "
          f"{before_n} → {len(self.all_object_list)} ({before_n - len(self.all_object_list)} dropped)")
```

后向兼容:FluffyElephant_tar 没 blacklist.txt → if 不进 → 行为不变。

#### Step 3 — Layer 3 落地(`model/loss.py:207`)

```python
# P4e: defensive guard. Empty mask -> rendering[mask] is empty tensor,
#      .mean() returns NaN, poisoning the backward pass.
#      Should be near-impossible after Layer 2 blacklist + Layer 1 (Phase 2+),
#      but cheap to keep as a final net (eval batch=1 / debug paths).
if mask.any():
    l2_loss = ((rendering[mask] - target[mask])**2).mean() + 1e-3 * F.mse_loss(rendering, target)
else:
    l2_loss = 1e-3 * F.mse_loss(rendering, target)
```

数值兼容:有前景时行为完全不变。

#### Step 4 — Issue 3 doc fix(独立小 commit)

`render_pipeline/render.py:4-13`:把 `- A3: pitch clipped to upper hemisphere [-15°, 80°]` 改成 `- A3: TRELLIS-original pitch/yaw (sphere_hammersley_sequence untouched — already 75/25 upper-hemisphere biased)`,加 `- A2: per-object FoV ...` + `- R2: per-view radius jitter ...`,跟 line 43-49 function docstring 对齐。

#### Step 5 — Layer 1 不做(Phase 1a scope 外),只留 TODO

不在 Phase 1a scope。新增 `render_pipeline/TODO_phase2.md`(或写进 plan §8.6),列具体修改点:
- `blender_script/render.py:~237`(normalize_scene 前 degenerate-mesh 断言)
- `blender_script/render.py:~428`(render 完第一帧 alpha 检查)
- `render_pipeline/render.py:~276`(tar 前 phases.json quality_check 字段校验)
- 测试:重渲已知 broken UID(从 blacklist 抽 5 个),验证 phases.json `status=skipped`,无 .tar 输出

### 8.5.7 验证 + commits

5 个 commit on `render44798-dataset-patches`(从 1e2bc7f 之后):

1. `data: post-render quality scan v1 + 669-UID blacklist for objaverse_44798` (数据文件,2 个 file,不入 git 因为太大?决定:JSON 写仓库 data/,blacklist 写 data root)
   - 实际:
     - `cp /tmp/rng_low_alpha_fastscan.json $HOME/data/objaverse_renders_44798/quality_scan_v1.json`
     - 用 Python 抽 669 UID → `$HOME/data/objaverse_renders_44798/blacklist.txt`
     - 仓库内只 commit scan 脚本(`render_pipeline/scan_quality.py`) — 实际 JSON / blacklist 跟数据同住,不进 git
2. `tool(scan_quality): move /tmp fastscan into render_pipeline, produces blacklist.txt next to data` (scan 脚本入仓库)
3. `feat(P7): ObjaverseDataset auto-detect <data_root>/blacklist.txt and filter UIDs` (dataset 改)
4. `fix(P4e): loss exclude_bg guard against empty foreground mask (NaN -> bg-only term)` (loss 改)
5. `docs: render.py header — A2 per-obj, A3 TRELLIS-original, R2 radius jitter` (注释)
6. push

### 8.5.8 sanity 验证(每步打门)

| Step | 验证方式 |
|---|---|
| 1 (scan) | `wc -l blacklist.txt` == 669;`comm -12 <(sort objaverse_44798.txt) <(sort blacklist.txt) | wc -l` == 669;quality_scan_v1.json schema 跟 /tmp 原版完全等价 |
| 2 (scan script) | 跑 1 个 obj 看输出 dict 跟 fastscan.json 字段一致 |
| 3 (dataset) | 起 1 节点 1 GPU sbatch,**rank 0 console 必须**打印 `[ObjaverseDataset] applied .../blacklist.txt: 44798 → 44129 (669 dropped)` |
| 4 (loss) | inline python smoke:`LossComputer.forward(rendering=randn(1,7,3,448,448), target=randn(...), exclude_bg=True, target_alpha_mask=zeros(1,7,1,448,448))` 返回 l2_loss 必须是 finite scalar |
| 5 (doc) | git diff 自检 |

### 8.5.9 提交全量训练

`sbatch scripts/rngUP448_render44798_8h200.sbatch` 后监控:

- ✅ rank 0 log "44798 → 44129"
- ✅ 前 100 step l2_loss 不 NaN(老 P4d 修完已经稳定,加 P4e 是双保险)
- ✅ iter median ~6.5s
- ✅ 推算 wall: 27h < 48h
- ✅ wandb 看 fg PSNR > full PSNR(alpha mask 真在用)

### 8.5.10 Phase 2/3 → 见 §8.6

Phase 2 render strategy 在 §8.6 单独写(包含 Layer 1 degenerate gate + thin-obj adaptive framing 设计)。

---

## 8.6 Phase 2 render strategy:Layer 1 + thin-obj 自适应取景(2026-06-22 规划)

### 8.6.1 Context

`r_frame = 0.684` 内切视椎台是按"单位 cube"设计的。对细瘦物体(笔/章鱼触手/椅腿,bbox 比例 > 5)失效:longest axis = 1,silhouette 只占 1-5%,模型很难学到细节。

但 fg ∈ [0.005, 0.050) 的 ~5,000 个 obj 是**真实的细瘦物体**,不是 broken render。如果只是过滤掉,数据集失去这些 shape 的 coverage。

**Sim2real 视角**:真实手机拍照,**用户拍细物体时会移近 / 放大**,得到 fg 占比 ~10-30%(跟拍 cube 类似)。当前 render 强制 fg 跟 bbox-aspect 挂钩,跟真实分布有差距。**自适应取景 = sim2real patch 的天然延伸**(跟 A1/A2/R2 同类)。

### 8.6.2 5 个候选 strategy 比较

| Strategy | fg 分布变化 | TRELLIS 对齐 | sim2real | 实施 | 风险 |
|---|---|---|---|---|---|
| **A** Pilot 渲染 + 迭代 radius | fg<0.05 尾部 →~0.10-0.15;cube 不变 | per-obj 偏离(radius 不再是 fov 纯函数) | + 模拟"拍照拉近" | 高(额外 render call + 状态清理) | pilot 低 spp 误导;非确定性 |
| **B** 几何均值 bbox 归一化 | pen 1×0.05×0.05 fg 5%→37%;cube 不变 | **硬偏离** — aabb 不再 [-0.5,0.5]³ | + 真拍不按 max 轴归一 | 低(1 行改 normalize_scene) | 长物体可能 clip;下游假设 aabb=unit 都会炸 |
| **C** 大宽长比时缩 FoV | ≈ 不变(fov-radius 耦合抵消) | 保留 | 无 | 中 | **无效** — Plan agent 实测耦合让 FoV 抵消 |
| **D** 保 TRELLIS framing + 只加 Layer 1 quality gate | 拉黑 266,留细瘦 obj 1-5% fg | 完美 | 0 | 低(已规划) | 模型对细物体的 closeup 学不足 |
| **E** D + 温和 adaptive(detect bbox-aspect > 5 → k×radius) | fg<0.05 尾部 →~0.08-0.10;主分布不变 | 几乎对齐(只动 ~13% 极端比例 obj) | + 温和 sim2real | 中 | 极长物体边缘 clip 增加(可控) |

### 8.6.3 推荐:Strategy E(D + 温和 adaptive)

**E.1 — Layer 1 必做(quality gate)**:
- `blender_script/render.py:~224-244` 的 `normalize_scene()` 入口加断言:
  ```python
  extents = bbox_max - bbox_min
  if not np.isfinite(extents).all() or max(extents) < 1e-6:
      return SKIP("empty_or_nonfinite_bbox")
  if min(extents) < 1e-4 * max(extents):
      return SKIP("degenerate_planar_mesh")
  ```
- 编排器 `render.py` 看到 SKIP → 不写 .tar
- 兜住 §8.5.5b 的 266 broken(0.59%)

**E.2 — 温和 adaptive framing**(2026-06-28 修正公式 — 之前 plan 数学方向倒了):

**正确公式**(物理:fg 反比 radius²):
```python
# normalize_scene 之后, render 之前(per-obj 算一次, 40 view 共用)
ext = bbox_max - bbox_min   # max(ext) == 1.0
aspect = max(ext) / max(min(ext), 1e-3)
if aspect > ASPECT_THRESH:
    # bbox 几何均值预测 fg(0.15 = 全量 audit 主峰)
    fg_predict = 0.15 * (np.prod(ext) ** (1/3) / max(ext)) ** 2
    # k = sqrt(fg_current / fg_target),clamp [0.5, 1.0]
    k_raw = math.sqrt(fg_predict / FG_TARGET)
    k = max(0.5, min(1.0, k_raw))
    for view in views:
        view["radius"] *= k       # 拉近(k<1)或不动(k=1)
```

物理:`fg_after / fg_before = (r_before / r_after)² = 1/k²`,所以 `k = sqrt(fg_before / fg_target)`。
- `fg_predict < fg_target` → k<1 拉近 ✓
- `fg_predict ≥ fg_target` → k≥1 clamp 到 1.0 不动 ✓
- floor `k ≥ 0.5`:最多 4× lift,防过近(camera 不会穿物体)

**FG_TARGET 怎么选**(2026-06-28 离线模拟实测):
- 0.05:只拉细物体(self-filter),0 主分布误伤,lift 1.6× — **最安全**
- 0.08:覆盖 [0.05, 0.08) 部分主左,lift 2.6×
- 0.10:**拉所有 fg<0.10 的 obj(35%)**,主分布左 9654 obj 100% 误伤
- 0.15:更激进,主分布左+部分主峰被拉

**FG_TARGET=0.10 + ASPECT 门**:用户明确要 0.10,所以**必须**配合精确 ASPECT 门防止主分布破坏。aspect 在 normalize_scene 后从 bbox 直接拿(精确,不像离线用 fg_max/fg_min 当代理)。

**ASPECT_THRESH 待定**(2026-06-28 silhouette-代理小样本测了 30 obj/bucket,趋势:>5 在 [0.005-0.025] 极瘦档 80% 覆盖,但 [0.025-0.05] 瘦档只 23% 覆盖)。
- 待全量 silhouette audit(`aspect_audit_full.py`)出全 44,798 obj 真实分布再敲定
- 也可在 render time 用真 mesh bbox(最准),先用 silhouette 估出"合适阈值",再投真 mesh aspect 用

### 8.6.4 Decisions FINALIZED(2026-06-28 修订 v3)

**Critical discoveries**:
1. FluffyElephant 全量 audit(job 83346)实测 fg p50=0.190 vs render44798 p50=0.134,**整体小 ~30%**
2. **mesh 集 100% 相同**(我们和 FE 都用 LVIS UID 子集,44,527 obj 100% overlap),per-obj 1:1 配对 fg ratio r44/FE = 0.715(各分位常数)
3. **28.5% gap 完全来自 setting**:R2 占 9%,A1 占 ~0%,**A2 + 其他渲染细节占 ~20%**(可能大 FoV perspective distortion 放大 silhouette 收缩)
4. r_frame=0.684 内切 unit cube 对角 0.866,本身就有 3.15% clip(TRELLIS 设计接受的 baseline)

**最终改造方案 v3**(三层 stacking,保留 A1/A2 sim2real 同时对齐 FE 主分布):

| # | 决策 | 选定值 | 依据 |
|---|---|---|---|
| 1 | **R2 收紧** | `U(0.95, 1.15)` → **`U(0.95, 1.10)`** | mean radius 1.05 → 1.025,fg × 1.049 |
| 2 | **r_frame 提升** | 0.684 → **0.74** | fg × 1.170,p50 → 0.164,KS vs FE → 0.089 |
| 3 | A1 lookat offset | **保留 ±0.15**(不动) | 实测对 fg 几乎无影响,但保 sim2real |
| 4 | A2 per-obj FoV | **保留 [25°, 70°]**(不动) | 跟手机相机真实范围匹配(1× 主摄 70°, 3× 长焦 25°)|
| 5 | Strategy E fg_target | **0.20** | 细物体拉到主峰 |
| 6 | aspect 阈值 | **5×** | 全量 silhouette audit (job 83333) 实测 |
| 7 | k 粒度 | **per-obj** | 40 view 共用 k |
| 8 | aabb 字段 | **保留 [-0.5, 0.5]³** | 下游 assert 不炸 |
| 9 | Layer 1a/1c | **必做** | 防 broken obj |

**为什么不动 A2 FoV**:用户讨论确认手机相机 1× 主摄 ~70°,3× 长焦 ~25°,FE 的 40° 是产品摄影 50mm 焦距,不是手机自然分布。我们 [25°, 70°] 才贴近真实手机使用。

**完整公式 v3**(替换原 §8.6.3):
```python
R_FRAME = 0.74                                    # was 2·sin(20°) ≈ 0.684 (TRELLIS)
R2_LO, R2_HI = 0.95, 1.10                         # was 0.95, 1.15
FG_TARGET_E = 0.20
ASPECT_THRESH = 5

# Step 1: Layer 1a (degenerate gate)
ext = bbox_max - bbox_min
if max(ext) < 1e-6 or min(ext) < 1e-4 * max(ext):
    SKIP("degenerate")

# Step 2: base radius from new r_frame
radius_base = R_FRAME / math.sin(fov_rad / 2)

# Step 3: Strategy E adaptive zoom (only for thin obj)
aspect = max(ext) / max(min(ext), 1e-3)
if aspect > ASPECT_THRESH:
    fg_predict_old_r = 0.15 * (np.prod(ext)**(1/3) / max(ext))**2
    fg_predict = fg_predict_old_r * 1.228   # × (r_frame/0.684)² × R2_mean²
    k_raw = math.sqrt(fg_predict / FG_TARGET_E)
    k = max(0.5, min(1.0, k_raw))
else:
    k = 1.0

# Step 4: per-view radius (tighter R2)
for view in views:
    view["radius"] = radius_base * k * uniform(R2_LO, R2_HI)
```

**预期完整分布(离线模拟 v3,vs FE baseline)**:

| 分位 | 原 | + R2 紧 | + r_frame 0.74 | + Strategy E | FE |
|---|---|---|---|---|---|
| p10 | 0.040 | 0.042 | 0.049 | **0.075** | 0.061 |
| p50 | 0.134 | 0.140 | **0.164** | **0.190** | 0.190 |
| p90 | 0.276 | 0.290 | 0.339 | 0.339 | 0.383 |
| mean | 0.151 | 0.158 | 0.185 | 0.198 | 0.211 |
| **KS vs FE** | 0.213 | 0.184 | **0.089** | 0.089 | — |

- **R2 紧 + r_frame 0.74**:主分布 KS=0.089 跟 FE 接近
- **+ Strategy E**:p50 提到 0.190 == FE,但细物体长尾偏离(有意,为 sim2real)
- Strategy E 触发预估 6,734 obj (15%)

**⚠ Clip 风险**(必 smoke 验证):
- 单位 cube 对角线 = sqrt(3)/2 ≈ 0.866
- r_frame 0.684 余量 0.182(老 setting,实测 clip 3.15%)
- r_frame 0.74 余量 **0.126**(margin 缩 1.44×,线性外推 clip ≈ 4.5%,真实可能 5-8%)
- **不需要 r_frame=0.794** 的激进路线 — 0.74 已经够大,clip 风险可控
5. **是否也对 fg ≥ 0.05 启用 E.2**:推荐**否**,保持主分布对齐 TRELLIS。

### 8.6.5 实施位置 + 改动范围(实测代码 mapping,2026-06-28)

**4 个 layer × 3 个文件 × 共 ~30 行新增代码**:

| Layer | 文件 | 行号 | 现有代码 | 改动内容 |
|---|---|---|---|---|
| **1a degenerate gate** | `blender_script/render.py:224-244` `normalize_scene()` | L237 现状 `scale = 1 / max(bbox_max - bbox_min)` 没断言 | L236 之后插 `extents = bbox_max - bbox_min`,断言 `max(extents) ≥ 1e-6` 且 `min(extents) ≥ 1e-4 * max(extents)`,失败抛 `SkipRender("degenerate_mesh")` 异常 |
| **1b thin-obj adaptive zoom** | `blender_script/render.py:382-415` `render_one_object()` 取 view["radius"] 前 | L385 `radius = view["radius"]` 单读 | 在 L365 `normalize_scene()` 之后 +1 段 `aspect/k` 计算;若 aspect>阈值,**整 obj 所有 40 view 的 `view["radius"] *= k`**(per-obj 方案,简洁) |
| **1c post-render alpha check** | `blender_script/render.py:382-495` 渲染循环 | L428 `bpy.ops.render.render(write_still=True)` 后无校验 | L428 之后 (仅 i==0 即第一帧) 读 PNG alpha,`alpha.max() < 0.01` 即 break 出 view 循环,标 `phases["status"] = "skipped_empty_alpha"` |
| **Orchestrator skip** | `render.py:280-284` 当前只判断 `phases.json` 存在 | 现状:phases.json 在 = 渲染完 → tar | 改成额外读 `phases["status"]`,`== skipped_*` 也跳过 tar + append 到 `<output_dir>/quarantine.txt`(uid + reason) |

**关键既有能力**(可直接复用,不需新依赖):
- L211 已经过滤非 mesh object(`isinstance(obj.data, bpy.types.Mesh)`)→ 我们只需加 bbox 退化检查
- L303 已 `import PIL`,L30 已 `import numpy as np` → Layer 1c 读 PNG alpha 不需要新 import
- L312-320 `draw_debug_overlay` 已有 PIL+numpy 读 alpha 的范式 → 直接抄
- L179-188 orchestrator 已有 restart-safe `.tar` 存在性 skip → 跟 Layer 1c orchestrator 端的逻辑同模式

**不动**:
- `render.py` jobs.json 生成(`_patched_views`)/ RADIUS_REF 公式 — 整 obj 渲染前的相机采样逻辑不变
- `transforms.json` schema — 不污染下游 metadata(k 只作用 render 阶段的 `view["radius"]`,不写回 transforms.json)
- `aabb` 字段仍 `[-0.5, 0.5]³` — 下游 FluffyElephant 代码 assert aabb 不会炸

### 8.6.6 测试协议(50K 渲染前)

| 阶段 | 内容 | 通过准则 |
|---|---|---|
| 1. unit | 4 合成 mesh(cube / 10×1×1 rod / 1×1×0.01 plane / sphere) | E.1 抓 plane,E.2 cube/sphere k=1.0,rod k≈0.5 |
| 2. smoke 5 obj | 从 viz_bucket_001_to_005.png 取 5 个 fg<0.05 真细物体,E off vs E on | E on 后 fg 提到 ≥ 0.08;clip 不增加 |
| 3. 500 obj stratified pilot | 每 fg bucket 100 个(< 0.01 / 0.01-0.05 / 0.05-0.15 / 0.15-0.30 / > 0.30) | fg<0.05 中位 ≥ 0.08;fg≥0.05 中位变化 ≤ ±10%(保留主分布);edge-clip rate < 15% |
| 4. KS 对齐 | 跟 FluffyElephant fg 直方图算 KS distance | KS < 0.10 in fg≥0.05 region |
| 5. 通过才开 50K | — | — |

### 8.6.7 Render-time 开销

- E.1:~0.5 ms/obj(一次 bbox 检查)
- E.2:~1 ms/obj(闭式 k 计算)
- **没有额外 render pass**
- 预测吞吐**不变**:7.3 vps × 4 GPU(§4.2 已 benchmark)

### 8.6.8 跟现有 plan 接口

- §8.5 Phase 1a 拉黑 413 obj:**只对老数据**(44,798 那一批)。Phase 2 起 E.1 上线后,新数据物理上不会产生 broken obj。
- §8.5.5 三层(prevent/detect/guard):E.1 = Layer 1 PREVENT 的具体实现。Layer 2 (detect) 仍跑 (作为 E.1 的兜底);Layer 3 (loss guard) 仍保留 (eval batch=1 防御)。
- Phase 2 渲染量 50K → 验证 E.1+E.2 工作正常 → Phase 3 渲 330K 时直接用。

---

## 8.7 Sim2real GSO 重渲 + roll on/off 双 eval(2026-06-28 规划)

### 8.7.1 Context

Phase 1a 训完 (`RnGUP_obj_448_render44798`, step 15000) 在**原 FluffyElephant-rendered GSO** 上跑出 PSNR 23.08 / RA5 80.68(详 §8.5/§8.6 之后的 eval 结果)。这个测试集用 TRELLIS-FluffyElephant 收紧渲染 convention(uniform FoV=40°,严格 centered,无 roll),对 sim2real 训练出来的我们模型是 **out-of-distribution**,被压 -1.91 dB PSNR / -6 RA5 属正常 tradeoff。

**用户需求**:重渲一组 GSO 用我们 sim2real 渲染算法(A1/A2/A3/R2 production 配置)→ 在 in-distribution 数据上测我们模型;两组对比:
- (a) **无 z-roll**:dataloader `roll_augment_max_deg=0` — 输入图无 roll,跟 FluffyElephant 风格对齐除 sim2real 渲染参数外
- (b) **有 z-roll**:`roll_augment_max_deg=10.0` — 每 view 独立 ±10° roll,模拟手持

回答的问题:模型对 z-roll 的 robust 程度;sim2real 渲染分布下模型表现是否回升到 FA3 baseline 水平甚至更好。

### 8.7.2 输入 / 输出 / 工具

| 项 | 值 |
|---|---|
| GSO mesh 源 | `modelscope.cn/datasets/XiangMochu/GSO_mesh` (1029 个 PLY mesh 期望对得上 `data/gso.txt`) |
| 下载工具 | `pip install modelscope` → `modelscope download --dataset XiangMochu/GSO_mesh --cache_dir <local>` |
| GSO mesh 落位 | `/home/z50057756/data/gso_meshes/` |
| 渲染 sbatch | `sbatch_gpu_multi.sh` 改名 + 改 `OBJ_LIST`/`OUTPUT_DIR`/`NUM_VIEWS` |
| 渲染输出 | `/home/z50057756/data/gso_sim2real_25v/<scene>/{000..024}.png + {000..024}_depth.png + transforms.json` (**loose 不 tar**,跟 dataset_gso_ours.py 期望对齐) |
| 渲染 wall | **~15 min** (1029 × 25 view / 29.32 VPS,4 H200 N=4 procs/GPU,§4.2 benchmark) |
| Eval 模型 | `RnGUP_obj_448_render44798` step 15000 (已 push 的 ckpt) |

### 8.7.3 Render command(关键 — production 44,798 同套参数)

```bash
$BLPY render.py --obj_list <gso_obj_list.txt> \
    --output_dir /home/z50057756/data/gso_sim2real_25v/ \
    --num_views 25 \
    --resolution 512 --device GPU --samples 128 \
    --offset_max 0.15 --offset_zero_prob 0.20 \
    --fov_min_deg 25 --fov_max_deg 70 \
    --radius_min_factor 0.95 --radius_max_factor 1.15 \
    --r_frame 0.684 \
    --persistent
```

**注**:render.py 默认值是 FoV [40, 75] / r_frame=0.74,**production 44,798 是 cmdline override [25, 70] / 0.684**(plan §1.3)。新渲染必须用相同 override 才能跟训练分布对齐。**没有 `--tar_output`**,要 loose 因为 dataset_gso_ours.py `use_tar=False`。

### 8.7.4 obj_list 构造

GSO mesh 文件名跟 `data/gso.txt` line(`11pro_SL_TRX_FG` / `2_of_Jenga_Classic_Game` ...)期望一一对应。

实施:
1. modelscope 下载完看实际文件 layout(可能是 `<scene_name>/model.obj` 或 `<scene_name>.glb`)
2. 写 Python 一行:`[f"/home/z50057756/data/gso_meshes/{name}/model.obj" for name in open('data/gso.txt').read().split()]` → `gso_obj_list.txt`
3. 验证全部存在:`while read p; do test -f "$p" || echo MISS: $p; done < gso_obj_list.txt`
4. 缺的 mesh → 从 `gso.txt` filter 出来另存一份 `gso_available.txt` 给 inference 用

### 8.7.5 双 inference 跑

复用 `scripts/inference_obj_repa_postmigration.sh`(已 post-migration patched)。两 job 共用同一渲染输出 + 同 ckpt,只 dataloader knob 不同:

```bash
# Run A (no roll)
CONFIG_PATH=configs/RnGUP_obj_448_bf16_15k_render44798.yaml \
CHECKPOINT_DIR=/mnt/data-alpha-sg-01/.../experiments/checkpoints/RnGUP_obj_448_render44798 \
INFERENCE_OUT_DIR=/home/z50057756/.../experiments/evaluation/RnGUP_render44798_gsoSim2real_noroll \
SPLIT_FILE=data/gso_available.txt \
EXTRA_OVERRIDES="training.val_dataset_cfgs.root_dir=/home/z50057756/data/gso_sim2real_25v \
                 training.val_dataset_cfgs.suffix='' \
                 training.roll_augment_max_deg=0" \
sbatch scripts/inference_obj_repa_postmigration.sh

# Run B (roll ±10°) — same as A but training.roll_augment_max_deg=10.0
```

sbatch 需要扩成接受 EXTRA_OVERRIDES,把变量内嵌到 cmdline。

**注**:`val_dataset_cfgs.suffix` 在现有 FluffyElephant config 是 `render_mvs_25/model/`(因为 layout 是 `<scene>/render_mvs_25/model/<frame>.png`),我们新渲染是 flat `<scene>/<frame>.png`,suffix 要清空。

### 8.7.6 Roll 在 inference 时的语义验证(关键)

`dataset_gso_ours.py` 继承 `ObjaverseDataset.preprocess_frames`,其中 P2 patch (§6.2) 每 view 独立采 roll_deg,**image + depth + alpha + c2w 同角度旋转**。所以:

- Run A (roll=0):roll_deg 强制 0,等于 identity,跟现有 GSO eval 同
- Run B (roll=10):每 view 独立采 [-10°, +10°] → image/depth rotated by that angle, c2w 后乘 Rz_cam(roll_rad)

**Eval 公平性**:input 跟 target view 各自独立 roll,target c2w 也跟着 target image 一起 roll,所以 GT 和 prediction 在 rolled frame 内可比 → metrics 公平。

### 8.7.6b View 选择改读 `gso_pairs.txt`(用户决定 2026-06-28)

**Context**:eval-side `dataset_gso_ours.py:32-33` 现在用 `random.seed(0) + random.sample(range(25), 14)` 实时算每 scene 14-tuple。这跟仓库根目录 `gso_pairs.txt`(1030 行,`<scene_name> <14 indices>`)在原 gso.txt 顺序下 **1030/1030 完全相同**(实测验证)。

**问题**:RNG 是 position-based。如果 sim2real 重渲只 cover 1029 中部分 scene → split file 被 filter → 同 scene name 在新 list 的 idx 错位 → 14-tuple 跟原 FluffyElephant baseline 不一样了 → per-scene 不可比。

**修改**:改 `data/dataset_gso_ours.py` 的 `__init__`,从位置-based RNG 改成 **scene-name-based lookup `gso_pairs.txt`**:

```python
# 旧 (line 32-33):
random.seed(0)
self.rand_idx = [random.sample(range(0, 25), 14) for _ in range(len(self.all_object_list))]

# 新:
pairs_file = self.config.training.val_dataset_cfgs.get(
    'view_indices_file', 'gso_pairs.txt')
with open(pairs_file) as f:
    scene_to_idx = {}
    for line in f:
        toks = line.strip().split()
        if toks:
            scene_to_idx[toks[0]] = [int(x) for x in toks[1:]]
self.rand_idx = []
missing = []
for name in self.all_object_list:
    if name not in scene_to_idx:
        missing.append(name)
    else:
        self.rand_idx.append(scene_to_idx[name])
if missing:
    raise KeyError(
        f"{len(missing)} scene(s) in split_file but not in {pairs_file}: "
        f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
    )
```

**新增 config field**(可选,有默认):
```yaml
val_dataset_cfgs:
  view_indices_file: gso_pairs.txt   # 仓库根目录;可不写,有默认
```

### 8.7.6c 影响面 + 兼容性

| 调用点 | 是否受影响 | 行为 |
|---|---|---|
| **训练侧** (`train.py` 主循环 用 ObjaverseDataset) | ❌ 完全不受影响 | dataset_gso_ours.py 不在 training data path 里 |
| **训练 val pass** (train.py `validate()` 调 GSODataset_ours 在 `gso_subset64.txt`) | ✅ 受影响,但**数值不变** | gso_pairs.txt 包含 1030 个完整 scene,subset64 是其子集,scene_name lookup → 同样的 14-tuple |
| **inference.py + 现有 GSO 全量 eval** (FA3 baseline + render44798) | ✅ 受影响,**数值不变** | 同上,scene_name lookup 跟原 random.seed(0) bit-exact |
| **新 sim2real GSO eval** | ✅ 这就是修改目的 | filter 后的 split file 仍能找到正确 14-tuple |

**承诺数值兼容**:gso_pairs.txt 跟 random.seed(0) 在 1030/1030 完全一致,改完后**任何已经跑过的 GSO eval 重跑 → 数字完全不变**(只要 split_file 里的 scene 都在 gso_pairs.txt 内)。

**重训 / 已存 checkpoint 完全不动**。仅 inference / val pass 数据加载路径改 ~15 行。

### 8.7.7 预算 + 阶段

| Stage | wall | 备注 |
|---|---|---|
| modelscope 下载 | ~30-60 min | 取决于 mesh 数据集大小(GSO ~10 GB 预估) |
| obj_list 校验 + 缺漏处理 | ~10 min | python script |
| 渲染 4 H200 sbatch | **~15-30 min** | 一次性,留出 buffer 跟 OPTIX 编译 |
| Run A (no roll) inference | ~40 min (1 H200) 或 ~10 min (4 H200) | 看分配 |
| Run B (roll 10°) inference | 同上 | |
| 聚合 + 8 metric LaTeX 行 × 2 | ~5 min | 复用 §8.5 之后的脚本 |
| **总** | **~2-3 小时** | |

### 8.7.8 结果 schema(预期 deliverable)

3 个 LaTeX 行:

```latex
 & GSO (FluffyElephant render, eval已有)      & 80.68 & 81.46 & 81.91 & 1.156 & 99.66 & 23.08 & 0.875 & 0.140 \\
 & GSO (sim2real render, no roll)             &  ?    &  ?    &  ?    &  ?    &  ?    &  ?    &  ?    &  ?    \\
 & GSO (sim2real render, roll ±10°)           &  ?    &  ?    &  ?    &  ?    &  ?    &  ?    &  ?    &  ?    \\
```

预期(假设):
- sim2real-no-roll vs FluffyElephant:**改善**(in-distribution,sim2real 渲染参数跟训练分布对齐)
- sim2real-roll vs sim2real-no-roll:**接近或略降**(roll augment 训练过,模型应该 robust)
- 如果 sim2real-roll 显著 better than sim2real-no-roll → 模型甚至**依赖** roll 多样性,no-roll 反而是 OOD(过拟合 roll 分布的可能,需要复议)

### 8.7.9 风险 + 应对

| 风险 | 缓解 |
|---|---|
| modelscope mesh 名跟 gso.txt 不全对得上 | 下载完做 set intersection,只用交集;**§8.7.6b 后 split file 可任意 filter** — view 选择跟 scene name 锁死 |
| GSO mesh 是 OBJ 不是 PLY,有 material/texture 文件依赖 | render.py 已支持 OBJ。如果 mesh 没 baked texture,渲出来是 gray solid — 可接受(GSO 本身大部分 mesh 单色) |
| modelscope 包装在新 home 装 conda env 但 env 在 /mnt/data-alpha-sg-01 | 用 `/mnt/data-alpha-sg-01/.../conda/envs/rng-fa3/bin/pip install modelscope` |
| inference 跑 25 view 而非 14 → 数据集 dataset_gso_ours.py 写死 25 总数 + 挑 14 | OK ✓(25 跟现有 GSO eval 一样,挑 14 也一样,改后由 gso_pairs.txt 锁定) |
| roll augment 改 c2w 后,GT pose 也变,跟 baseline 的 RA5/RT5 不公平比 | 这是设计:roll 模拟手持自然 roll,model 在 rolled frame 内预测 rolled GT pose,内部一致就行 |
| render 时间超预期 | 4 H200 仍是 baseline 比例,15 min 大概率达标;如果 >1h,scale down 到 200 scene smoke 先 |
| dataset_gso_ours.py 改完影响现有 FA3/render44798 eval 数字 | 改完先 smoke 跑 FA3 baseline 1 个 sample,对比 24.99/86.89 数字 → 必须 bit-exact |

### 8.7.10 执行顺序(每步明确 stopper)

| Step | 内容 | 通过准则 |
|---|---|---|
| 1 | Edit `data/dataset_gso_ours.py:32-33` 读 `gso_pairs.txt` | git diff 自检 |
| 2 | Smoke FA3 baseline eval 1 个 scene(或 64-subset val pass)→ 对比数字 | metrics 跟 §8.5 之后跑的数 bit-exact |
| 3 | `pip install modelscope` 到 /mnt/data-alpha-sg-01 conda env | `import modelscope` OK |
| 4 | `modelscope download --dataset XiangMochu/GSO_mesh --cache_dir /home/z50057756/data/gso_meshes/` | mesh 文件落盘,数量 ≥ 1000 |
| 5 | 检查 mesh 名跟 gso.txt 交集,写 obj_list | `wc -l obj_list ≥ 1000` |
| 6 | sbatch 4 H200 N=4/GPU 渲 sim2real(plan §1.3 production 参数,num_views=25,loose 不 tar) | 输出 `<scene>/000.png ... 024.png + transforms.json`,1029 ± 缺漏 |
| 7 | Run A inference:roll_augment_max_deg=0 | 25 min 内出 average_metrics_* |
| 8 | Run B inference:roll_augment_max_deg=10 | 同 |
| 9 | 聚合 + 3 行 LaTeX(FluffyElephant / sim2real-noroll / sim2real-roll10) | 8 指标 × 3 行 deliverable |

### 8.7.12 不在 scope

- 真实手机拍摄(Phase 1b 任务)— 等真实测试集采集
- 重训模型 — 只跑 inference
- Phase 2 渲染管线 Layer 1 改造(§8.6 那个)— 独立任务

| 资源 | 当前状态 |
|---|---|
| GPU partition | **`gpu`** (10 节点 H200 × 8),QOS=`low` (24h max) / `lowest` (无 cap) |
| CPU partition | `cpu` (1 节点 256 cores, 503 GiB) |
| Blender | 用户态 `$HOME/tools/blender-4.2.9-linux-x64/` (numpy/cv2/imageio bundled) |
| X11 lib | `$HOME/tools/x11_libs/` (compute 节点缺 libSM) |
| OPTIX cache | `$HOME/.cache/OptixCache` (跨节点持久化) |
| 已知坏节点 | gpu05 (SIGBUS) — 在 sbatch 加 `--exclude` |

### sbatch 模板(production 4-GPU 单 job)
```bash
#SBATCH --partition=gpu --qos=low
#SBATCH --gres=gpu:h200:4 --nodes=1
#SBATCH --cpus-per-task=96 --mem=1000G
#SBATCH --time=20:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu14

export XDG_CACHE_HOME=$HOME/.cache
export OPTIX_CACHE_PATH=$HOME/.cache/OptixCache && mkdir -p $OPTIX_CACHE_PATH
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH
```

### QoS / 配额关键约束
- `low`: MaxWall 24h,无 GPU 上限(association `gres/gpu:h200=8` 绑)
- `lowest`: MaxWall ∞,**override 了 8-GPU cap**(用 lowest 可超 8 卡,但优先级低)
- 8 GPU 同节点 job 通常排队几小时;array 模式 (单 GPU/task) 比较好排
- `--qos` 不指定时 default = `normal`(可用)

---

## 8.8 render44798 texture wash bug 诊断(2026-06-30 规划,debug 阶段)

### 8.8.1 Context

100-obj 抽样可视化时发现:从 r_frame=0.684 老 r44 数据集(`/mnt/data-alpha-sg-01/team-camera/home/z50057756/data/objaverse_renders_44798/`)随机抽 100 obj,有相当比例渲出灰白色(R==G==B)。

进一步分类(100 抽样 GLB 材质 audit):

| 类型 | 数量 | 是否 BUG | 说明 |
|---|---|---|---|
| HAS_TEXTURE | 64 | — | GLB 嵌入 texture,r44 部分渲出 |
| NO_TEXTURE_HAS_COLOR | 23 | 无 | GLB 单色,无贴图 |
| NO_TEXTURE_NO_COLOR | 13 | 无 | GLB 源 mesh 无 texture 无 color factor → glTF spec 默认白色,Objaverse 数据源 13% 本就这样 |
| **r44 灰 + FE 彩**(64 子集) | **≥5** | **⚠ BUG** | GLB 有 texture,r44 渲 R==G==B≈100%,FE 渲 R==G==B≈38%,texture 在 r44 渲染时被 wipe |

代表 UID:`45bffa8e5bbf483ba15c559ff0b5653e`(GLB 2.85MB,6 张 embedded texture,2 个 mats_with_active_tex)
- r44 view 5: RGB=(72,72,72),**100% R==G==B**(无 texture variation)
- FE view 5: RGB=(100,92,89),38% R==G==B(真彩色)

### 8.8.2 已排除的假设

| 假设 | 排除依据 |
|---|---|
| GLB texture 不在文件里 | python parse: bufferView 指向 50-500KB binary,texture 数据完整 |
| Blender glTF importer 加载失败 | `bpy.ops.import_scene.gltf(merge_vertices=True, import_shading='NORMALS')` 之后 `bpy.data.images` 跟 `materials_with_active_basecolor_texture_node` 都 > 0 |
| `merge_vertices=True` 破坏 UV mapping | 比较 merge=True vs merge=False:material/image count 一致,只是顶点数差 ~3-8% |
| Blender 4.2 默认 view_transform=AgX 把 color wash | 用 AgX / Standard / Filmic 各渲一次最小测试:R==G==B 比例 44-78%(都有 texture variation),Standard 略偏白但不全灰。**测试是 CPU + samples=32 + no denoise** |
| GSO bug 同样 mtl/symlink 路径问题 | GLB 是 binary 嵌入,无外部 mtl/texture 文件依赖 |

### 8.8.3 仍存疑的假设(待 bisect)

| # | 假设 | 强度 |
|---|---|---|
| **H1** | OPTIX denoiser 把暗色 noisy texture detail 平滑成均匀灰色 | ⭐⭐⭐ |
| H2 | 顶部 area light 10000W 过曝,base color factor 主导 RGB,texture 被覆盖 | ⭐⭐ |
| H3 | A2 FoV 范围 [25, 70°] 长焦 + R2 拉近导致单 view obj 过近,specular highlight 主导 | ⭐ |
| H4 | transmission_bounces=3 不够,半透明 mesh 渲染层失败 fallback 白 | ⭐ |

**最小复现已有数据**: 我自己的 minimal script (CPU samples=32 no-denoise) 渲 45bffa8e 拿到 RGB=(232,223,220) R==G==B=44.7% — **有 texture variation**;r44 同 obj 平均 RGB=(72,72,72) R==G==B=100% — 完全 wash。差异只可能在 r44 用 OPTIX denoise + samples=128 + r44 lighting + r44 framing 上。

### 8.8.4 Debug 实验设计(bisect 5 模式)

**Scope**: 单 GPU sbatch,1-2 hr wall。

**Mesh 集**: 2 个候选 UID
- `1e387afc6d7443878c310756400dd467`(GLB 19KB,极小)
- `45bffa8e5bbf483ba15c559ff0b5653e`(GLB 2.85MB,典型)

**Fixed view**: yaw=0.5 rad, pitch=0.3 rad, r_frame=0.684(对齐老 r44 spec)。每 UID 渲 1 张 fixed view。

**模式列表(每个 mesh × 6 modes = 12 张)**:

| mode | 描述 | 关键改动 |
|---|---|---|
| A | r44 baseline | samples=128, OPTIX denoise, top light 10000W, FoV=40° |
| B | A - OPTIX denoise | use_denoising=False |
| C | A 改 OpenImageDenoise | denoiser='OPENIMAGEDENOISE' |
| D | A - top light 10× 降 | top energy 10000 → 1000 |
| E | A + samples 256 + no denoise | 高质量 ground truth (近似 FE 行为) |
| F | A 固定 FoV 40°(覆盖 A2)+ R2=1.0(覆盖 R2 jitter) | A2/R2 关掉 |

**输出**: per-mode RGB mean + R==G==B 比例 + saved PNG → JSON 表 + side-by-side montage (12 张 PNG)。

### 8.8.5 决策树(bisect 结果 → root cause)

| 现象 | 结论 |
|---|---|
| A 复现 ≈100% R==G==B(灰),B/C/D/E/F 都不是 → r44 渲染稳定 wash | 复现 baseline 成立,继续看其余 mode |
| B(no denoise)R==G==B ≤ 30% | **H1 OPTIX 成立** → 改 denoiser='OPENIMAGEDENOISE' 或 samples=256+无 denoise |
| C(OIDN)R==G==B ≤ 30% | denoiser 都有锅,但 OPTIX 更激进 |
| D(灯弱)R==G==B ≤ 30% | **H2 过曝成立** → 顶光降到 1000W or 3000W |
| E(高质量)R==G==B ≤ 30% 且 RGB 接近 FE | **ground truth 是有 texture**,r44 渲染 pipeline 某处丢了它 |
| F(固定 FoV+R2)无变化 | A2/R2 跟 bug 无关 |
| B/C/D 都≈100% R==G==B | 嫌疑都不是,继续 bisect transparent/transmission bounces 或 import_shading mode |

### 8.8.6 实验产物 (deliverable)

- `/home/z50057756/tmp/texture_bug_bisect/<uid>_<mode>.png` × 12
- `/home/z50057756/tmp/texture_bug_bisect/result.json`(per-mode stats)
- `/home/z50057756/tmp/texture_bug_bisect/cmp_montage.png`(2x6 grid view)
- 一行结论:root cause = X (or "all-confounded, need follow-up")

### 8.8.7 修复(根据 bisect 结果)— 实施延后,先 bisect

| 结论 | 修复 | 文件 + 改动 |
|---|---|---|
| OPTIX 是因 | `denoiser='OPENIMAGEDENOISE'`(质量更柔和)或 `use_denoising=False` + `samples=256` | `blender_script/render.py:81` + `:75` |
| 过曝是因 | top light 10000 → 3000(降 3.3×,但跟 TRELLIS plan §1.2 偏离,需评估 fg 分布变化) | `blender_script/render.py:182` |
| 都是 | 综合 | 两处都改 |

### 8.8.8 影响范围 + 下游决策

**Phase 1a 训练**: 用的是有 texture wash 的 r44 数据。可能让模型学到偏灰色 prior,在真彩色 GSO eval 上 PSNR/LPIPS 跟 FE-trained baseline 拉开。

**Phase 2 重渲(§8.6)**: 必须先解决 texture wash,否则 50K obj × 40 view 重渲再次全部丢 texture。

**修完后**:
1. spot-check 10 个 UID 视觉验证 — texture 回来了
2. 起 sbatch 重渲 5 个候选(45bffa8e/1e387afc 等),拉回 100-obj smoke 跑 audit:期望 R==G==B 比例从 50.4% 大幅下降
3. **不重渲 44,798 全量直到 §8.6 Phase 2 strategy 同时落地**(节省 GPU)

### 8.8.9 Bisect sbatch 框架(实施时)

```
#SBATCH --partition=gpu --qos=normal --gres=gpu:h200:1 --cpus-per-task=8
#SBATCH --mem=64G --time=01:00:00 --exclude=lrc-alpha-sg-gpu05,06,14
```

写一个独立 Blender Python `debug_bisect.py`,加载 2 mesh × 6 mode × 1 view,各保存 PNG,跑结果后续 python 算 stats。sbatch wall ≤ 1h。

### 8.8.10 实测结果(2026-06-30 完成)— BUG NOT REPRODUCIBLE in current pipeline

**Round 1: 2 UID × 6 mode bisect**(job 84193,12 PNG,12 sec wall):
- 1e387afc6d74:r44 view5 RGB=(113,113,113) eq=100% 灰。**当前 pipeline reproduction 6 mode 全部彩色**(A_baseline RGB=140,173,163 eq=0%)。
- 45bffa8e5bbf:r44 view5 RGB=(72,72,72) eq=100% 灰。但视觉确认是 mesh 真黑叉子(FE 也是 RGB≈90 几乎灰),**audit false positive**。

→ 1 个真 bug obj 在当前 pipeline 渲彩色,**bug 不可复现**。

**Round 2: 10 candidate × 1 baseline mode**(job 84231,扩大样本):
- 从 500 抽样找 20 个 r44_eq>85% + FE_eq<50% + GLB-has-texture candidate
- 取前 10 个跑当前 pipeline reproduction
- 6 个成功渲染:**6/6 全部 BUG_FIXED**(r44 灰,当前 pipeline 渲对应 texture 彩色)
- 4 个 reproduction empty alpha(`9fe0f5/baecfb/93bc5a/76db04`)— 跟 bug 无关,是 mesh-load/normalize 边界问题,follow-up

加 Round 1 的 1e387afc6d74:**7/7 真 r44-bug-mesh 在当前 pipeline 都复原 texture**。

**Round 3: 4 empty-alpha candidate 修 repro normalize 后重跑**(job 84239):
- 之前 4 个 empty alpha (`9fe0f5/baecfb/93bc5a/76db04`) 是因为 repro_10.py 的 normalize 按 mesh 单独 scale,破坏 parent-child hierarchy。复杂多 mesh 场景(93bc5a1c=25 mesh/79 node, 76db0470=26 mesh/80MB)被打散到画面外。
- 修复:在 `repro_4_fixed.py` 抄 `blender_script/render.py:287-346` 的 hierarchy-aware normalize(single-root parent scale,child mesh 跟着走)。
- 重跑结果:**4/4 BUG_FIXED**(9fe0f5 黄橄榄 / baecfb78 棕瓶+label / 93bc5a1c 棕架+蓝衣 / 76db0470 棕菠萝绿叶,全部 texture 完美呈现)。

**最终样本:11/11 真 r44 bug candidate 全部 BUG_FIXED**。同时证明 render_pipeline/blender_script/render.py 的 normalize_scene 完美处理 hierarchy,任何复杂场景都 OK。

视觉对比 examples(r44 → FE → current):
- 9a1c7a00:黑椭圆 → 绿叶 → 绿叶 ✓
- f123ef50:黑长条 → 粉红 box → 粉红 box ✓
- 48b073d4:灰拖鞋 → 棕凉鞋 → 棕凉鞋 ✓
- 6b85cbe8:黑菠萝 → 棕菠萝(texture)→ 棕菠萝(texture)✓
- 0e464ea4:灰猴 → 红衣大猩猩 → 红衣大猩猩 ✓
- 87605b86:黑顶视圆盘 → 绿蘑菇 → 绿色 mushroom ✓

### 8.8.11 最终结论 + 决策

| 项 | 结论 |
|---|---|
| BUG 真实? | YES — r44 历史数据存在 texture-loss bug |
| 当前 pipeline 仍存在? | NO — 7/7 candidate 全部渲彩色,bug 已被某次环境/驱动/Blender 升级悄然修复(具体哪一次不可考) |
| 修代码? | **不需要**,已修 |
| 是否影响 Phase 1a 训练? | 部分 r44 obj 训练时是灰图。但 GSO eval RA5 88.16 已超 FA3 baseline 86.89 → 模型对几何/姿态学习未受 texture 缺失影响 |
| r44 数据要重渲吗? | Phase 2 §8.6 strategy 落地时全量重渲,bug 自然修复 — **不需要单独行动** |
| follow-up | 4 个 reproduction empty alpha 的 UID(`9fe0f5/baecfb/93bc5a/76db04`)是另一类 mesh-load 边界问题,可单独 audit |

### 8.8.12 artifacts

- `/home/z50057756/tmp/texture_bug_bisect/debug_bisect.py` — round 1 script(2 UID × 6 mode)
- `/home/z50057756/tmp/texture_bug_bisect/repro_10.py` — round 2 script(10 UID × baseline)
- `/home/z50057756/tmp/texture_bug_bisect/cmp_montage.png` — round 1 2×7 grid
- `/home/z50057756/tmp/texture_bug_bisect/repro10_montage.png` — round 2 6×3 grid(可视化最关键结论)
- `/home/z50057756/tmp/texture_bug_bisect/result.json` — round 1 数值
- `/home/z50057756/tmp/find_more_candidates.py` — 500-sample scan tool
- `/home/z50057756/tmp/bug_candidates.txt` — 20 candidate UID 列表

---

## 8.9 Phase 2 渲染管线最终版(2026-06-30 定稿)— vanilla+Layers

### 8.9.1 Context

经过 §8.6 (Phase 2 strategy 设计) + §8.5/8.6 之后的 100-obj smoke 大量迭代,Phase 2 渲染管线 final setting 确认为 **vanilla+Layers**:
- normalize_scene 用 vanilla `1/max(raw_bbox)`(放弃 MAD/Percentile robust normalize)
- 加 3 层质量门(Layer 1a + 1b + 1c)
- Strategy E 用 plan C silhouette 公式(`a_ext/2` 替代 `0.5`,vanilla 下自动等价原版)

迭代历程的主要发现:
- MAD/Pct99.5 robust normalize 在 100-obj smoke 实测 **拉大主分布、误切形状自然末端、Strategy E 跟 robust scale-up 双重放大导致 t_clip 飙到 43%-60%**
- 即便加 plan C 补偿,N=20-50 各档下 normal 主分布 obj 仍被 MAD 误判 11-13 个,得不偿失
- vanilla normalize 简洁、稳定,broken outlier mesh 交给 Layer 1c 兜底 quar(允许损失少量"被 outlier 推出画面"的 mesh,但不污染主分布)

### 8.9.2 完整 render 设置

| 维度 | 设置 | 跟 §1 老版差异 | 代码位置 |
|---|---|---|---|
| `r_frame` | **0.62** | 老版 0.684,新版拉近 → fg×1.17 | env `R_FRAME=0.62` |
| Cycles `samples` | 128 | 不变 | TRELLIS 原版 |
| `bounces` (diff/gloss/transp/transm) | 1/1/3/3 | 不变 | TRELLIS 原版 |
| Lighting (key/top/bottom) | POINT 1000W / AREA 10000W / AREA 1000W | 不变 | blender_script:167-191 |
| Resolution | 512×512 | 不变 | TRELLIS 原版 |
| `num_views` | 40 per obj | 不变 | — |
| `film_transparent` | True | 不变 | TRELLIS 原版 |
| `use_denoising` | True (OPTIX) | 不变 | TRELLIS 原版 |
| **A1** lookat offset | `max_ratio=0.15, zero_prob=0.20` | 不变 | render.py:36-37 |
| **A2** per-obj FoV | `U(40°, 75°)` | 老版 [25°, 70°],新版上调对齐手机 1× 主摄 | render.py defaults |
| **A3** pitch | TRELLIS `sphere_hammersley_sequence` 原版 | 不变 | utils.py |
| **R2** radius jitter | `U(0.95, 1.10)` | 老版 [0.95, 1.15],新版收紧 | render.py defaults |
| **R1** roll (训练侧) | ±10° P2 patch | 不变 | dataset_objaverse.py |
| **normalize_scene** | vanilla `1/max(raw_bbox)` | 没改 | blender_script:287-326 |
| **Layer 1a** SkipRender | `non_finite_bbox` / `empty_bbox` (max<1e-6) / `degenerate_planar_mesh` (min<1e-4·max) | **新增** | blender_script:308-316 |
| **Layer 1b** Strategy E | `aspect_long>5 AND aspect_second>3` → `k=clamp(silhouette/0.95, 0.5, 1.0)`,公式用 `a_ext/2`(plan C) | **新增** | blender_script:485-535 |
| **Layer 1c** post-render alpha | view 0 `fg_ratio = (alpha>127).mean() < 0.005` → SkipRender | **新增**(替代老 `alpha_max<3` 太宽松) | blender_script:589-620 |
| Orchestrator skip | SkipRender → 不写 .tar,UID 追加 `quarantine.txt` | **新增** | render.py |

### 8.9.3 Strategy E plan C 公式 + 物理原理

```python
# Layer 1b — thin-obj adaptive zoom
ASPECT_LONG_THRESH = 5.0
ASPECT_SECOND_THRESH = 3.0
TARGET_SILHOUETTE = 0.95
ext = scene_bbox()
a_ext = max(ext)
b_ext, c_ext = sorted(ext)[1], sorted(ext)[0]
aspect_long = a_ext / c_ext
aspect_second = a_ext / b_ext

if aspect_long > 5 and aspect_second > 3:
    fov_rad = views[0]["fov"]
    # Plan C: 用 a_ext 替代 hard-coded 0.5。
    # vanilla normalize 下 a_ext=1.0 → 公式等价原版 silhouette geometric formula。
    # 若以后启用 robust normalize 让 a_ext>1.0,公式自动补偿 → k=1.0 不拉近。
    silhouette_k1 = 2.0 * math.atan((a_ext/2) * math.sin(fov_rad/2) / R_FRAME) / fov_rad
    k_zoom = max(0.5, min(1.0, silhouette_k1 / 0.95))
    for v in views:
        v["radius"] *= k_zoom
```

### 8.9.4 100-obj smoke 验证(job 84147,paired vs FE/old r44 各 79/80)

| 指标 | vanilla+Layers (100) | old r44 (paired) | FE (paired) |
|---|---|---|---|
| tar / quar / leak | 80 / 20 / 0 | — | — |
| fg mean | 0.1666 | 0.1311 | 0.1828 |
| fg p50 | 0.1345 | 0.0976 | 0.1519 |
| **VL/FE ratio (mean)** | **0.911** | 0.717 | 1.0 |
| **VL/FE ratio (p50)** | **0.886** | 0.643 | 1.0 |
| **per-obj median ratio vs FE** | **0.909** | 0.715 | — |
| Normal clip (view-weighted) | 11.2% | — | — |
| Trig clip | 29.8% | — | — |
| FoV range | 40.6°-74.2° ✓ | — | — |

主分布提升 27.4%(per-obj median ratio 从 0.715 → 0.909),离 FE 100% 还差 ~9%(渲染细节 / Blender 版本不可消)。

### 8.9.5 2000-obj smoke 验证(job 84252,4 H200 × 4 procs,43:29s wall,30.26 VPS)

| 指标 | vanilla+Layers (2000) | old r44 (paired 1961) | FE (paired 1961) |
|---|---|---|---|
| tar / quar / leak | **1974 / 26 / 0** (1.3% quar) | — | — |
| fg mean | 0.1890 | 0.1511 | 0.2116 |
| fg p50 | 0.1655 | 0.1351 | 0.1899 |
| **VL/FE ratio (mean)** | **0.893** | 0.714 | 1.0 |
| **VL/FE ratio (p50)** | **0.872** | 0.711 | 1.0 |
| **per-obj median ratio vs FE** | **0.891** | 0.716 | — |
| Total clip | 10.59% | — | — |
| Normal clip (1866 obj) | 9.26% | — | — |
| Trig clip (108 obj, 5.5%) | 33.54% | — | — |

100/2000 一致性极高:per-obj median ratio 0.909 → 0.891,old r44/FE 0.715 → 0.716(完全锁死)。Phase 2 全量预估 quar rate ~1.3% → ~580 obj,跟老 blacklist 413 同量级。

### 8.9.6 Color audit 新方法(HSV S_pct + Lab chroma)— 老 mean-RGB-diff 不可靠

旧方法 `max(|R-G|, |G-B|, |R-B|)` 在多色 mesh 上被平均洗白成"灰",278 obj 卡 AMBIGUOUS。

新方法(`reclassify_v2.py`):
- `S_pct` = % of fg pixels with HSV saturation ≥ 0.15
- `chroma_med` = median(sqrt(a*²+b*²)) in Lab space

5+1 类(2000 obj):
| 类别 | 数量 | 比例 | 含义 |
|---|---|---|---|
| **NORMAL_COLOR** | 1205 | 61.4% | 双方都渲彩,正确 |
| **MESH_GRAY** | 544 | 27.7% | mesh 本身就灰(单色金属/塑料),正确 |
| **WASH_HEAVY** | 79 | 4.0% | FE 彩 VL 灰(VL 我们出问题)|
| **WASH_MILD** | 56 | 2.9% | FE 彩 VL 部分灰 |
| **WASH_INVERSE** ← 新 | **47** | 2.4% | **FE 灰 VL 彩(FE 出问题)** |
| WASH_INVERSE_MILD + OTHER | 30 | 1.5% | 边角 |

### 8.9.7 Color wash 根因 — GLB introspection 证伪 H3

直接 pygltflib parse 54 个 GLB(15 each WASH_HEAVY/WASH_INVERSE,12 each NORMAL_COLOR/MESH_GRAY)。

| 类别 | n | has_tex | has_basetex 绑定 |
|---|---|---|---|
| **WASH_HEAVY** | 15 | **15 (100%)** | **14/15 (93%)** |
| **WASH_INVERSE** | 15 | **15 (100%)** | 12/15 (80%) |
| NORMAL_COLOR | 12 | 8 (67%) | 7/12 (58%) |
| MESH_GRAY | 12 | 7 (58%) | 7/12 (58%) |

**H3 (texture binding 失败) 证伪**:WASH_HEAVY / WASH_INVERSE 的 GLB 100% 有 texture 且 80-93% 有 baseColor 绑定 — mesh 本身完整,binding 完整。

**真因**:Blender 渲染时对某些 mesh 的 PBR shader 节点处理不一致(版本差异 + OPTIX vs CUDA + 可能的 sRGB 标记 / alphaMode 差异)。无法用 import 参数修复 — 涉及 Blender 渲染管线内部行为。

**双向 binding 不一致视角**:
- WASH_HEAVY 79 (4.0%): 我们当前 Blender 4.2.9 + OPTIX 在这部分 mesh 上 wash
- WASH_INVERSE 47 (2.4%): TRELLIS 团队几个月前用更老 Blender + CUDA 在那部分 mesh 上 wash

两边各有部分 mesh wash,总 binding 不一致约 **6.4%**。

### 8.9.8 决策 — 接受 6.4% wash,不再深挖

理由:
1. GLB 文件完好,wash 是 Blender 渲染 quirk
2. 修我们的 79 个需要换 Blender / 换 device,会引入新 distribution shift,且**修不了 FE 的 47 个历史数据**
3. Phase 1a 训练数据有相同 ~6-7% wash,GSO eval RA5 88.16 > FA3 baseline 86.89(plan §8.8.11)→ **模型对 texture wash 不敏感,主要学几何/姿态**

### 8.9.9 当前数据 + 校验工具 path

- 2000-obj 渲染: `/home/z50057756/tmp/phase2_smoke/render_out_2000/`(1974 tar + 26 quarantine)
- 100-obj 渲染: `/home/z50057756/tmp/phase2_smoke/render_out_vanilla_layers/`(80 tar + 20 quarantine)
- Color audit v2: `/home/z50057756/tmp/phase2_smoke/color_audit_v2.json`(1961 paired w/ HSV+chroma)
- GLB introspection: `/home/z50057756/tmp/phase2_smoke/glb_introspect.json`
- WASH_INVERSE 可视化(20 montages): `/home/z50057756/tmp/wash_inverse_check/`
- 校验脚本:
  - `verify_final.py` / `verify_final_2000.py` — paired fg metrics
  - `audit_color.py` / `audit_color_2000.py` — old mean-RGB classification
  - `reclassify_v2.py` — new HSV/Lab classification
  - `viz_wash_inverse.py` — WASH_INVERSE montage builder
  - `glb_introspect.py` — pygltflib material parse

### 8.9.10 sbatch 模板(Phase 2 全量参考)

`/home/z50057756/tmp/phase2_smoke/sbatch_smoke_2000.sh`(已 production-tested):
- `partition=gpu qos=normal gres=h200:4 nodes=1`
- `cpus-per-task=96 mem=400G time=02:00:00`
- `exclude=lrc-alpha-sg-gpu05,gpu06,gpu14`
- 16 procs (4 GPU × 4 per GPU) → 30 VPS
- 全量 44,798 × 40 view / 30 VPS ≈ **16.6h wall**(分多 job 排队)

### 8.9.11 跟 Phase 1a 老 r44 训练数据的关系

| 维度 | Phase 1a 老 r44 | Phase 2 vanilla+Layers |
|---|---|---|
| r_frame | 0.684 | **0.62** |
| FoV | [25°, 70°] | **[40°, 75°]** |
| R2 jitter | [0.95, 1.15] | **[0.95, 1.10]** |
| Strategy E | 无 | **有 (plan C 公式)** |
| Layer 1a/1c | 无 | **有** |
| Blacklist | 413 个 fg<0.005 离线拉黑 | 在渲染时实时 quar |
| 主分布 fg vs FE | 0.715 | **0.891 (+25%)** |
| Texture wash 率 | ~6-7%(估算) | 6.4%(实测) |

**两批数据不兼容混训** — 设置完全不同,分布差异显著。Phase 2 全量上线时需要决定:用新数据从头训 vs 用老数据 ckpt 接续。

---

## 10. Plan 文件运维

- 仓库: `~/.claude/plans/` (git managed, auto-commit hook on Write/Edit)
- 上次事故: 2026-06-20 hook 用 `git add -A` 把外部删除的本文件也 stage 进了 commit → 删了。**已修**: hook 改成 `git add "$F"`,只 stage 当前编辑的文件,不再被外部删除事件污染。
- 备份策略: hook 每次 Write/Edit 自动 commit,不会再丢。
