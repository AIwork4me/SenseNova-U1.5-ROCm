# 移植笔记：SenseNova-U1.5 在 AMD GPU 上（ROCm）

umbrella 仓库如何在 AMD GPU 上运行 SenseNova-U1.5-8B-MoT，以及它为什么
保持很薄。硬件相关的深入内容在 [docs/hardware/](hardware/) 的各硬件
profile 页；证据优先的结论在 [docs/results/](results/)。

**一句话结论：上游代码本身就是 ROCm 就绪的**（仅一处硬件无关的
interleave 热修复例外——见 §6）。NEO-unify 参考实现基于 `torch.cuda`
命名空间（ROCm PyTorch 原生提供），且在 flash-attn 缺失时自动回退
PyTorch SDPA。真正的工作只有两件：(1) 拿到包含你的 gfx 目标内核的
PyTorch ROCm wheel；(2) 把 ~50 GB 的 bf16 权重装进你的 GPU 内存预算。
这正是 `scripts/01-setup-venv.sh` 和 `--vram_mode` 参数所处理的。

## 1. "ROCm 支持"在 torch 里意味着什么

ROCm 版 PyTorch 把 GPU 重新暴露为 `cuda` 设备类型：

- `torch.cuda.is_available()` → `True`
- `device="cuda"` / `.to("cuda")` → 跑在 AMD GPU 上
- `torch.version.hip` → wheel 构建所用的 ROCm 版本

所以只使用 `torch.cuda` API 的代码（如 SenseNova-U1 的
`src/sensenova_u1/utils/accel.py`，其文档字符串明确写着
"CUDA (incl. ROCm via the cuda namespace)"）**无需任何修改**即可运行。

唯一不能自动继承的是编译内核覆盖：每个 wheel 只包含它编译时指定的
gfx 架构内核。`scripts/rocm_check.py` 会在你浪费数小时调试模型深处的
`HIP error` 之前，先验证 wheel 是否真的带有你 GPU 的内核。

## 2. 注意力：用 SDPA 替代 flash-attn

上游优先使用 flash-attn（`--attn_backend auto` → 可导入即用）。多数
ROCm gfx 目标（包括 gfx1151）没有 flash-attn 构建。于是上游的 `auto`
在 ROCm 上会选择 **PyTorch SDPA**；本仓库的运行脚本显式固定
`--attn_backend sdpa`，保证行为可复现。

ROCm 上 SDPA 分发到 AOTriton flash、MIOpen 融合内核或 math 回退——
对本模型用到的形状都能给出正确的 bf16 注意力。

后端选择与确定性说明：见各硬件 profile 页。

## 3. 50 GB 权重如何装进更小的内存

bf16 权重约 50 GB。两套互补机制：

### a) 统一内存 GTT 溢出（APU）

统一内存 GTT 溢出（APU）：见
[docs/hardware/strix-halo/README.md](hardware/strix-halo/README.md)。

### b) 分层 layer offload（任何 GPU）

上游自带分层 offload 引擎（`sensenova_u1/utils/layer_offload.py`），
通过 `--vram_mode` 暴露：

| vram_mode | 策略 | 适用 GPU 内存 |
|---|---|---|
| `full` | 全部驻留 GPU | 需 ~52+ GiB 独立显存；APU 上可计入 VRAM+GTT 池——见 [strix-halo profile](hardware/strix-halo/README.md) |
| `fast` | 多数层驻留；生成层按步换入 | ~24 GB |
| `balanced` | 更重的 offload + 异步预取 | ~16-24 GB |
| `low` | 最激进 offload | ~12-16 GB |

本仓库的 `scripts/run-task.sh` 会自动注入 `--vram_mode`，默认
`balanced`；用 `VRAM_MODE` 环境变量（或自行传 `--vram_mode`）切换档位。

## 4. PyTorch wheel 矩阵（已知 ROCm wheel 来源）

| 来源 | 索引 | 覆盖 |
|---|---|---|
| **AMD 官方多架构** | `https://repo.amd.com/rocm/whl-multi-arch/`（`torch 2.12.0+rocm7.14.0`） | 多 gfx 家族 wheel；gfx1100 与 gfx1151 均在覆盖内——gfx1151 无需 HSA 预加载（verified on: gfx1151；见 [strix-halo profile](hardware/strix-halo/README.md)） |
| 官方 PyTorch ROCm 7 | `https://download.pytorch.org/whl/rocm7.0`（`torch 2.10.0+rocm7.0`） | gfx942、gfx950、gfx1100/1101、gfx1151、gfx1200 —— gfx1151 需 HSA 预加载（见 [strix-halo profile](hardware/strix-halo/README.md)） |
| AMD 按架构 nightly | `https://rocm.nightlies.amd.com/v2/<gfx>/` | 官方 wheel 缺架构时的回退 |

`scripts/01-setup-venv.sh` 从单一索引安装（`PY_INDEX_ROCM` 可覆盖；
默认为官方 PyTorch rocm6.3 索引，即 `torch 2.8.0+rocm6.3`——本仓库在
gfx1100 上验证的栈），安装后跑 GPU 冒烟测试。`scripts/rocm_check.py`
是更深入的检查，验证 wheel 是否带有你 gfx 目标的内核。已验证的栈连同
凭据记录在 [docs/results/](results/)。

## 5. 一个真实的坑：wheel 自带的 HSA 运行时（gfx1151）

该坑仅影响 gfx1151——见
[docs/hardware/strix-halo/README.md](hardware/strix-halo/README.md)。

## 6. 唯一的上游修改：interleave 热修复

模型代码**近乎零修改**，仅一处上游 bug 修复例外——改动 10 个
`_t2i_predict_v` 调用点，以补丁形式携带
（[patches/0001-interleave-pass-image_size-to-_t2i_predict_v.patch](../patches/0001-interleave-pass-image_size-to-_t2i_predict_v.patch)）：

- 像素头启用时（发布权重 `use_pixel_head=true`）`_t2i_predict_v()` 会
  解引用 `image_size`，但 interleave 生成循环（`interleave_gen`、
  `interleave_gen_image_only`）从不传它 → 首张生成图前即
  `TypeError: 'NoneType' object is not subscriptable`。`t2i_generate`
  一直传参正确。
- 该 bug **与硬件无关**（CUDA 上同样存在这条代码路径；在 gfx1100 与
  gfx1151 上独立复现）——见上游
  [PR #260](https://github.com/OpenSenseNova/SenseNova-U1/pull/260)
  （开放中，2026-08-26 重定向至 `main`；本地留存：
  [docs/upstream/pr-interleave-image-size.md](upstream/pr-interleave-image-size.md)），
  其中也带有一次独立跨平台确认。上游 `main` 已通过 `2f42002` 吸收了
  10 处调用点的修复；该 PR 现在只补充逐图进度条那一行。本仓库的
  0001（建模文件与该 PR 逐字节一致）在固定 checkout
  `feat/u1.5@76c32c2` 越过 `2f42002` 之前仍然必需。
- `scripts/01-setup-venv.sh` 在克隆固定 commit 的上游后自动应用补丁
  （`git apply`，幂等——见
  [patches/README.md](../patches/README.md)）。

一旦固定 commit 越过 `2f42002`，即删除补丁文件与对应安装步骤。

## 7. 本仓库相对上游改了什么

模型代码：**近乎零修改**——共三个小补丁，全部记录在
[patches/README.md](../patches/README.md)：§6 的 interleave 热修复
（0001）、面向融合 SDPA wheel 损坏的安装的 math 回退（0002，gfx1100
叶 wheel）、可选的 torch.compile/cudagraph 加速（0003，默认关闭）。
`scripts/01-setup-venv.sh` 做的是：

1. 自行安装 ROCm PyTorch（默认官方 PyTorch rocm6.3 索引，
   `PY_INDEX_ROCM` 可覆盖——见 §4）——上游钉死 `torch==2.8.0` 于
   cu128 索引，是 NVIDIA 构建，用 `pip install -e ... --no-deps`
   绕开该钉死；
2. 从 git 以**固定 commit** 安装上游，保证可复现，并应用上述补丁；
3. 用 ROCm 验证过的默认值封装上游四个示例 CLI（`t2i`、`editing`、
   `vqa`、`interleave`，见
   [scripts/run-task.sh](../scripts/run-task.sh)）。

若上游某天原生提供 ROCm wheel，本仓库就收缩为纯验证套件——这是刻意
设计。
