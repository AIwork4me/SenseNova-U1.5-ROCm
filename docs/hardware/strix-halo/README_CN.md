# 硬件 Profile：Strix Halo（Radeon 8060S 核显，gfx1151）

verified on: gfx1151 —— Ryzen AI MAX+ PRO 395，系统 ROCm 7.2.1 +
torch 2.12.0+rocm7.14.0 wheel。完整证据集（V1–V8 提速、质量评测、
确定性、档位测试）：
[SenseNova-U1.5-ROCm-8060S](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S)。
gfx1100 凭据在本仓库 [docs/results/](../../results/)。
通用背景见[移植笔记](../../porting_CN.md)。

## 为什么"32 GB"核显装得下完整 bf16 —— GTT 溢出

AMD APU 的 GPU 与 CPU 共享物理内存。驱动暴露两个池：专用 **VRAM**
（BIOS 挖出，参考机型为 32 GB）与 **GTT**（GPU 可映射的其余系统内存，
如 80 GB）。ROCm 分配器越过专用 VRAM 时会溢出到 GTT，因此
`torch.empty(50 GiB, device="cuda")` 能成功。两个池都是同一颗封装内
的 LPDDR5X 物理内存，代价不大——8060S 仓库套件运行期间，分配器把
~1.2 GiB 放进 VRAM、最多 41.6 GiB 进 GTT（GPU 可寻址合计 112 GiB），
full 档峰值 ≤ 42.9 GiB，实践上 ~45+ GiB 可寻址即可。

本仓库的 `scripts/rocm_check.py --alloc-test 48` 正是验证这一溢出
（48 GiB 分配；8060S 仓库存证 `evidence/stack714/env_check.json`）。
这就是"32 GB 显存"也能跑 `--vram_mode full`（约 50 GB 权重全驻留
GPU、不做 layer offload）的原因。7.14 栈上选档提示：`full` 同时也是
最快档——不要为提速改选 `fast`/`balanced`（8060S PERFORMANCE.md
2026-08-25 更正；档位仍是小显存 GPU 的容量工具）。

## HSA 运行时的坑（torch +rocm7.0 wheel，驱动 ≥ 7.1）

官方 `torch 2.10.0+rocm7.0` 轮子自带 ROCm 7.0 的 HSA 运行时，在
gfx1151（Strix Halo）+ ROCm ≥ 7.1 内核驱动上冻结首个 code object 时
（`GpuAgent::QueueCreate` → `ReleaseQueueMainScratch`）段错误——任何
GPU 算子都跑不起来。解法：预加载系统 ROCm 运行时
（`LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so`），它必然与内核驱动
匹配。

本 profile 以 [hsa_fix.sh](hsa_fix.sh) 携带该 shim——source（而非
执行）本文件后调用 `hsa_fix_apply "$PY" "$REPO_ROOT"`；仅当系统运行时
比 wheel 自带的**更新**时才 export `LD_PRELOAD`，自带运行时本来就
正常（或更新）的机器不受影响。推荐的 7.14 wheel 裸跑，不会触发。

## gfx1151 wheel 指南

- **默认（推荐）**：AMD 官方多架构 `torch 2.12.0+rocm7.14.0`，
  索引 `https://repo.amd.com/rocm/whl-multi-arch/`——gfx1151 实测
  裸跑（无需 HSA 预加载），且在 8060S A/B 中快于 torch-2.10 构建
  （1024×1024 t2i fast 档 167.9 s vs 232.1 s——当年会话存在争用，
  幅度仅供参考）。这类 wheel 自带 ROCm 用户态——系统 ROCm 主要提供
  内核驱动工具（`rocminfo`、`rocm-smi`）。
- **回退**：官方 PyTorch rocm7.0 索引
  （`https://download.pytorch.org/whl/rocm7.0`，`torch 2.10.0+rocm7.0`；
  覆盖 gfx942/950/1100/1101/1151/1200）——gfx1151 上需上面的 HSA
  预加载。
- **兜底**：AMD 按架构 nightly
  （`https://rocm.nightlies.amd.com/v2/<gfx>/`），用于官方 wheel 缺
  架构时。

本仓库的 `scripts/01-setup-venv.sh` 从单一索引安装——用
`PY_INDEX_ROCM` 指向你想要的来源；更深入的"wheel 是否带你 gfx 内核"
检查是 `scripts/rocm_check.py`
（见[移植笔记 §4](../../porting_CN.md#4-pytorch-wheel-矩阵已知-rocm-wheel-来源)）。

## AOTriton 告警（hd=72 ViT 头）

gfx1151 + torch 2.12+rocm7.14 下，AOTriton flash SDPA 在 head_dim=72
且 seq>1024、**仅非因果注意力**时**静默输出错误值**（同形状因果注意力
正常）——误差逐次漂移，典型相对误差 16~730%，以有限错误值为主、约
1/5 次调用出现 NaN。SenseNova 自身的注意力形状（head_dim 64/128）
不受影响；但若复用环境跑 16 头/hidden 1152 的外部 ViT 模型，请对其
关闭 AOTriton。根因：上游 AOTriton
[issue #54](https://github.com/ROCm/aotriton/issues/54)（fwd 内核缺
`out_dtype=tl.float32`）；修复 commit `8232d69672` 已存在但至今未合并
进任何发布——追踪详情见
[docs/upstream/aotriton-54.md](../../upstream/aotriton-54.md)。

## 配套仓库

日常驱动脚本、一条命令 quickstart 与完整证据集都在
[8060S 仓库](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S)
——即 Strix Halo 实验室，上述数字都在那里测得。实验在那里达到毕业
标准（graduation criteria，
[CONTRIBUTING.md](../../../CONTRIBUTING.md)）后，通过 PR 毕业进入
本 umbrella 仓库。
