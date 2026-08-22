# SenseNova-U1.5-ROCm（中文说明）

[English](README.md) · Apache-2.0

**一条命令，在 AMD Radeon GPU（ROCm）上本地运行 SenseNova-U1.5-8B-MoT——
原生统一多模态大模型（一个模型同时做"看图问答"和"文生图/改图"）。
所有结论附带可复现的证据收据，每个数字都能溯源。**

- ✅ 在 **gfx1100（48GB 显存）/ ROCm 7.2.1** 上全量验证四类任务：
  文生图（t2i）、图像编辑（edit）、视觉理解（VQA）、图文交错生成（interleave），
  含 `--think` 推理模式
- ✅ **48GB 显卡跑 50.2GB 的 bf16 全量模型**：走上游自带的分层卸载
  （`--vram_mode`），权衡全部实测
- ✅ **模型权重逐文件 SHA256 校验**（24 个文件对照清单）
- ✅ **该定的地方确定**：同种子两次生成逐字节一致（sha256 证明）
- 🔧 **附赠社区贡献**：定位并绕过了 PyTorch 官方 rocm6.3 轮子在 gfx1100 上
  三处数学库原生崩溃（rocBLAS Tensile 段错误 / 7.x 数据格式不兼容 / MIOpen JIT
  编译器段错误），完整根因分析见
  [findings 文档](docs/results/findings/rocm63-wheel-blas-on-gfx1100.md)

```bash
bash scripts/quickstart.sh          # 可选任务: t2i | vqa | edit | interleave
```

## 效果展示：电影海报风格

以下海报由本仓库管线在 gfx1100 参考机上生成
（`bash scripts/run-task.sh t2i --jsonl examples/posters-2026-08.jsonl …`，
1664×2496 训练桶，50 步，每张约 5 分钟，13.4 img-tok/s，
[运行日志](docs/results/logs/posters.log)）。题材取自 2026 年 8 月热门影片；
这些是**原创 AI 生成作品**（海报风格），非官方海报、无演员肖像、
无厂标、与影片方无关。

| 《八仙！》动画奇幻 | THE ODYSSEY 史诗神话 |
|:---:|:---:|
| ![八仙！](docs/results/gallery/posters/baxian.webp) | ![The Odyssey](docs/results/gallery/posters/the-odyssey.webp) |

| 《功夫女足》运动喜剧 | 《欢迎来龙餐馆》奇幻喜剧 |
|:---:|:---:|
| ![功夫女足](docs/results/gallery/posters/kungfu-girls.webp) | ![欢迎来龙餐馆](docs/results/gallery/posters/dragon-restaurant.webp) |

主标题文字经逐字校验（含两个中文标题）；功夫女足一张为移除球衣上
类品牌标识而重生成并复验通过。全部提示词与种子在
[`examples/posters-2026-08.jsonl`](examples/posters-2026-08.jsonl)，可复现：

```bash
bash scripts/run-task.sh t2i --jsonl examples/posters-2026-08.jsonl \
    --output_dir outputs/posters/ --cfg_scale 4.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 --profile
```

## 一键上手

前置条件：Linux + AMD GPU（RDNA3 级，参考机 gfx1100）+ ROCm 用户态 +
约 62GiB 空闲磁盘 + ≥64GiB 内存（卸载路径需要把 47GiB 权重放在内存里）。

```bash
git clone <本仓库> && cd SenseNova-U1.5-ROCm
bash scripts/quickstart.sh                 # 首次运行自动装环境(~10GiB)+下模型(~50GiB)
bash scripts/quickstart.sh vqa             # 看图问答
bash scripts/quickstart.sh edit            # 改图
bash scripts/quickstart.sh interleave      # 图文交错教程
```

每个阶段幂等可断点续跑。之后的日常使用：

```bash
bash scripts/run-task.sh t2i --prompt "山间日出的小村庄" --width 2048 --height 2048 --seed 42
bash scripts/run-task.sh vqa --image 你的图片.jpg --question "图里有什么？"
```

## 实测数据（gfx1100 / ROCm 7.2.1 / torch 2.8.0+rocm6.3）

环境指纹：[docs/results/environment.json](docs/results/environment.json)。
以下数字全部来自收据（`docs/results/validation/*.json`），
`bash scripts/validate.sh` 可完整复现：

| 任务 | 参数 | 端到端耗时 | 峰值显存 |
|---|---|---|---|
| VQA 视觉理解（greedy，1.6万 patch 菜单图） | ≤768 新 token | 602 s | 24.8 GiB |
| 文生图 | 2048×2048 @ 50 步，seed 42 | 420 s（约 7 s/步） | 22.3 GiB |
| 文生图 + 推理模式 | 同上 + `--think` | 547 s | 22.3 GiB |
| 图像编辑 | 2048 级 @ 50 步 | 484 s | 29.8 GiB |
| 图文交错教程 | 7 张 2048×1152 + 交错文本 | 3392 s | 47.7 GiB |

模型加载（页缓存热）约 66 秒。**确定性**：同种子两次生成 sha256 逐字节一致。
**vram_mode 对比**（10 步探针）：balanced 200.5s/22.3GiB ·
fast 199.2s/22.3GiB · low 208.4s/**3.4GiB**——`low` 只慢 4% 却省 15 倍显存，
小显存卡的福音。

生成样图（含确定性双图对比）：[gallery](docs/results/gallery/README.md)。

## 显存策略（为什么 `--vram_mode` 是关键）

模型 bf16 权重 50.23GB > 参考卡 48GB 显存——权重都放不下，`full` 模式必然
OOM。上游为此内置了分层卸载：从内存经 PCIe 异步预取层权重。
默认 `balanced`（异步预取）；`fast` 会把生成层常驻显存（批量出图更优）；
`low` 同步换入换出（显存最省、最慢）。

## 常见问题

- **`/dev/kfd` 不可写**：`sudo usermod -aG render,video $USER` 后重新登录。
- **加载很慢/像卡住**：首次从磁盘读 50GiB 权重 + 内核预热，看 `rocm-smi`
  等几分钟；页面缓存热了之后加载约 66 秒。
- **显存不够（OOM）**：别用 `VRAM_MODE=full`；默认 `balanced` 即为 48GB 卡调校。
- **`torch.cuda.is_available()` 为假**：检查 ROCm 安装与 `rocminfo` 是否列出 gfx 设备。

更多见 [getting-started](docs/getting-started.md) 与
[英文 README](README.md)。

## 项目结构

| 脚本 | 作用 |
|---|---|
| `scripts/00-check-env.sh` | 检查 ROCm / GPU / 磁盘 / 内存 |
| `scripts/01-setup-venv.sh` | 建 venv（ROCm torch 2.8.0 + 上游推理栈 + GPU 冒烟测试） |
| `scripts/02-fetch-model.sh` | 下载并 SHA256 校验模型（断点续传） |
| `scripts/run-task.sh` | 四任务分发器（注入验证过的默认参数） |
| `scripts/quickstart.sh` | 一键命令（环境→venv→模型→生成） |
| `scripts/validate.sh` | 全量验证套件，产出收据 |
| `scripts/summarize_results.py` | 从收据生成 README 证据表 |

## 致谢

- [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)：
  模型与本项目封装的推理代码（锁定 `76c32c2`），以及让 48GB 显卡跑起来的
  分层卸载机制。
- AMD ROCm 与 PyTorch ROCm 轮子团队。

## 许可

本仓库 Apache-2.0；上游 Apache-2.0；模型权重遵循
[其许可](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)。
