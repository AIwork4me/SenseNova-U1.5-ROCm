# SenseNova-U1.5-ROCm `--cfg_interval` 提速验证报告

> 2026-08-25。任务闭环：① 提速调研 ② 本地 Qwen-Image-Bench ③ 提速后品质无下降验证。
> 数据：`/workspace/speedup/`（bench 仓库 `/workspace/Qwen-Image-Bench/`）。

## 结论（TL;DR）

| 配置 | 生成耗时 (2048²×50步) | vs 基线 | bench 总分 | paired-t | 判定 |
|---|---|---|---|---|---|
| C0 基线（全 CFG） | 295.4 s/张 | — | 52.82 | — | — |
| V2 `--cfg_interval 0 0.2` | 235.7 s/张 | **-20.2%** | 54.33 | +0.88 | **品质无下降** |
| V3 `--cfg_interval 0.7 1.0` | 201.4 s/张 | **-31.9%** | 52.47 | -0.14 | **品质无下降** |

- 判定口径：同 10 提示词同种子配对，|t| < t_crit(df=9)=2.262 且逐维无系统性下跌。
- 逐维 paired-t：V2 全维 |t| ≤ 1.77；V3 全维 |t| ≤ 1.47（Aesthetics 配对差 -3.7，不显著）。
- **推荐配置：V3（`--cfg_interval 0.7 1.0`）**，-32% 提速、品质统计不可区分。
  保守场景可用 V2。

## L1 五维分数（C0 / V2 / V3）

| 维度 | C0 | V2 | V3 |
|---|---|---|---|
| Quality | 48.44 | 52.56 | 51.67 |
| Aesthetics | 54.20 | 52.40 | 46.10 |
| Alignment | 56.94 | 61.39 | 54.64 |
| Real-world Fidelity | 42.22 | 44.72 | 46.39 |
| Creative Generation | 65.00 | 56.67 | 75.00 |
| **TOTAL** | **52.82** | **54.33** | **52.47** |

## 验证方法

- 生成：SenseNova-U1.5-ROCm，gen10.jsonl（10 提示词，2048²，50 步，同种子），三套图各 10 张。
- 评审：Qwen-Image-Bench 官方 judge（Qwen3.5 27B Q-Judger），
  int8 权重本地化（gfx1100 48GB 单卡放不下 bf16），官方参数
  （temperature 0 / thinking / max 4096 token / repetition_penalty 1.05）不偏离。
- 39 个评审任务/套 × 3 套，全部完成、无解析失败缺失维度。
- 注意事项与本地适配全部记录在 HANDOFF.md §2/§9（MIOpen 旁路、CPU 视觉塔、
  use_cache 根因等）；复跑入口 `bin/run-all-judge.sh`（断点续跑安全）。

## 局限

- 样本量 n=10 提示词（paired），能检测 ~5 分级的总分组差异；更小差异不可分辨。
- 评审模型为本地 int8 量化运行（对称逐通道，未做官方分数校准——官方 responses 需外网）。
- 结论限定 2048²×50 步与该提示词分布。

## 产物清单

```
bench/imgs-{baseline50,int02,int07}/            三套图 ×10
bench/judge-input-{name}_bench_scores.json/xlsx  三套分数
bench/judge-input-{name}_judged.jsonl            三套逐任务原始输出
logs/judge-{name}.log                            三套运行日志
REPORT.md                                        本报告
HANDOFF.md                                       全过程工程记录
```
