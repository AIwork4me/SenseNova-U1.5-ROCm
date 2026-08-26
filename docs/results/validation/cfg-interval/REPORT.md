# SenseNova-U1.5-ROCm `--cfg_interval` 提速验证报告

> 2026-08-26 终版（扩样复判后）。任务闭环：① 提速调研 ② 本地 Qwen-Image-Bench
> ③ 提速后品质验证（n=10 初判 + n=36 扩样复判，每步独立验核）。
> 数据：`/workspace/speedup/`（bench 仓库 `/workspace/Qwen-Image-Bench/`）。

## 结论（TL;DR）——扩样复判修正了初判

| 配置 | 生成耗时 | vs 基线 | n=10 初判 | **n=36 复判（终版）** | 终版判定 |
|---|---|---|---|---|---|
| C0 基线（全 CFG） | 295.9 s/张 | — | 52.82 | 49.59 | — |
| V2 `--cfg_interval 0 0.2` | 234.8 s/张 | **-20.6%** | t=+0.88 无偏移 | **+0.34 ± 2.57，t=+0.26**（全维不显著） | **✅ 推荐配置** |
| V3 `--cfg_interval 0.7 1.0` | 200.0 s/张 | -32.4% | t=-0.14 无偏移 | **-6.06 ± 3.85，t=-3.19，p=0.003**（Quality/Aesthetics 双显著） | **❌ 撤回：有品质代价** |

- **推荐配置：V2（`--cfg_interval 0 0.2`）**——-20.6% 提速，n=36 配对下总分差 CI 收窄至 ±2.57 且含零，五维全部不显著。
- **V3 的"品质无下降"结论被扩样推翻**：n=10 时 V3 总分差 CI ±5.5，真实退化藏得住；n=36 后信号显形（-6.06，p=0.003；Aesthetics -10.8、Quality -6.5，均显著；多条 prompt 掉 15-30 分）。初判样本恰好未覆盖受损的内容类型——这正是扩样复判的价值所在。
- 两档提速幅度本身在 n=36 上复现（-20.64%/-32.41%，与初判 -20.2%/-31.9% 一致）。

## n=36 逐维配对（t_crit：df35=2.030 / df23=2.069 / df20=2.086）

| 维度 | V2 d (t) | V3 d (t) |
|---|---|---|
| Quality (n=36) | +0.46 (+0.17) | **-6.54 (-2.15) 显著** |
| Aesthetics (n=36) | -3.06 (-1.33) | **-10.76 (-3.01) 显著** |
| Alignment (n=36) | +1.69 (+1.12) | -3.64 (-1.48) |
| Real-world Fidelity (n=24) | +1.06 (+0.36) | -6.33 (-1.89) |
| Creative Generation (n=21) | +1.33 (+0.32) | -3.44 (-0.67) |
| 配对总分（扁平口径） | +0.34±2.57 (+0.26) | **-6.06±3.85 (-3.19)** |
| 配对总分（bench 层级口径） | +0.72 (+0.61) | **-5.53 (-2.98)** |

结论对两种聚合口径稳健；统计经独立复算（偏差 ≤0.01）。

## 验证方法

- 选样：原 10 条全保留 + 按四维模式分层等距抽 26 条 = 36 条（Q/A/Al=36、RWF=24、CG=21；153 任务/套）；确定性重跑逐字节一致（选样验核 7/7 PASS）。
- 生成：同种子同参数，2026-08-25 三套 ×36 张（13:11→20:36 零中断）；新旧同 ID 图 30/30 sha256 逐字节一致（生成确定性 100%，新旧证据互证；图片验核 6/6 PASS）。
- 评审：官方 judge（27B Q-Judger，int8 本地，官方参数），153×3 任务零解析失败；评审完整性验核 6/6 PASS（含官方计分口径逐位复现）。
- 统计：配对 t 检验，独立复算 PASS（双口径、逐维、t 临界值核对）。

## 局限

- n=36 配对可检测 ~2.6 分的总分漂移；V2 的"无下降"结论强度=CI ±2.57 内含零。
- 评审为本地 int8 运行（配对设计抵消 judge 偏差）。
- 结论限定 2048²×50 步与该提示词分布。

## 产物清单

```
（验证工作区 /workspace/speedup/）
bench/gen36.jsonl                                   36 提示词（选样验核确定性）
bench/imgs36-{baseline,int02,int07}/                三套图 ×36
bench/judge-input-{baseline36,int0236,int0736}_*    三套分数与逐任务输出
logs/gen36-*.log / judge-*36.log / monitor36.log    全程日志与看护心跳
REPORT.md / HANDOFF.md                              本报告与工程记录

（仓库内对应收据 SenseNova-U1.5-ROCm/docs/results/）
validation/cfg-interval/cfg-interval36.json         主收据（终版）
validation/cfg-interval/gen36-prompts.jsonl         题集副本
validation/cfg-interval/judge-input-*36_judged.jsonl 逐任务输出副本（459 任务）
validation/cfg-interval/quality-scores-*36.json     三套分数副本
logs/cfg-interval36-{baseline,int02,int07}.log      生成日志副本
validation/cfg-interval/cfg-interval.json           n=10 历史收据（已标 superseded）
```
