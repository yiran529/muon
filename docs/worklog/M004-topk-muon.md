# M004 Top-K-Muon Worklog

## 2026-09-12：实现 tensor-wise Top-K DDP 基线

- 目的与假设：按 ARC-TopK release 的 Top-K 基线语义，为 ARC 的 index-free All-Reduce 设计提供直接对照。
- 修改：复用独立 Sparse-K 框架；各 rank 逐参数选择绝对值 Top-K，All-Gather values/int32 indices，重叠坐标累加，默认 EF14。
- 验证：纯 CPU 原语、状态生命周期、checkpoint schema、单 rank hook、双 rank Gloo 上不同/重叠 support 的重建、两 rank DCP 改 bucket 后续训，以及双 GPU NCCL fake/真实 DDP hook smoke。
- 观察：Top-K 需要按 world size 接收 values 和 indices，其通信口径不同于 Rand-K/ARC 的 selected-values All-Reduce。
- 结论与下一步：实现进入 testing；后续进行通信量统计和质量比较。

## 2026-09-13：CM038d / CM040d 正式质量实验计划

- 沿用 CM038（GPT-60M）和 CM040（GPT-130M）的无 LR warmup、最后 20% 线性衰减、无 clipping 配方，只将 all-2D ARC-TopK-EF14 替换为 M004 Top-K-EF14。
- 固定 ratio 0.2、seed42、step1000 后压缩、bucket160 MiB、4 GPU DDP；分别训练 8393/16785 updates。
- Top-K 通信解释将同时计入各 rank 的 values 与 indices；与 M003 串行运行并共用正式产物根目录。

## 2026-09-13：CM038d / CM040d 启动

- 10:42 在 tmux `cm038cd_cm040cd` 中启动与 M003 共享的串行 controller，使用 GPU `2,3,4,5`。
- 四个正式 cell 将在全部强制压缩 probe 通过后按 CM038c、CM038d、CM040d、CM040c 的顺序执行；10:43 已进入 CM038d probe。
- 产物根目录为 `artifacts/compressed_muon/CM038cd-CM040cd-m003-m004-sparse-k-ef14-muon-ws4-s42/`，完整训练结果待队列结束后记录。

## 2026-09-13：CM040d r2 从头重跑

- CM040d 首次运行到 step 约 7130/16785 时因系统盘 ENOSPC 以 exit `120` 停止；该配方禁用 checkpoint，不能续训，中间 loss 不作最终结果。
- 保留首次失败产物，15:20 在 tmux `cm040cd_r2` 中以完全相同的训练配方从头启动 r2，使用 GPU `4,5,6,7`，启动时可用磁盘 26 GiB。
- r2 先执行强制压缩 probe，通过后运行 CM040d Top-K，再串行 CM040c Rand-K；两个 formal cell 之前均有 16 GiB 剩余空间门禁。
- retry 产物根目录：`artifacts/compressed_muon/CM040cd-r2-m003-m004-sparse-k-ef14-muon-ws4-s42/`。
- r2 已于 15:27 按用户指定停止，所有脱离 tmux 的 r2 worker 也已退出；随后在 GPU `2,3,4,5` 直接启动 r3，当前从 CM040d 强制压缩 probe 开始。
- r3 产物根目录：`artifacts/compressed_muon/CM040cd-r3-m003-m004-sparse-k-ef14-muon-ws4-s42/`。
