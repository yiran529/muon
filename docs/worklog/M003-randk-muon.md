# M003 Rand-K-Muon Worklog

## 2026-09-12：实现 tensor-wise Rand-K DDP 基线

- 目的与假设：按 ARC-TopK release 的 Rand-K 基线语义，为 ARC-TopK 和 GreedyLore 提供同训练路径的稀疏基线。
- 修改：新增共享 Sparse-K 原语、稳定布局、异步 DDP hook、`train_sparsek.py` 入口和 M003 配置；所有二维参数使用共享随机支持，非二维参数 dense 同步，默认 EF14。
- 验证：纯 CPU 原语、状态生命周期、checkpoint schema、单 rank hook、双 rank Gloo、两 rank DCP 改 bucket 后续训，以及双 GPU NCCL fake/真实 DDP hook smoke。
- 观察：Rand-K 只同步选中 values，不传 indices；支持由 step 和稳定参数 ID 推导，不依赖 bucket 顺序或全局 RNG。
- 结论与下一步：实现进入 testing；后续进行性能归因和与同配方 dense/ARC/Top-K 的短训练比较。

## 2026-09-13：CM038c / CM040c 正式质量实验计划

- 沿用 CM038（GPT-60M）和 CM040（GPT-130M）的无 LR warmup、最后 20% 线性衰减、无 clipping 配方，只将 all-2D ARC-TopK-EF14 替换为 M003 Rand-K-EF14。
- 固定 ratio 0.2、seed42、step1000 后压缩、bucket160 MiB、4 GPU DDP；分别训练 8393/16785 updates。
- 与 M004 串行运行，产物根目录为 `artifacts/compressed_muon/CM038cd-CM040cd-m003-m004-sparse-k-ef14-muon-ws4-s42/`。

## 2026-09-13：CM038c / CM040c 启动

- 10:42 在 tmux `cm038cd_cm040cd` 中启动与 M004 共享的串行 controller，使用 GPU `2,3,4,5`。
- 启动前确认目标产物不存在、无重复训练进程、每卡空闲显存约 24 GiB，数据集、W&B 认证和启动文件预检通过。
- 原自动启动链因 `tail --pid` 等待已退出但未被回收的 zombie PID 而停滞；替换该等待 pane 后直接启动已有 controller，未修改实验配方。
- CM038c 强制压缩 probe 已 exit `0`；完整训练结果待串行队列结束后记录。

## 2026-09-13：CM040c r2 恢复队列

- 首次队列因前置 CM040d 遇到系统盘 ENOSPC 而停止，CM040c 尚未进入正式训练。
- 清理本账号可再生成的 TorchInductor/uv 缓存后，可用空间从 8.2 GiB 回复到 26 GiB；实验数据和 checkpoint 未删除。
- 15:20 在 tmux `cm040cd_r2` 中启动独立 retry controller，使用 GPU `4,5,6,7`；CM040c r2 排在 CM040d r2 成功后串行运行。
- retry 产物根目录：`artifacts/compressed_muon/CM040cd-r2-m003-m004-sparse-k-ef14-muon-ws4-s42/`。
- 15:27 按用户指定停止 r2，并改用空闲 GPU `2,3,4,5` 启动 r3；CM040c 仍排在 CM040d 之后。r3 产物根目录为 `artifacts/compressed_muon/CM040cd-r3-m003-m004-sparse-k-ef14-muon-ws4-s42/`。
