# M003 Rand-K-Muon Worklog

## 2026-09-12：实现 tensor-wise Rand-K DDP 基线

- 目的与假设：按 ARC-TopK release 的 Rand-K 基线语义，为 ARC-TopK 和 GreedyLore 提供同训练路径的稀疏基线。
- 修改：新增共享 Sparse-K 原语、稳定布局、异步 DDP hook、`train_sparsek.py` 入口和 M003 配置；所有二维参数使用共享随机支持，非二维参数 dense 同步，默认 EF14。
- 验证：纯 CPU 原语、状态生命周期、checkpoint schema、单 rank hook、双 rank Gloo、两 rank DCP 改 bucket 后续训，以及双 GPU NCCL fake/真实 DDP hook smoke。
- 观察：Rand-K 只同步选中 values，不传 indices；支持由 step 和稳定参数 ID 推导，不依赖 bucket 顺序或全局 RNG。
- 结论与下一步：实现进入 testing；后续进行性能归因和与同配方 dense/ARC/Top-K 的短训练比较。
