# M004 Top-K-Muon Worklog

## 2026-09-12：实现 tensor-wise Top-K DDP 基线

- 目的与假设：按 ARC-TopK release 的 Top-K 基线语义，为 ARC 的 index-free All-Reduce 设计提供直接对照。
- 修改：复用独立 Sparse-K 框架；各 rank 逐参数选择绝对值 Top-K，All-Gather values/int32 indices，重叠坐标累加，默认 EF14。
- 验证：纯 CPU 原语、状态生命周期、checkpoint schema、单 rank hook、双 rank Gloo 上不同/重叠 support 的重建、两 rank DCP 改 bucket 后续训，以及双 GPU NCCL fake/真实 DDP hook smoke。
- 观察：Top-K 需要按 world size 接收 values 和 indices，其通信口径不同于 Rand-K/ARC 的 selected-values All-Reduce。
- 结论与下一步：实现进入 testing；后续进行通信量统计和质量比较。
