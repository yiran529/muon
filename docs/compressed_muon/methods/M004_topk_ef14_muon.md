# M004：Top-K-EF14-Muon

## 定位

M004 在 DDP 梯度同步阶段对所有二维参数逐参数展平，各 rank 独立选择绝对值最大的坐标。因为各 rank 的支持通常不同，hook 分别 All-Gather FP32 values 和 int32 local indices，按 rank 顺序累加重复坐标并除以 world size；非二维参数保持 dense All-Reduce。重建结果随后进入未修改的 ordinary Muon。

## 方法语义

默认使用 EF14：`u_i = local_gradient_i + residual_i`，本地 residual 更新为 `u_i - C_i(u_i)`。residual 必须由通信前的本地压缩结果计算，不能使用全局平均结果。`noef` 作为消融模式。

第一步、`step <= sparse_k_start_compress_step` 或 `ratio=1` 时使用 full-support dense 平均；已有 residual 会在 full-support 步被消费并清零。首版只支持固定 process group 的 DDP、完整参数覆盖和 `find_unused_parameters=False`。

## 实现

- 入口：`train_sparsek.py --sparse_k_method topk`
- 原语与配置：`dion/sparse_k.py`
- 布局：`dion/sparse_k_layout.py`
- DDP hook：`dion/sparse_k_ddp_hook.py`
- 基础配置：`configs/compressed_muon/m004_topk_muon_ddp.yaml`

Top-K 的 payload 包含每个 rank 的 values 与 indices，不能把保留率直接当作端到端通信压缩率。该方法是梯度输入有损的 Muon 变体。
