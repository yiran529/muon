# M003：Rand-K-EF14-Muon

## 定位

M003 在 DDP 梯度同步阶段对所有二维参数逐参数展平，并保留固定比例的随机坐标。随机支持由基础 seed、one-based optimizer step 和稳定参数 ID 推导，各 rank 一致，因此只需对选中值执行 All-Reduce；非二维参数保持 dense All-Reduce。重建的近似全局梯度随后进入未修改的 ordinary Muon。

## 方法语义

默认使用 EF14：先计算 `u_i = local_gradient_i + residual_i`，在共享随机支持上得到 `C_i(u_i)`，将本地 residual 更新为 `u_i - C_i(u_i)`，再平均各 rank 的稀疏值。Rand-K 不乘 `numel / k`，与 ARC-TopK release 中的基线语义一致，因此不将它表述为无偏压缩器。`noef` 作为消融模式。

第一步、`step <= sparse_k_start_compress_step` 或 `ratio=1` 时使用 full-support dense 平均；已有 residual 会在 full-support 步被消费并清零。首版只支持固定 process group 的 DDP、完整参数覆盖和 `find_unused_parameters=False`。

## 实现

- 入口：`train_sparsek.py --sparse_k_method randk`
- 原语与配置：`dion/sparse_k.py`
- 布局：`dion/sparse_k_layout.py`
- DDP hook：`dion/sparse_k_ddp_hook.py`
- 基础配置：`configs/compressed_muon/m003_randk_muon_ddp.yaml`

该方法近似的是 Muon 的数据并行梯度输入，不是 dense Muon 的等价通信实现。Muon 动量、正交化和结果通信未修改。
