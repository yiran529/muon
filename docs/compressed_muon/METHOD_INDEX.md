# Compressed Muon 方法索引

| 方法编号 | 名称 | 核心思路 | 主要文件 | 状态 |
|---|---|---|---|---|
| M001 | ARC-TopK-EF21M-Muon | 用完整 ARC-TopK 与 EF21M 替换 DDP dense gradient All-Reduce，再进入原 Muon 更新路径 | `docs/compressed_muon/methods/M001_arc_topk_ef21m_muon.md` | testing |
| M002 | GreedyLore-Muon | 在 unchanged ordinary Muon 之前，用 GreedyLore DDP hook 压缩数据并行梯度输入；非 Muon 参数保持 dense 同步 | `docs/compressed_muon/methods/M002_greedy_lore_muon.md` | testing |
| M003 | Rand-K-EF14-Muon | 对二维 DDP 梯度使用跨 rank 共享的逐参数随机支持，只 All-Reduce 被选值，随后进入 ordinary Muon | `docs/compressed_muon/methods/M003_randk_ef14_muon.md` | testing |
| M004 | Top-K-EF14-Muon | 各 rank 对二维 DDP 梯度逐参数选择绝对值 Top-K，All-Gather 值和索引后重建平均梯度 | `docs/compressed_muon/methods/M004_topk_ef14_muon.md` | testing |
| M005 | PowerSGD-Muon | 在 unchanged ordinary Muon 之前，用 PowerSGD 压缩 DDP 二维矩阵梯度；dense auxiliary 与不满足收益条件的矩阵保持精确同步 | `docs/compressed_muon/methods/M005_power_sgd_muon.md` | testing |
