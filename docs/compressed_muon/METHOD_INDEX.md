# Compressed Muon 方法索引

| 方法编号 | 名称 | 核心思路 | 主要文件 | 状态 |
|---|---|---|---|---|
| M001 | ARC-TopK-EF21M-Muon | 用完整 ARC-TopK 与 EF21M 替换 DDP dense gradient All-Reduce，再进入原 Muon 更新路径 | `docs/compressed_muon/methods/M001_arc_topk_ef21m_muon.md` | testing |
| M002 | GreedyLore-Muon | 在 unchanged ordinary Muon 之前，用 GreedyLore DDP hook 压缩数据并行梯度输入；非 Muon 参数保持 dense 同步 | `docs/compressed_muon/methods/M002_greedy_lore_muon.md` | testing |
