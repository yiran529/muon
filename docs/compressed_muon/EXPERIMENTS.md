# Compressed Muon 实验登记

| 实验编号 | 方法 | 配置 | 目的 | 状态 | 结果位置 |
|---|---|---|---|---|---|
| CM001-m001-gpt160m-ddp-ws4-s42 | M001 ARC-TopK-EF21M-Muon | `configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`；4 GPU DDP；ARC seed 42；compression warmup 300 | 沿用既有 DDP Muon 160M 超参数，首次启动 M001 正式训练并验证正常执行路径 | running；2026-09-04 01:13 CST；W&B `21qkn1do` | `artifacts/compressed_muon/CM001-m001-gpt160m-ddp-ws4-s42/` |
