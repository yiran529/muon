# Task 3 报告：纯 ArcTopKAdamW

## 完成内容

- 新增 dion/adamw_arctopk.py，实现独立的 ArcTopKAdamW，不继承
  DistributedOrthoBase。
- 压缩组按稳定的 shape/dtype 首次出现顺序创建 AsyncTask，统一通过
  AsyncRuntime(max_concurrent_tasks=3) 调度，并复用 arc_topk_sync。
- 非压缩组使用共享 dense gradient all-reduce；缺失本地梯度按零张量处理，
  不改变 collective 顺序；param.grad 不被永久替换。
- 所有参数构造时预填充 AdamW 的 momentum、variance、FP32 step_dev；
  压缩参数额外预填充 ARC EF21M 三个状态。
- 增加 DeviceMesh 拒绝、压缩组 2D 校验、optimizer-wide _arc_step、
  一致性校验、state dict 恢复和旧 checkpoint 的
  arc_start_compress_step=0 迁移。
- 从 dion 顶层导出 ArcTopKAdamW。
- 新增本地与两 rank Gloo 测试。

## 验证

- /home/wyr/dion/.venv/bin/python -m pytest tests/test_adamw_arctopk.py -q
  - 14 passed
- GLOO_SOCKET_IFNAME=lo /home/wyr/dion/.venv/bin/python -m pytest
  tests/test_adamw_arctopk_distributed.py -v
  - 1 passed（需要最小 loopback socket 权限）
- 全仓库 pytest：416 passed，16 skipped，8 failed。
  8 个失败全部来自既有 tests/test_dion2_post_ortho_triton.py 的
  wrapper subclass 与当前 PyTorch Inductor 的 data-dependent guard，
  不涉及本任务文件。
- git diff --check 和 Python 编译检查通过。

## 注意事项

当前环境无可用 CUDA；全仓库既有 Triton/Inductor wrapper 测试仍有上述环境/依赖相关失败。

## Review round 1 修复

- RED：新增持久 LR tensor、BF16/缺失 `step_dev`、梯度不变、非 bool
  `arc_compress`、调度器并发边界以及分布式 ratio=1/状态恢复测试；原实现
  对 LR tensor 和 BF16 `step_dev` 测试分别失败。
- GREEN：每组 LR 现持有参数本地设备上的持久 FP32 0-d tensor，支持调度器
  重新赋值而保持 tensor identity；加载后 `step_dev` 强制为本地 FP32，缺失
  时补零初始化；AsyncRuntime 并发上限固定为 3。
- Review focused verification：本地 19 passed；两 rank Gloo 1 passed（最小
  loopback socket 权限）；`git diff --check` 和 Python 编译检查通过。

## Review round 2 修复

- 将 scheduler boundary 测试改为真实消费 `AsyncRuntime` 的任务生成器，避免
  仅构造任务而未推进 generator 的假覆盖。
- 使用 `(4, 3), (2, 3), (4, 3)` 三个压缩参数，观察到同步器收到稳定顺序
  `[(0, [(4, 3), (4, 3)]), (1, [(2, 3)])]`，可检测 shape 分组遗漏或
  `task_index` 重置。
- Round-2 focused verification：本地 19 passed；两 rank Gloo 1 passed。
