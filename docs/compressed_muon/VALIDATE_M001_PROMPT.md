# M001 最低验证 Prompt

你是独立验证 agent。请在 Dion 仓库根目录只读审查并验证 M001（ARC-TopK-EF21M-Muon）；不要修改代码、配置、文档，不要提交 commit，不要启动正式训练或性能实验。

## 验证目标

确认第一版达到“实现论文 ARC-TopK Algorithm 1 与 EF21M 公式 11a–11c，并接入现有 Muon 正常更新路径；仅支持 DDP；通过自动化测试”的完成标准。不要把测试通过扩大解释为收敛、精度或通信性能已经得到实证。

## 环境约束

- 不升级或安装依赖。
- 使用仓库现有锁文件与环境：`uv run --frozen --extra dev ...`。
- Gloo 多进程测试需要允许本机 localhost socket；若沙箱阻止绑定端口，请申请执行测试所需的最小权限。
- 保留工作区现有改动，不执行 reset、checkout、clean 或删除操作。

## 必须执行

1. 检查工作区与提交：

   ```bash
   git status --short
   git log --oneline -8
   ```

2. 执行 M001 聚焦测试（包含两 rank Gloo）：

   ```bash
   uv run --frozen --extra dev pytest tests/test_train_factories.py tests/test_arc_topk.py tests/test_arc_topk_distributed.py tests/test_muon_arctopk.py tests/test_muon_arctopk_distributed.py tests/test_train_arctopk.py -v
   ```

3. 执行与原 Muon/训练入口有关的回归测试：

   ```bash
   uv run --frozen --extra dev pytest tests/test_configs.py tests/test_state_prepopulation.py tests/test_optimizers.py tests/test_dion3_alias.py -v
   ```

4. 执行静态最低检查：

   ```bash
   uv run --frozen --extra dev python -m compileall -q dion train.py train_arctopk.py tests
   git diff --check
   ```

## 扩展检查（时间允许时）

执行完整测试目录：

```bash
uv run --frozen --extra dev pytest tests -v
```

完整套件包含大量慢速 CPU/多进程和 CUDA 条件测试，不属于 M001 的最低通过门。如果出现失败，必须检查失败文件是否在 M001 变更范围及调用链内，不能把无关既有失败误报成 M001 失败。


## 必须人工核查

- `train.py` 的直接执行默认仍使用原 `Hyperparameters` 与 `init_optimizer`，没有 ARC 专属分支。
- `train_arctopk.py` 复用共享训练循环，且非 DDP `device_mesh` 明确报错。
- DDP backward 在 `replicate_mesh_grad_sync=True` 时保留本地梯度，矩阵梯度在 optimizer 内进入 ARC-TopK-EF21M。
- ARC 顺序固定为 rank-0 seed broadcast、sketch All-Reduce 平均、公共 Top-K support、selected-values All-Reduce 平均。
- 各 rank 按相同参数/shape 顺序发 collective；局部 `grad is None` 使用零张量，不能造成 collective 次序分叉。
- EF21M 状态包括 `arc_h_local`、`arc_g_local`、`arc_g_global`，并随 optimizer `state_dict()` 保存和恢复。
- 首个 optimizer step 使用 dense All-Reduce 初始化 `h_0`、`g_0`，满足 `g_0 = h_0`；配置默认 `arc_start_compress_step=1000`，第 1–1000 步保持 dense，第 1001 步开始 ARC。
- projection 与 sketch 跟随梯度/状态 dtype；BF16 路径不得把 sketch 固定提升为 FP32。
- `arc_g_global` 进入原 Muon momentum、Nesterov、正交化、结果收集和参数更新路径；原 `dion/muon.py` 行为未被修改。
- AdamW/Lion 参数仍执行 dense All-Reduce，两个 rank 的参数更新保持一致。
- `ratio=1` 的测试验证 ARC/EF21M 在指定条件下等价于 dense gradient average。
- 配置 `configs/compressed_muon/m001_arc_topk_muon_ddp.yaml` 的 `dp_size`、`fs_size`、`tp_size` 都是 `null`，且 `checkpoint_freq: 0`。

## 失败证据要求

若任一步失败，报告：完整命令、退出码、首个根因堆栈、是否可稳定复现，以及失败属于实现问题、环境限制还是既有无关问题。不要自行修复。若测试跳过，列出跳过数量和原因，特别说明分布式测试是否真的执行。

## 最终报告模板

```text
结论：通过 / 不通过 / 环境阻塞
提交：<HEAD>
工作区：<是否存在预先已有改动>
聚焦测试：<passed/failed/skipped 与耗时>
回归测试：<passed/failed/skipped 与耗时>
完整测试（扩展，可未执行）：<passed/failed/skipped 与耗时，或未执行原因>
静态检查：<结果>
人工核查：<逐项简述>
限制：未验证正式训练、收敛质量、吞吐或实际通信收益
失败证据：<无，或按要求列出>
```
