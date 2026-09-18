# M005 PowerSGD-Muon Worklog

## 2026-09-16：实现记录与验证

### 实现提交

M005 的实现和修订由以下提交组成：

- `a42c78b`：PowerSGD tensor foundations；
- `125d1e6`：tensor validation；
- `6d88fea`：stable parameter layout；
- `9d1ca8e`：DDP state lifecycle；
- `78519c0`：PowerSGD DDP communication hook；
- `7eac1a0`：cancellation/failure lifecycle；
- `c9f240f`：bucket pipeline；
- `073a6c7`：CUDA completion export；
- `e42c43b`：PowerSGD-Muon training integration；
- `83415f5`：make the PowerSGD-Muon recipe runnable。

方法语义固定为：DDP gradient compression before unchanged ordinary Muon；默认
只压缩 Muon 的可获益二维矩阵，dense auxiliary 与不满足收益条件的矩阵精确
同步；每个矩阵的逻辑因子 payload 为 `r(m+n)`，但协议包含两次 All-Reduce。
这是 approximate Muon，不宣称 dense-equivalent、speedup 或 convergence gain。

### 实际验证

- `.venv/bin/python -m compileall -q dion train_powersgd.py`：exit 0。
- `.venv/bin/python -m pytest -q tests/test_power_sgd.py tests/test_power_sgd_layout.py tests/test_power_sgd_ddp_checkpoint.py tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_distributed.py tests/test_train_powersgd.py`：105 passed，14 warnings，28.36 s。
- `.venv/bin/python -m pytest -q -m 'not multi_gpu'`，并排除显式 NCCL、CUDA graph 和 sharded-GPU tests：957 passed，16 skipped，2 failed，15 warnings，11:00.30。

两项 failure 均为 unrelated GreedyLore/profiler contract drift，随后单独重跑仍为
`2 failed, 14 warnings in 3.69s`：

1. `tests/test_greedy_lore_profiler_launcher.py::test_print_plan_rotates_greedylore_modes_and_parameterizes_resources`
   期望 mapping 没有当前实现新增的 `calibrated_bucket_cap_mb_list: None`；
2. `tests/test_training_profiler_trace.py::test_greedylore_bucket_timeline_reports_queue_launch_and_completion_offsets`
   期望 timeline 没有当前实现新增的 `prepare_*` / `chain_wait_*` 字段。

没有启动 formal training、timing、profiler trace capture、benchmark 或 CM run；
没有改动上述 GreedyLore 文件，也没有为 M005 编造 experiment ID、指标或质量
结果。Task 5 最终 CUDA review fix 在只读检查确认可安全使用的本地 GPU 上运行了
`.venv/bin/pytest -q -m multi_gpu tests/test_power_sgd_ddp_hook_nccl.py`：
`3 passed, 14 warnings in 22.59s`。FP32 与 BF16 case 均为真实本地 two-rank NCCL
correctness，覆盖 rank-skewed delay、allocator churn、returned-Future 可见性、三轮
压缩、mixed dense auxiliary、跨 rank collective signature 一致性与真实 DDP
backward。第三个 warmed CUDA regression 在所选两张 GPU 上分别验证：未等待任一
bucket Future 时，`finish_step` aggregate 仍能向 caller stream 暴露多 bucket、独立
reconstruction stream 上的 gradient、EF14 error、Q memory 和 `q_initialized`
写入。该 suite 是 correctness smoke，不构成 timing 或性能实验。

### 结论与下一步

实现、checkpoint schema、CPU/Gloo hook lifecycle 和 DDP integration 已有测试
覆盖，本地 two-rank NCCL correctness 与 aggregate CUDA visibility 也已有上述
短测试证据，方法记录进入 `testing`。下一步仍需完成 multi-node correctness、
端到端 training、profiler-off paired timing 和公平 quality/convergence runs；在此
之前不应把 payload 公式写成端到端 speedup，也不应把 PowerSGD 的 SGD 证据写成
Muon convergence 证据。

## 2026-09-16：CM093/CM094 rank32 主实验启动

按 GreedyLore 论文 Table IV 与 CM070 的低秩口径，只安排 M005 rank32 主实验：
CM093 为 GPT-60M/10,000 updates，CM094 为 GPT-130M/20,000 updates；均使用
4 GPU DDP、BF16参数、FineWeb10B、seq256、global/device batch512/128、
bucket160 MiB、EF14、warm start、step1000后压缩和 seed1234。dense 与 M002
对照复用 CM070，不重跑 high-rank PowerSGD。

## 2026-09-16：warm-start 与 batched orthogonalization 性能修复

针对 CM093/CM094 在压缩阶段约 3 倍于 dense 的 step time，修复两条明确的本地
开销路径：warm start 仅在首个压缩 phase 生成随机 Q，后续直接复用 `q_memory`；
相同 shape 的 Q/P 按组堆叠，以 3-D FP32 Gram–Schmidt 合并矩阵间 CUDA kernel，
保留原逐列算法、epsilon、BF16输入输出和 P/Q All-Reduce 语义。

新增 CM095 timing-only 实验，复用 CM089 的 GPT-130M BF16 训练几何，20 步
PowerSGD warmup 后测量 800 个稳态 rank32 PowerSGD step。CM089 的 calibrated role-isolation
是 GreedyLore 专属布局，CM095 不声称与其 bucket role layout 完全一致。

CM095 于 2026-09-16 22:32 获得四张GPU后启动，但launcher传入未被
`train_powersgd.py`接受的`--val_loss_every`，入口参数解析即失败（exit 1），
没有训练或timing数据。保留失败产物；CM095a 使用独立YAML设置
`val_loss_every: 0`并移除非法CLI参数，作为修正后的重跑。
CM095a 启动前进行了4卡短试跑：dim64、2层、rank2、BF16、4 updates、
`start_compress_step=0`、无compile，exit 0，最终打印 step4/4 与验证指标。
此试跑只验证入口参数和基本DDP压缩路径，不作为rank32性能数据。
随后启动 `cm095a-powersgd-timing` tmux 控制器；按用户要求不轮询。
但启动门槛只检查每卡空闲显存，错误地选入已有其他计算进程的GPU7。
用户指出后立即向本次torchrun发TERM，确认CM095a父子进程均退出；保留
产物但不使用部分运行数据。CM095b启动门槛改为每卡显存占用低于1 GiB且
GPU UUID不在计算进程表中，四卡同时满足才运行；不复用CM095a产物。

首次 controller 启动后按用户要求关闭 checkpoint 保存；训练 worker 被定向停止，
原 controller 与 CM093 早期产物以 `-aborted-20260916T181005-checkpoint-enabled`
后缀保留。两份正式配置均改为 `checkpoint_freq: 0`，随后使用原实验编号重新启动；
本次按要求未执行额外脚本测试。

## 2026-09-16：CM093/CM094 rank32 主实验完成

关闭 checkpoint 后，由 `CM093-CM094-m005-rank32-main-controller` 在 GPU 2–5
串行完成两项正式训练；两项 formal cell、controller 和 W&B run 均正常退出，git
HEAD 为 `8f9260245cc39a63cda1df7b87be4581a4c111f1`。正式结果为：

| 实验 | 模型 / updates | final val loss | step average | peak allocated |
|---|---|---:|---:|---:|
| CM093 | GPT-60M / 10,000 | 4.1482 | 274.85 ms | 6209 MiB |
| CM094 | GPT-130M / 20,000 | 3.7253 | 612.31 ms | 11580 MiB |

两项均使用 BF16、FineWeb10B、seq256、global/device batch512/128、bucket160 MiB、
rank32、EF14、warm start、step1000后压缩和 seed1234。相对 CM070 同 dtype dense，
60M/130M 的 loss 分别高 `0.0403/0.0723`，PPL 分别高 `4.11%/7.50%`；相对
CM070 的 GL rank32，loss 分别低 `0.0637/0.0392`。但全程累计平均 step 分别比
dense 慢 `195.38%/190.80%`，没有 wall-clock 收益。该结果支持 M005 rank32 的
质量优于当前 GL rank32，但仍有 dense quality gap，且当前压缩阶段成本过高；不能
据此声称加速或 time-to-quality 改善。结果已同步至 `docs/compressed_muon/RESULTS.md`。

首次关闭 checkpoint 的启动中止产物继续保留并排除；正式产物为：

- `artifacts/compressed_muon/CM093-m005-powersgd-muon-gpt60m-bf16-rank32-ddp-ws4-s1234/`
- `artifacts/compressed_muon/CM094-m005-powersgd-muon-gpt130m-bf16-rank32-ddp-ws4-s1234/`
- `artifacts/compressed_muon/CM093-CM094-m005-rank32-main-controller/`

对应 W&B run IDs 为 `5pnhy9mg` 和 `74umk3n3`。

## 2026-09-17：优化后完整质量复跑排队

CM095b 在四张满足“显存占用低于1 GiB且无计算进程”的GPU上完成：800个稳态
rank32 PowerSGD step为`204.89 ms`，peak `11603 MiB`。相对CM089历史
sharded-SVD `198.67 ms`慢`3.13%`；相对优化前CM094压缩阶段估计
`633.25 ms`降低`67.64%`。后者不是严格paired ablation，仅说明两项实现修复
消除了大部分已观察开销。

为验证优化后的完整训练质量，新建CM096/CM097，分别逐项复用CM093/CM094配置，
只改变当前代码版本和实验名称。两项由同一controller串行运行；首次启动及两个cell
开始前均要求所选四张GPU显存占用低于1 GiB且不存在计算进程。不保存checkpoint，
不执行launcher测试，不由agent轮询。

## 2026-09-17：GreedyLore历史几何的stage-1 timing排队

按阶段一方案登记CM098–CM101共7个单样本profiler-off cell：BF16 350M/1B最大
可行batch、FP32 350M batch64、FP32 350M batch8的bucket160/80桥接，以及
FP32 720M batch12/8。统一rank32、EF14、warm start、step0起压缩、20个
PowerSGD warmup；测量窗口逐项匹配历史GreedyLore几何的800或200 updates。

controller先等待CM096/097 controller结束，再等待四张显存占用低于1 GiB且无计算
进程的GPU；所有cell串行，并在每项开始前重新确认同组GPU空闲。容量失败只记录，
不自动降低batch。阶段一仅与历史结果作诊断比较，正式wall-clock结论仍需对候选点
补当前代码dense/PowerSGD rotated repeats。

## 2026-09-17：CM096/CM097 优化后完整质量复跑完成

controller 在 GPU 2/3/4/7 串行完成 CM096/CM097，两个 formal cells 与 controller
均 exit `0`，git HEAD 为 `43330d708a760d496bcfe30bdf4d0a8ab519acf3`。结果如下：

| 实验 | final val loss | step average | peak allocated | 相对优化前同配实验 |
|---|---:|---:|---:|---:|
| CM096 / 60M | 4.1485 | 94.43 ms | 6310 MiB | loss `+0.0003`；step `-65.64%` |
| CM097 / 130M | 3.7255 | 206.52 ms | 11580 MiB | loss `+0.0002`；step `-66.27%` |

两次完整训练的最终 loss 与 CM093/CM094 基本相同，说明 warm-start Q 复用和 batched
Gram–Schmidt 没有造成可见质量退化，同时消除了约三分之二的全程 step 时间。相对
CM070 历史 BF16 dense，60M慢`1.48%`、130M快`1.92%`；相对历史 GreedyLore
rank32分别快`0.41%/3.96%`。这些小幅性能差是跨实验而非同期配对，不作统计显著结论。
W&B run IDs 为 `8w4eai61` 和 `3ufxprln`；原始产物与 controller 分别位于对应
CM096/CM097目录及`artifacts/compressed_muon/CM096-CM097-m005-optimized-rank32-main-controller/`。

## 2026-09-17：CM098–CM101 stage-1 timing完成

依赖 CM096/CM097 完成后，stage-1 controller 在同一 GPU 2/3/4/7 上通过严格空闲
门槛并串行完成7个cell；所有cell和controller均exit `0`，git HEAD为
`d1cbb16a9b461db9e10e9ffa1d41bf4625e29123`。

| cell | PowerSGD step | peak | 相对历史 dense | 相对历史 local-SVD M002 |
|---|---:|---:|---:|---:|
| 350M BF16 batch72 | 384.06 ms | 18410 MiB | +3.71% | -6.04% |
| 1B BF16 batch24 | 694.61 ms | 20188 MiB | +33.91% | +9.74% |
| 350M FP32 batch64 | 460.53 ms | 21653 MiB | +17.29% | +6.57% |
| 350M FP32 batch8 bucket160 | 239.26 ms | 9298 MiB | +0.06% | +18.65% |
| 350M FP32 batch8 bucket80 | 363.22 ms | 9274 MiB | +43.74% | +68.21% |
| 720M FP32 batch12 | 723.26 ms | 20095 MiB | +40.22% | +60.83% |
| 720M FP32 batch8 | 716.56 ms | 18509 MiB | +56.20% | +67.26% |

阶段一表明 M005 在350M BF16 batch72和350M FP32 batch8/bucket160上距历史 dense
不超过5%，可作为后续current-code配对复测候选；1B BF16及720M FP32明显为负。
bucket80相对同配置bucket160额外慢`123.96 ms (51.81%)`，而peak几乎不变，是当前
最需要profile的异常点，可能涉及bucket数量/顺序、packing或collective串行链。
所有对照均为历史单样本跨实验比较，尚不构成正式wall-clock结论。结果已同步至
`docs/compressed_muon/RESULTS.md`和`docs/compressed_muon/EXPERIMENTS.md`。

## 2026-09-17：CM102/CM103 60M BF16 paired timing启动

为直接比较当前优化后的M005与M002 sharded-SVD，安排两组GPT-60M、4卡、BF16、
seq256、GA1、bucket80、rank32 timing。CM102使用global/device batch512/128，桥接
CM067完全相同的历史几何；CM103使用32/8，观察小physical batch下固定压缩成本。
GreedyLore使用interval200、independent score与sharded-SVD；PowerSGD使用EF14与
warm-start Q，因算法没有周期refresh而不设置interval。每种几何运行3组交替顺序配对，
每cell为20个压缩态warmup加800个measured updates；不重跑dense，不启用profiler、
W&B或checkpoint。controller严格等待四张无计算进程且显存占用低于1 GiB的GPU，
所有12个cell串行执行，agent不轮询。

## 2026-09-17：CM102/CM103 60M BF16 paired timing完成

controller在GPU 2/3/6/7上完成全部12个cell并exit `0`。CM102 batch128的
PowerSGD/GreedyLore sharded-SVD mean为`96.030/94.343 ms`，M005慢
`1.687 ms (1.79%)`；CM103 batch8为`35.480/26.840 ms`，M005慢
`8.640 ms (32.19%)`。两组各自的3个配对差值全部为正，范围分别为
`[+0.78,+2.64] ms`和`[+8.07,+9.23] ms`。PowerSGD peak仅低`8/107 MiB`。

结论为paired-negative-for-M005：batch128下两者接近；batch8下模型计算缩短，
PowerSGD每步投影、正交化及P/Q两阶段collective的固定成本占比显著增加。CM102
GreedyLore mean与CM067历史local-SVD mean恰好同为`94.343 ms`；CM103没有同期或
历史同配置dense，因此不作相对dense claim。结果已同步至RESULTS和实验登记。

## 2026-09-18：CM104 scripted Gram–Schmidt timing完成并归档

在保持既有PowerSGD计算流程的前提下，当前工作区实现使用TorchScript执行同一
Gram–Schmidt循环，并去除不必要的clone。CM104在GPU 3/4/5/6串行测量15个
PowerSGD-only几何；15/15 cells和controller均exit `0`，短窗口val loss均有限。
统一配置为4卡DDP、seq256、GA1、rank32、EF14、warm-start、seed42、step0起压缩；
20个压缩态warmup后通常测量800 updates，350M FP32 batch8的bucket160/80两项
测量200 updates。无同期dense或GreedyLore，也没有repeats或完整质量复跑。

同几何历史M005比较，9项中8项step降低、1项（350M BF16 batch72）增加`0.66%`；
1B BF16从`694.61`到`633.50 ms`，350M FP32 batch64从`460.53`到`427.57 ms`，
720M FP32 batch8从`716.56`到`637.66 ms`。对历史GreedyLore，60M BF16 batch128
接近持平（`93.85`对`94.343 ms`）；batch8仍慢于sharded-SVD `14.61%`。
350M FP32 batch64相对local-SVD快`1.06%`、相对sharded-SVD慢`3.79%`；
350M/720M FP32小batch对相同几何的GreedyLore仍明显落后。350M FP32 batch8
的bucket80比bucket160慢`85.43 ms (37.07%)`，bucket敏感性仍在。

以上均为单次跨实验诊断，不能把收益归因于单个代码优化，也不能据小幅差异判定
稳定胜负。完整15项数据和GreedyLore对照已写入`docs/compressed_muon/RESULTS.md`，
实验状态已登记在`docs/compressed_muon/EXPERIMENTS.md`。原始产物位于
`artifacts/compressed_muon/CM104-m005-scripted-gs-timing-ws4-s42/`；记录的HEAD为
`ac091d38db15d009311d35101d2f5511ac688579`，但优化代码未提交，产物中保存
`git_status.txt`与`code.patch`，复现时必须连同补丁一起使用。
