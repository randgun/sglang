# SGLang DSpark PR 文件说明与 NPU 适配分析

```sh
                 Scheduler
                     |
                     |
             SpeculativeRunner
                     |
        +------------+-------------+
        |                          |
   Eagle/DFlash                 DSpark
                                   |
                          DSparkWorker
                                   |
                         Draft Model Forward
                                   |
                        confidence scheduling
                                   |
                         variable verify
                                   |
                          Target Model
```



## 1. 文档范围

本文针对 SGLang PR [#30261: Add DSpark: confidence-scheduled speculative decoding](https://github.com/sgl-project/sglang/pull/30261)，说明该 PR 修改文件的目的、DSpark 推理链路、硬件强相关模块，以及 Ascend NPU 的建议适配路线。

本文以 PR 的合并提交 `6cc9352dfe6c5c013750e72b39c127870ef5b54f` 为分析基线。该 PR 于 2026 年 7 月 12 日合入 `main`，包含：

- 84 个修改文件；
- 38 个新增文件；
- 46 个已有文件修改；
- 约 17,700 行新增代码；
- 约 287 行删除代码。

需要特别说明：该 PR 实现的是 **DSpark 推理运行时**，不包含完整的 Draft 模型训练流水线。运行时假定已经存在训练好的 Draft Backbone、Target Hidden Projection、Markov Head，以及可选的 Confidence Head。

## 2. PR 的总体目标

DSpark 将以下能力集成进 SGLang：

1. 使用轻量 Draft Backbone 一次计算一个候选 Block；
2. 使用 Markov Head 恢复候选 Token 之间的顺序依赖；
3. 使用 Confidence Head 估计各候选位置的接受概率；
4. 使用 STS 校准 Confidence；
5. 结合 SPS 硬件成本表动态决定每个请求的验证长度；
6. 使用 Ragged Verify 对不同请求执行不同长度的 Target Verify；
7. 支持 Greedy、Rejection Sampling 和混合采样 Batch；
8. 将 Draft、Verify、Accept、KV Commit 等路径纳入 CUDA Graph 和 Triton Kernel；
9. 输出接受长度、截断长度、Confidence 和性能观测数据。

完整链路如下：

```text
ServerArgs / Spec Registry
          │
          ▼
     DSparkWorkerV2
          │
          ├── Target Prefill
          │     └── 捕获指定层 Target Hidden States
          │
          ├── Target Hidden → Draft KV Cache
          │
          ├── DraftBlockProposer
          │     ├── 输入 [bonus, mask, mask, ...]
          │     ├── Draft Backbone
          │     ├── Shared Target LM Head
          │     ├── Markov Head
          │     └── Confidence Head
          │
          ├── DSparkVerifyPlanner
          │     ├── STS 校准
          │     ├── SPS 成本预测
          │     └── 分配逐请求 Verify Length
          │
          ├── TargetVerifyExecutor
          │     ├── Static / Cap-Accept / Compact
          │     ├── Greedy / Rejection Sampling
          │     └── Accept + Bonus Token
          │
          └── Commit
                ├── 更新 Target KV
                ├── 注入 Draft KV
                └── 返回 Token 和统计信息
```

## 3. 模型相关文件

### 3.1 `python/sglang/srt/models/dspark.py`

这是通用稠密 DSpark Draft 实现，目前入口类为 `Qwen3DSparkModel`。主要内容包括：

- `VanillaMarkov`：使用 Token 低秩 Embedding 生成 logits bias；
- `GatedMarkovHead`：结合 Draft Hidden 控制 Markov bias；
- `RNNHead`：在候选 Block 内维护轻量递归状态；
- `DSparkConfidenceHead`：输出逐位置 Confidence；
- `DSparkDraftMixin`：加载 Markov/Confidence 权重、共享 Target LM Head、注入 Draft KV；
- `Qwen3DSparkModel`：通用 Qwen3 DSpark 注册入口。

通用实现复用 DFlash Draft Backbone，不在 Draft checkpoint 中重复保存 Target 的 Embedding 和 LM Head。运行时会挂接 Target LM Head，计算：

```text
Draft Hidden
    ↓
Shared Target LM Head
    ↓
Base Logits
    ↓
Markov Head(previous token)
    ↓
Corrected Logits
```

### 3.2 `python/sglang/srt/models/deepseek_v4_dspark.py`

这是 DeepSeek V4 专用实现，不能直接由通用 Qwen3 路径替代。主要包括：

- DSpark 专用 MQA/MLA Attention；
- Sliding Window Attention KV 写入；
- Target Hidden 的 `main_proj/main_norm`；
- DeepSeek V4 HC Head；
- Markov W2 Tensor Parallel Shard；
- Target Embedding 和 LM Head 共享；
- Q/KV 多 Stream 执行；
- MoE、TP、DP Attention 相关处理；
- DeepSeek V4 DSpark checkpoint 权重重映射。

### 3.3 `python/sglang/srt/models/deepseek_v4.py`

对 Target DeepSeek V4 增加 DSpark Hidden Capture 接口：

- 保存 `dspark_layers_to_capture`；
- 提供 `set_dspark_layers_to_capture()`；
- 在指定 Target Layer 完成后捕获 Hidden State；
- 将多层 Hidden State 返回给 DSpark Worker；
- 在启用 DSpark Capture 时限制部分不兼容的 TBO/CP 路径。

## 4. DSpark 核心运行组件

核心目录为 `python/sglang/srt/speculative/dspark_components/`。

| 文件 | 主要目的 |
|---|---|
| `dspark_worker_v2.py` | 总控 Worker，串联 Prefill、Draft、Verify、Accept 和 Commit |
| `dspark_config.py` | 解析 gamma、Mask Token、Markov Rank、Target Layer IDs 和 checkpoint 形式 |
| `dspark_draft.py` | 构造 Bonus+Mask Block，执行 Draft Backbone 和 Markov Sampling |
| `dspark_kv_inject.py` | 将 Target Hidden 投影并写入 Draft KV Cache |
| `dspark_planner.py` | 根据 Confidence、STS 和 SPS 分配逐请求验证长度 |
| `dspark_verify.py` | 执行 Target Verify、接受判断、Bonus Token 和 Hidden/KV Commit |
| `dspark_sps.py` | 定义和查询硬件执行成本表 |
| `dspark_sts.py` | 加载 STS 温度并记录校准样本 |
| `dspark_observability.py` | 记录每步 Draft、Verify、Confidence、接受长度和阶段耗时 |
| `dspark_block_accept_estimator.py` | 估计未被动态验证窗口截断时的完整 Block 接受长度 |

```sh
Scheduler
   │
   ├── Target TpModelWorker
   │      └── Target ModelRunner
   │             └── Qwen3ForCausalLM / DeepseekV4ForCausalLM
   │
   └── DSparkWorkerV2
          ├── 持有 Target TpModelWorker
          └── 内部创建 Draft TpModelWorker
                  └── Draft ModelRunner
                         └── Qwen3DSparkModel /
                            DeepseekV4ForCausalLMDSpark
```



### 4.1 Prefill

Prefill 由 Target 模型执行：

1. Target 处理完整 Prompt；
2. 捕获配置指定的 Target Layer Hidden States；
3. Target 生成首个 Token；
4. Target Hidden 经 Projection 转为 Draft Hidden；
5. 为 Draft 各层生成 K/V 并写入 Draft KV Cache；
6. Target 首个输出作为下一轮 Draft 的 Bonus Token。

Draft 不需要重新 Prefill 完整 Prompt。

### 4.2 Draft Decode

Draft 每轮构造：

```text
[bonus_token, mask_token, mask_token, ..., mask_token]
```

其中：

- 第一个位置是上一轮 Target 确认的 Bonus Token；
- 后续位置是并行 Draft 的占位 Mask Token；
- Token Embedding 来自 Target Embedding；
- 历史上下文通过 Draft KV Cache 提供。

Draft Backbone 一次产生整个 Block 的 Hidden 和 Base Logits，Markov Head 再按位置使用前一个已采样 Token 修正 logits。

### 4.3 Verify 与 Commit

Target Verify 完成后，系统计算：

- 连续接受长度；
- Bonus Token；
- 最终提交长度；
- 被动态窗口截断的 Token 数；
- 新序列长度；
- 对外输出 Token。

只有 Target 最终提交的前缀会被写入 Draft KV Cache，未接受的 Draft Token 不会污染后续上下文。

## 5. Ragged Verify

相关文件：

- `python/sglang/srt/speculative/ragged_verify.py`；
- `python/sglang/srt/speculative/ragged_verify_kernels.py`。

传统 Verify 通常让同一 Batch 中所有请求验证相同长度：

```text
request A: 8 tokens
request B: 8 tokens
request C: 8 tokens
```

DSpark Ragged Verify 支持：

```text
request A: 2 tokens
request B: 5 tokens
request C: 8 tokens
```

`RaggedVerifyLayout` 保存：

- `verify_lens`；
- `extend_start_loc`；
- `qo_indptr_device`；
- `total_verify_tokens`；
- `graph_num_tokens`；
- Host 和 Device 两套 Attention metadata。

支持三种运行模式：

| 模式 | 行为 |
|---|---|
| `static` | 所有请求执行固定长度 Verify，不依赖 Confidence 动态裁剪 |
| `cap-accept` | 主要用于估计和观测，限制可接受长度但不充分减少 Forward Token |
| `compact` | 真正压缩 Target Verify 输入，减少 Target Forward Token 数 |

`compact` 是 DSpark 获得主要动态调度收益的模式，同时也是 NPU 适配难度最高的模式。

## 6. Kernel 文件及目的

目录为 `python/sglang/srt/speculative/dspark_components/kernels/`。

### 6.1 `dspark_accept.py`

负责：

- Greedy Accept；
- Rejection Sampling；
- 同一 Batch 中 Greedy/Sampling 混合；
- Temperature Softmax；
- Bonus Token Gather；
- 接受长度 Finalize；
- 根据 Verify Cap 截断接受长度。

### 6.2 `dspark_schedule.py`

负责：

- 计算逐位置前缀存活概率；
- 按 Confidence 对额外 Verify Token 排序；
- 在总 Verify Budget 下执行 Top-K 分配；
- 生成每个请求的 Verify Length。

### 6.3 `dspark_verify_window.py`

负责：

- 构造 Ragged Verify Window；
- Compact Verify Token；
- 生成 Compact Row Index；
- 将 Compact Target Hidden Scatter 回固定布局；
- 构造 KV Commit/Inject 位置；
- 构造最终输出 Token。

### 6.4 `dspark_draft_model.py`

负责：

- Markov 单步采样；
- Draft KV Projection；
- DeepSeek V4 多层 KV Projection 融合；
- TP-local logits 和 Markov bias 处理。

### 6.5 `dspark_attn_metadata.py`

负责 DeepSeek V4：

- Sliding Window Gather；
- SWA Page Index；
- Block Causal Sequence Length；
- Draft Block Attention metadata。

### 6.6 `dispatch.py`

根据输入设备选择 Triton 快路径或 Torch Reference 路径。Torch 路径便于测试正确性，也可以作为 NPU bring-up 阶段的参考实现。

## 7. 配置和模型注册文件

| 文件 | 修改目的 |
|---|---|
| `server_args.py` | 注册 DSPARK 及 block size、SPS、STS、Graph Tier 等参数 |
| `arg_groups/speculative_hook.py` | 校验 DSpark 组合、解析 gamma、设置 Draft 路径和默认参数 |
| `arg_groups/deepseek_v4_hook.py` | 允许 DeepSeek V4 使用 DSPARK |
| `speculative/spec_info.py` | 注册 `SpeculativeAlgorithm.DSPARK` 和 Worker 类型 |
| `speculative/spec_registry.py` | 为自定义算法补充 DSpark/Ragged 能力接口 |
| `configs/model_config.py` | 识别 Qwen3/DeepSeek V4 DSpark checkpoint 和内嵌 Draft 权重 |
| `model_executor/model_runner.py` | 解析 Draft 配置并设置 Target Hidden Capture Layer |
| `model_executor/pool_configurator.py` | 为 Target 和 Draft KV Cache 重新计算内存占用 |

当前 `_handle_dspark()` 明确拒绝非 CUDA 设备，这是 NPU 适配首先需要解除的入口限制。但只有在基础执行路径可用后才应解除，否则会将运行失败延后到更难定位的位置。

## 8. Scheduler、输出和观测文件

### 8.1 Scheduler 与 Overlap

| 文件 | 修改目的 |
|---|---|
| `managers/scheduler.py` | 接入 Confidence Budget、动态控制和 DSpark 记录查询 |
| `managers/overlap_utils.py` | 使用异步 D2H Ring Buffer 将 GPU Confidence 传给 CPU Planner |
| `managers/schedule_batch.py` | 保存 Verify Tier、Cap 和 Block Accept 统计 |
| `scheduler_components/batch_result_processor.py` | 更新请求级接受长度与 Cap 统计 |
| `scheduler_components/metrics_reporter.py` | 聚合 DSpark 性能指标 |
| `speculative/base_spec_worker.py` | 增加请求结束通知，用于清理在线估计状态 |

### 8.2 API 输出链路

以下文件把 DSpark 指标从 Scheduler 传递到 API 和 Benchmark 输出：

- `managers/io_struct.py`；
- `managers/output_streamer.py`；
- `managers/detokenizer_manager.py`；
- `managers/multi_tokenizer_mixin.py`；
- `managers/tokenizer_manager.py`；
- `benchmark/serving.py`。

新增的主要指标包括：

- `spec_cap_length`；
- `spec_block_accept_length`；
- `spec_cap_lens_histogram`。

### 8.3 Request Slot Generation

`mem_cache/memory_pool.py` 和 `disaggregation/decode.py` 为 Request Slot 增加 Generation Counter，避免槽位被新请求复用后，DSpark Block Accept 在线估计器将新旧请求状态混淆。

## 9. STS、SPS 与性能观测

### 9.1 STS

文件：

- `python/sglang/benchmark/dspark_sts_fit.py`；
- `python/sglang/srt/speculative/dspark_components/dspark_sts.py`。

STS 对 Confidence Head 原始 logits 执行逐位置温度缩放：

```text
q[k] = sigmoid(logit[k] / temperature[k])
survival[k] = product(q[0:k+1])
```

当前实现通过 `0.1～10.0` 的对数网格搜索，为每个 Block 位置选择使 ECE 最小的 Temperature。

STS 数学逻辑与硬件无关，但校准数据应接近真实线上请求分布。

### 9.2 SPS

文件：

- `python/sglang/benchmark/dspark_sps_profiler.py`；
- `python/sglang/srt/speculative/dspark_components/dspark_sps.py`。

SPS 描述部署环境中的执行成本：

```text
T(batch_size, verify_token_count)
```

Planner 使用 SPS 判断增加 Verify Token 是否值得。SPS 表的格式和插值算法与硬件无关，但表中数据与下列条件强相关：

- GPU/NPU 型号；
- Target/Draft 模型；
- TP、DP、EP 配置；
- Attention Backend；
- Graph 配置；
- 量化方式；
- Block Size；
- Batch 分布。

CUDA 上采集的 SPS 表不能直接用于 NPU，NPU 适配完成后必须重新采集。

## 10. 硬件相关性分类

### 10.1 强硬件相关

| 模块 | 原因 |
|---|---|
| `dspark_components/kernels/*` | Triton Kernel、CUDA Tensor 和 CUDA 内存访问模型 |
| `ragged_verify_kernels.py` | Triton 变长布局与前缀和 Kernel |
| `decode_cuda_graph_runner.py` | CUDA Graph、Stream、Event、固定地址 Buffer 和 Token Bucket |
| `deepseek_v4_backend.py` | CUDA Attention metadata、SWA/MLA Page Table 和 Graph Replay |
| `dsv4/attn_metadata_kernels.py` | Torch/Triton DSV4 metadata Kernel |
| `flashattention_backend.py` | FlashAttention 的变长 Q/K metadata |
| `trtllm_mha_backend.py` | TRT-LLM MHA 和 `cum_seq_lens_q` |
| `deepseek_v4_dspark.py` | CUDA Stream、SWA KV、Fused RoPE/Norm、TP/MoE 通信 |
| `dspark_kv_inject.py` | KV Pool 物理布局、Scatter 和 Prefix Commit |

### 10.2 中等硬件相关

| 模块 | 原因 |
|---|---|
| `overlap_utils.py` | Device Stream/Event、Pinned Memory、异步 D2H |
| `dspark_planner.py` | 算法可复用，但依赖硬件 SPS 和 DP Graph Tier |
| `model_runner.py` | Hidden Capture 接口通用，但执行和 Tensor Device 相关 |
| `pool_configurator.py` | 内存计算通用，但 KV Layout 和硬件容量相关 |
| TP/DP/MoE 路径 | CUDA 使用 NCCL，NPU 需要 HCCL 对等实现 |

### 10.3 基本硬件无关

- DSpark checkpoint 配置解析；
- Markov Head 数学定义；
- Confidence Head 数学定义；
- STS 拟合与加载；
- SPS JSON 格式和插值算法；
- Prefix Survival 概率；
- Block Accept Estimator；
- Observability 数据结构；
- API 输出和统计字段；
- Spec Registry 和参数语义。

这些模块仍需检查是否存在写死的设备字符串，但算法本身可以直接复用。

## 11. Attention 和 Graph 修改

### 11.1 Attention Backend 能力

`base_attn_backend.py` 新增 `supports_ragged_verify_graph` 能力标记。不同 Backend 的处理方式为：

| Backend | PR 中的行为 |
|---|---|
| DeepSeek V4 CUDA | 支持 DSpark Draft Block 和 Ragged Target Verify |
| FlashAttention | 增加 Ragged Verify metadata |
| TRT-LLM MHA | 支持变长 `cum_seq_lens_q` |
| FlashInfer | 对不支持的 Ragged Graph 组合显式报错 |
| HIP Radix | 对尚不支持的 DeepSeek V4 Ragged Graph 显式报错 |

Ascend Attention 需要增加等价能力，至少能够处理固定长度 `TARGET_VERIFY`；完整 Compact 模式还必须消费逐请求 `verify_lens` 和 `qo_indptr`。

### 11.2 Graph Key 变化

传统 Decode Graph 通常以 Batch Size 为 Key：

```text
graph key = batch_size
```

DSpark Compact Verify 改为以总 Verify Token Bucket 为 Key：

```text
graph key = graph_num_tokens
```

这允许多个不同的逐请求 Verify Length 组合复用同一个 Token Tier Graph，例如真实需要 37 个 Verify Token 时回放 40-Token Graph，而不是按 `batch_size × 最大长度` 执行。

## 12. NPU 适配目标拆分

建议将 NPU 适配拆成三个可独立验收的阶段。

### 阶段 A：Dense DSpark Static Verify

目标：先实现正确性，不追求完整动态调度收益。

范围：

- Qwen3 等 Dense Target；
- `SGLANG_RAGGED_VERIFY_MODE=static`；
- 固定长度 Target Verify；
- Greedy Sampling；
- Torch Reference Accept/Commit；
- 暂不启用 NPU Graph；
- 暂不启用 Confidence Relay 和 SPS 动态调度。

必须完成：

1. 解除入口 CUDA-only 限制；
2. Target 模型捕获指定层 Hidden State；
3. Draft 模型在 NPU 上加载和执行；
4. Target Embedding、LM Head 与 Draft 共享；
5. Target Hidden Projection；
6. Draft KV Inject；
7. 固定长度 Target Verify；
8. Greedy Accept、Bonus Token 和 KV Commit；
9. 与不开启 DSpark 的 Target 输出逐 Token 对齐。

阶段 A 能证明 DSpark 算法链路在 NPU 上正确，但可能还没有明显加速。

### 阶段 B：Dynamic Ragged/Compact Verify

目标：获得 DSpark 动态验证收益。

必须完成：

1. Ascend Attention 消费 `RaggedVerifyLayout`；
2. 支持逐请求 Query Length；
3. 实现 Compact Verify IDs；
4. 实现 Compact Hidden Scatter；
5. 实现 Prefix KV Commit；
6. 实现 Verify Length Top-K 调度；
7. 支持 Confidence Head 和 STS；
8. 适配 NPU→CPU 异步 Confidence Relay；
9. 在 NPU 上采集 SPS 表；
10. 支持 Rejection Sampling 和混合 Sampling Batch。

### 阶段 C：NPU Graph 与 DeepSeek V4

目标：恢复 CUDA 路径上的主要性能优化和复杂模型支持。

必须完成：

1. 使用 `torch.npu.NPUGraph` 实现 Token Bucket Graph；
2. 验证 Ragged metadata 是否能进入或配合 Graph Replay；
3. 实现 NPU Stream/Event 同步；
4. 将 Accept、Scatter、Commit 等小算子融合；
5. 实现 DeepSeek V4 SWA/MLA Draft Attention；
6. 实现 `DSV4NPUTokenToKVPool` 的 DSpark 写入；
7. 适配 HC Head、RoPE、Norm 和 KV Store；
8. 适配 HCCL TP/DP/EP 通信；
9. 验证 Markov W2 TP Shard；
10. 验证 MoE 和 DP Attention 下的 Graph Tier 一致性。

## 13. NPU Kernel 优先级

### P0：正确性必需

- Greedy Accept；
- Accept Length Finalize；
- Bonus Token Gather；
- Verify Window 构造；
- Prefix KV Commit；
- Target Hidden → Draft KV Inject；
- 输出 Token 构造。

### P1：Compact Verify 必需

- Compact Verify IDs；
- Compact Row Index；
- Compact Hidden Scatter；
- Ragged `qo_indptr`；
- Verify Length Scheduling；
- Commit/Inject Layout。

### P2：性能优化

- Markov Sampling 融合；
- Confidence Head 融合；
- 多层 KV Projection 融合；
- Norm/RoPE/KV Store 融合；
- NPU Graph Epilogue；
- 多 Stream Draft Attention。

## 14. NPU 测试建议

建议增加以下专项测试：

```text
test_npu_dspark_static_greedy.py
test_npu_dspark_target_hidden_capture.py
test_npu_dspark_kv_inject.py
test_npu_dspark_accept_parity.py
test_npu_dspark_sampling.py
test_npu_dspark_ragged_verify.py
test_npu_dspark_compact.py
test_npu_dspark_graph.py
test_npu_dspark_dp_attention.py
test_npu_dspark_dsv4.py
```

测试应覆盖：

- Torch Reference 与 NPU Kernel 一致性；
- DSpark 与 Target-only 输出一致性；
- Batch 内不同 Verify Length；
- 请求动态加入、结束和 Slot 复用；
- Prefix Cache 命中；
- Greedy、Temperature、Top-K、Top-P；
- TP、DP Attention、MoE/EP；
- Graph Capture 和 Replay；
- 长上下文和 KV Pool 边界；
- SPS 表重新采集后的吞吐收益。

## 15. 风险点

### 15.1 通用路径回归面较大

PR 修改了通用 Scheduler、CUDA Graph、Attention metadata、KV Pool 和输出路径。NPU 适配时应避免为了 DSpark 破坏已有 EAGLE、EAGLE3、DFLASH 和普通 Decode。

### 15.2 Graph 与动态 Shape 冲突

Compact Verify 的核心是逐请求动态长度，而 Graph 偏好固定 Shape。Token Bucket 可以缓解这一矛盾，但需要保证：

- Buffer 地址稳定；
- Graph Tier 选择一致；
- DP Rank 使用相同 Collective Shape；
- Padding Token 不参与错误的 KV Commit；
- Replay 前后的 metadata 不残留旧请求数据。

### 15.3 NPU 异步 D2H 行为需要实测

Confidence Relay 依赖 Stream、Event、Pinned Memory 和异步 D2H。不能仅将 `torch.cuda` API 替换成 `torch.npu`，需要验证拷贝是否真正与计算重叠。

### 15.4 DeepSeek V4 应后置

DeepSeek V4 同时涉及 MLA、SWA、MoE、HC、TP Shard 和专用 KV Pool，建议先完成 Dense Static/Compact，再适配 DeepSeek V4。

## 16. 结论

PR #30261 的本质不是增加两个模型类，而是引入一套跨越模型、调度、Attention、Graph、Kernel、KV Cache、输出和观测的 DSpark 推理系统。

从 NPU 角度可以将其划分为：

```text
算法与配置层              基本可复用
Target Hidden/Draft KV     需要设备与 KV Pool 适配
Static Target Verify       可作为第一阶段正确性目标
Ragged Compact Verify      需要 Ascend Attention 和 NPU Kernel
Confidence Relay/SPS       需要 NPU 性能标定
Graph/Fusion               需要 NPUGraph 和算子融合
DeepSeek V4                需要专用 SWA/MLA/MoE 适配
```

推荐实施顺序为：

```text
Dense + Static + Greedy
        ↓
Dynamic Confidence + STS + NPU SPS
        ↓
Ragged Compact Verify
        ↓
NPU Graph 与 Kernel 融合
        ↓
DeepSeek V4 SWA/MLA/MoE
```

第一阶段重点是证明无损正确性，第二阶段实现动态 Verify，第三阶段再恢复 Graph 和融合带来的性能收益。
