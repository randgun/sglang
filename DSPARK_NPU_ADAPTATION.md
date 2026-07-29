# SGLang DSpark 实现与 NPU 适配方向

## 1. 文档目的

本文基于当前 SGLang 仓库代码，汇总 DSpark 的代码组织、模型结构、推理链路、动态验证机制，以及 Ascend NPU 适配所需的主要工作。本文描述的是当前代码快照，后续接口和支持范围可能继续变化。

## 2. DSpark 在 SGLang 中的定位

DSpark 是 SGLang 中一套完整的投机解码实现，而不是单独的模型文件。它由以下部分共同组成：

- 半自回归草稿模型；
- Markov Head 和 Confidence Head；
- 动态验证预算调度器；
- 目标模型验证与接受算法；
- Hidden State/KV Cache 回注；
- Triton/CUDA Kernel 与 CUDA Graph；
- SPS、STS 和运行时观测工具。

其核心过程是：

1. 草稿模型一次生成长度为 `gamma` 的候选 Token Block；
2. Confidence Head 预测每个候选位置被接受的概率；
3. Planner 根据置信度和设备性能曲线决定每个请求验证多远；
4. 目标模型并行验证候选 Token；
5. 提交接受的 Token，并将目标 Hidden State 写回草稿 KV Cache。

验证窗口包含一个 Anchor Token 和 `gamma` 个 Draft Token：

```text
verify_num_draft_tokens = speculative_num_draft_tokens = gamma + 1
```

当前默认 `gamma` 为 7。

## 3. 代码结构

```text
python/sglang/srt/
├── models/
│   ├── dspark.py                    # 通用稠密 DSpark 模型
│   └── deepseek_v4_dspark.py        # DeepSeek V4 专用实现
├── speculative/
│   ├── dspark_components/
│   │   ├── dspark_worker_v2.py      # 总控 Worker
│   │   ├── dspark_config.py         # Checkpoint/运行参数解析
│   │   ├── dspark_draft.py          # Draft Block 生成
│   │   ├── dspark_planner.py        # 验证预算与长度调度
│   │   ├── dspark_verify.py         # Target Verify 与接受提交
│   │   ├── dspark_kv_inject.py      # Target Hidden/KV 回注
│   │   ├── dspark_sps.py            # 设备性能代价表
│   │   ├── dspark_sts.py            # 置信度温度校准
│   │   ├── dspark_observability.py  # 性能和置信度观测
│   │   └── kernels/
│   │       ├── dspark_accept.py
│   │       ├── dspark_schedule.py
│   │       ├── dspark_verify_window.py
│   │       ├── dspark_attn_metadata.py
│   │       └── dspark_draft_model.py
│   └── ragged_verify.py
└── arg_groups/speculative_hook.py
```

## 4. 模型结构

### 4.1 半自回归 Draft

DSpark Draft Backbone 一次计算整个 Block 的基础 Logits，然后 Markov Head 按位置注入前序 Token 的影响：

```text
Draft Backbone
      │
      ├── base_logits[0]
      ├── base_logits[1]
      └── ... base_logits[gamma-1]
                │
                ▼
Markov Head(previous_token, hidden_state)
                │
                ▼
        corrected step logits
                │
                ▼
        draft token block
```

昂贵的 Backbone 不需要针对 Block 中每个位置重复运行，因此源码将其称为 Semi-AR Draft。

通用实现支持三类 Markov Head：

| 类型 | 说明 |
|---|---|
| `vanilla` | 将前一个 Token 的低秩 Embedding 投影为 Logits Bias |
| `gated` | 使用 Hidden State 控制 Markov Bias |
| `rnn` | 在 Block 内维护轻量递归状态 |

Checkpoint 必须提供有效的 `markov_rank > 0`。

### 4.2 Confidence Head

Confidence Head 为每个 Draft 位置输出一个接受概率：

```text
draft hidden [+ markov embedding]
               │
               ▼
          Linear → Sigmoid
               │
               ▼
      [c1, c2, ..., c_gamma]
```

Planner 使用累积乘积估计前缀存活概率：

```text
P(length >= 1) = c1
P(length >= 2) = c1 * c2
P(length >= 3) = c1 * c2 * c3
```

Confidence 只决定“值得验证多远”，不会直接判定 Token 正确。最终接受结果仍由目标模型决定，因此置信度或 STS 校准误差不会破坏投机解码的无损性质。

### 4.3 模型类型

当前主要有两类实现：

- `Qwen3DSparkModel`：通用稠密 DSpark，已有 Qwen3 端到端测试；
- `DeepseekV4ForCausalLMDSpark`：包含 DeepSeek V4 专用 Attention、HC Head、TP Shard 和 SWA/MLA KV 写入。

Draft 模型会复用 Target 模型的 Embedding 和 LM Head，以减少大词表参数的显存重复占用。

## 5. 推理链路

### 5.1 Prefill

DSpark Prefill 要求 Target 模型捕获指定层的辅助 Hidden State：

```text
Target Prefill
    ├── 正常生成首个 Token
    └── 捕获 Auxiliary Hidden States
                │
                ▼
       project_target_hidden
                │
                ▼
        写入 Draft KV Cache
```

### 5.2 Decode

一轮 Decode 的主链路为：

```text
当前上下文 / 上轮 Bonus Token
              │
              ▼
分配 Verify Window 与 KV 位置
              │
              ▼
DraftBlockProposer
  ├── Draft Backbone
  ├── Markov Head 生成 gamma 个候选 Token
  └── Confidence Head 生成逐位置置信度
              │
              ▼
DSparkVerifyPlanner
  ├── 计算总验证预算
  └── 为每个请求分配 verify_len
              │
              ▼
Target Model Verify
              │
              ▼
AcceptGreedy / AcceptSampling
  ├── correct_len
  ├── bonus token
  ├── commit_len
  └── new_seq_len
              │
              ▼
提交 Target Hidden/KV 到 Draft Cache
              │
              ▼
进入下一轮 Decode
```

DSpark 支持 Greedy、Rejection Sampling，以及同一 Batch 中两者混合。最终结果包括接受长度、Bonus Token、提交长度、新序列长度和对外输出 Token。

## 6. 动态验证

DSpark 支持三种模式：

| 模式 | 行为 |
|---|---|
| `static` | 所有请求验证固定的 `gamma + 1` 个 Token |
| `cap-accept` | 限制每个请求允许接受到的位置，但目标计算仍接近固定形状 |
| `compact` | 每个请求采用不同验证长度，并将真实验证 Token 压紧执行 |

环境变量为：

```bash
SGLANG_RAGGED_VERIFY_MODE=static
SGLANG_RAGGED_VERIFY_MODE=cap-accept
SGLANG_RAGGED_VERIFY_MODE=compact
```

默认值为 `static`。Compact 模式示例：

```text
固定窗口：A=8, B=8, C=8
动态长度：A=6, B=2, C=4
压紧验证：[a0..a5, b0..b1, c0..c3]
```

Compact 能真正减少 Target Forward Token 数，是 DSpark 动态验证收益的关键。

### 6.1 验证预算

Planner 结合置信度和 SPS 性能表，近似最大化：

```text
预期接受 Token 数 / 预测 Step 时间
```

它会计算累计存活概率、枚举总验证预算、查询对应 Step 时间，然后将最优预算按置信度分配给各请求。

### 6.2 SPS 与 STS

- SPS 表描述特定设备在不同 Batch/验证 Token 数下的性能；
- STS 为每个 Block 位置设置独立温度，使 Confidence 更接近真实接受概率。

相关参数为：

```bash
--speculative-dspark-sps-table-path <sps.json>
--speculative-dspark-confidence-sts-path <sts.json>
```

未提供真实 SPS 表时，动态预算容易退化为 Verify-All，难以产生有效调度收益。

## 7. 当前 CUDA 优化

当前实现包含两层 CUDA Graph 融合：

1. Draft Graph：把 Greedy Markov Sampling 和 Confidence Head 作为 Tail Hook 融入 Draft Graph；
2. Verify Epilogue：在 Target Verify Graph 内完成 Compact Scatter、Accept、Finalize、Output Token 构造和 Hidden/KV Commit。

这些优化减少 Python 调度、Host 同步和小 Kernel 开销，也是 NPU 完整性能适配的主要难点。

## 8. NPU 已有基础与差距

SGLang NPU backend 已具备：

- Ascend Attention；
- 固定长度 `TARGET_VERIFY`；
- `torch.npu.NPUGraph` 与 NPU Graph Runner；
- EAGLE3 投机解码；
- NPU KV Pool 和 Paged Allocator；
- DeepSeek V4 专用 NPU Attention、KV Pool 和 Allocator；
- HCCL、MoE 和部分量化支持。

当前主要差距是：

1. `_handle_dspark()` 明确拒绝非 CUDA 设备；
2. Kernel Dispatcher 只区分 CUDA 与 Torch Reference；
3. Ascend Attention 尚未消费 DSpark `RaggedVerifyLayout`；
4. 部分 NPU Graph 逻辑只判断 `is_dflash()`，未覆盖 `is_dflash_family()`；
5. Observability、异步 D2H 和多 Stream 直接调用 `torch.cuda`；
6. DeepSeek V4 DSpark 使用 CUDA JIT Norm/RoPE 和 CUDA KV 写入 Kernel；
7. 尚无 DSpark NPU 专项 CI。

## 9. NPU 适配路线

### P0：Qwen3 Static Eager Greedy

建议首版限制为：

```text
Target/Draft：Qwen3 Dense
模式：static
执行：Eager，关闭 Graph
采样：Greedy
并行：TP=1
精度：BF16
```

需要完成：

- 放开 CUDA-only 校验，同时对 NPU 显式限制 Static 模式；
- 验证 Torch Reference Kernel 在 NPU 上正确执行且不发生隐式 CPU Fallback；
- 打通 Target Auxiliary Hidden Capture；
- 打通 Prefill Hidden 到 Draft KV 的注入；
- 验证固定长度 Target Verify；
- 验证 Greedy Accept、Bonus Token、KV Commit 和下一轮输入；
- 与 Target-only 做逐 Token 一致性测试。

这一阶段只建立最小正确性闭环。

### P1：Sampling、TP 与稳定性

随后增加：

- Rejection Sampling 和 Mixed Batch；
- TP2、TP4 与 HCCL；
- Prefix Cache、长上下文、EOS/Stop/Abort；
- Scheduler 压力测试。

需要重点验证 NPU 上的 `softmax`、`exponential_`、`multinomial`、Gather/Scatter 和随机数可复现性。

### P2：Static NPUGraph

复用现有 EAGLE/NPU Graph 基础，完成：

- 将适当的 `is_dflash()` 判断改为 `is_dflash_family()`；
- Replay 时更新 DSpark Draft `input_embeds`；
- 更新 Position、Sequence Length 和 KV Location；
- 验证 Draft Tail Hook 在 NPUGraph 中的可捕获性；
- 处理 Target Hidden 输出 Buffer；
- 确认 Padding 不会污染有效 KV；
- 对 Graph 和 Eager 做逐 Tensor 一致性测试。

### P3：Compact Ragged Verify

这是 NPU 高性能适配的核心。Ascend Attention 需要支持每请求不同的 Query Length，并消费：

```text
verify_lens
extend_start_loc
qo_indptr
actual_seq_lengths_q
actual_seq_lengths_kv
block_tables
max_q_len
```

同时需要实现或适配 NPU 版本的：

- Verify Length Top-k；
- Compact Row Index 和 Verify IDs；
- Compact-to-Strided Scatter；
- Greedy/Sampling Accept；
- Accept Length Finalize；
- Commit Inject Layout；
- Output Token 构造。

首个 Compact 版本可把 Epilogue 放在 Graph 外保证正确性，之后再融合优化。

### P4：Compact NPUGraph 与融合 Epilogue

Compact Graph 应按总验证 Token 数选择 Graph：

```text
graph key = graph_num_tokens
```

建议使用有限 Token Bucket，例如 `8, 16, 32, 64, 128, 256`。Replay 时需要动态更新 `verify_lens`、Q/KV Length、Block Table 和 Request Index，并保证 Padding Token 不污染 KV。

性能版本还应在 NPUGraph 中融合 Scatter、Accept、Finalize 和 Commit，可考虑提供：

```text
npu_dspark_accept_finalize
npu_dspark_scatter_compact
npu_dspark_commit_hidden_kv
```

### P5：DeepSeek V4 DSpark

DeepSeek V4 建议独立适配，主要包括：

- NPU `fused_q_norm_rope` 和 Inverse RoPE；
- 多 Stage KV Projection；
- 为 `DSV4NPUTokenToKVPool` 实现 DSpark SWA KV 写入；
- 替换 CUDA `fused_k_norm_rope_flashmla` 路径；
- 验证 HC Head、Markov W2 TP Shard 和 Confidence Head；
- 验证 NPU Paged Compress State；
- 依次支持 TP、MoE、DP Attention 和多机 HCCL。

推荐顺序：

```text
TP1 → TP2/TP4 → MoE TP → DP Attention → 多机 HCCL
```

## 10. 代码重构建议

### 10.1 Kernel 三路分发

建议将当前：

```text
CUDA → Triton
其他 → Torch
```

改为：

```text
CUDA → Triton/CUDA Kernel
NPU  → NPU Kernel，缺失时显式 Torch Fallback
CPU  → Torch Reference
```

从而明确区分正确性路径和性能路径，避免 NPU 静默进入低性能实现。

### 10.2 Device Runtime 抽象

逐步替换直接调用的：

```text
torch.cuda.Stream
torch.cuda.Event
torch.cuda.current_stream
torch.cuda.is_current_stream_capturing
```

优先使用 `torch.get_device_module(device)`，必要时在 NPU backend 中提供适配。

### 10.3 能力声明

建议增加明确的 Backend/Platform 能力：

```text
supports_dspark_static_verify
supports_ragged_verify
supports_ragged_verify_graph
supports_dspark_graph_epilogue
supports_async_device_observability
```

ServerArgs 根据能力提前拒绝非法组合，避免运行到深层 Kernel 才失败。

## 11. NPU SPS Profiling

CUDA SPS 表不能直接复用于 NPU。需要针对以下维度分别 Profiling：

- 硬件型号、Target/Draft 模型；
- Eager 或 NPUGraph；
- TP1、TP2、TP4；
- Batch Size、总 Verify Token 数；
- 典型上下文长度。

应记录 Draft、Verify、Accept、Commit 和整步延迟，以及接受长度、输出吞吐、Graph 命中率、Host/Device 同步和 HCCL 占比。只有获得可靠的 NPU SPS 表，Planner 才能选择真正有收益的验证预算。

## 12. 测试与验收

建议新增：

```text
test/registered/ascend/basic_function/speculative_inference/
├── test_npu_dspark_greedy.py
├── test_npu_dspark_sampling.py
├── test_npu_dspark_graph.py
├── test_npu_dspark_compact.py
└── test_npu_dspark_dsv4.py
```

测试分为：

1. Kernel：Torch Reference 与 NPU Kernel 对比；
2. Worker：Prefill、Draft、Confidence、Verify、Accept、Commit 链路；
3. 端到端：比较 Target-only、Static Eager/Graph、Compact Eager/Graph；
4. 性能：确认收益能够抵消 Draft、调度和同步开销。

关键验收要求：

- Greedy 输出逐 Token 一致；
- Sampling 分布与 Target-only 等价；
- EOS、Stop、Max Tokens、Prefix Cache 和长上下文行为一致；
- 无隐式 CPU Fallback 和异常 Device Synchronize；
- Compact 模式能够减少真实 Target Verify Token 数并提升吞吐。

## 13. 总结

SGLang DSpark 的核心可以概括为：

```text
Semi-AR Draft
+ Markov Head
+ Confidence Head
+ 设备性能感知的动态验证预算
+ Compact Ragged Target Verify
+ Hidden/KV 回注
```

NPU 适配可以复用现有 Ascend Attention、EAGLE3、Target Verify、NPUGraph 和 KV Pool 基础。最小可用版本应优先完成 Qwen3 Static Eager Greedy；完整性能版本则必须补齐 Ascend Ragged Attention、NPU Kernel、Token Bucket Graph、融合 Epilogue 和 NPU SPS Profiling。DeepSeek V4 还涉及 CUDA JIT Norm/RoPE 与 KV 写入路径替换，适合作为独立专项推进。
