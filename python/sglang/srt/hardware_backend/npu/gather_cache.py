import torch
import triton
import triton.language as tl
import triton.testing

@triton.jit
def gather_paged_kv_kernel(
    cache_ptr,
    actual_seq_len_ptr,
    block_table_ptr,
    out_ptr,
    stride_bt_b,
    stride_out_b,
    page_size: tl.constexpr, 
    max_seq_len,             # 🌟 关键修改 1：去掉了 tl.constexpr
    BLOCK_SIZE: tl.constexpr,# BLOCK_SIZE 依然是 constexpr，但我们在外部固定它的值
):
    seq_block_pid = tl.program_id(0)
    batch_pid = tl.program_id(1)

    seq_offsets = seq_block_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    actual_len = tl.load(actual_seq_len_ptr + batch_pid)

    # 这里的 max_seq_len 变成了运行时的寄存器变量，比较操作依然高效
    valid_seq_mask = seq_offsets < max_seq_len
    read_mask = valid_seq_mask & (seq_offsets < actual_len)

    logical_block_indices = seq_offsets // page_size
    block_offsets = seq_offsets % page_size

    bt_ptrs = block_table_ptr + batch_pid * stride_bt_b + logical_block_indices
    physical_block_indices = tl.load(bt_ptrs, mask=read_mask, other=0)

    cache_indices = physical_block_indices * page_size + block_offsets
    cache_ptrs = cache_ptr + cache_indices

    data = tl.load(cache_ptrs, mask=read_mask, other=0.0)

    out_ptrs = out_ptr + batch_pid * stride_out_b + seq_offsets
    tl.store(out_ptrs, data, mask=valid_seq_mask)


def gather_kv_cache_triton(cache: torch.Tensor, actual_seq_len_kv: torch.Tensor, block_table: torch.Tensor, page_size: int):
    b, n = block_table.shape
    max_seq_len = n * page_size
    
    output = torch.empty((b, max_seq_len), device=cache.device, dtype=cache.dtype)
    
    # 🌟 关键修改 2：使用固定的 BLOCK_SIZE
    # 无论 max_seq_len 是 10 还是 2000，BLOCK_SIZE 永远是 128。
    # 这样底层生成的二进制机器码永远只有一份，绝不重新编译。
    BLOCK_SIZE = 128 
        
    grid = lambda meta: (
        triton.cdiv(max_seq_len, meta['BLOCK_SIZE']), # 这里的 cdiv 会动态改变 grid 的大小
        b                                             
    )
    
    gather_paged_kv_kernel[grid](
        cache_ptr=cache,
        actual_seq_len_ptr=actual_seq_len_kv,
        block_table_ptr=block_table,
        out_ptr=output,
        stride_bt_b=block_table.stride(0),
        stride_out_b=output.stride(0),
        page_size=page_size,
        max_seq_len=max_seq_len, # 正常传入变量
        BLOCK_SIZE=BLOCK_SIZE,   # 传入固定的 constexpr
    )
    
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],       # X轴：测试不同的序列长度
        x_vals=[2**i for i in range(8, 15)], # 从 256 测到 16384 (16K 上下文)
        line_arg='provider',       # 不同的测试方案作为不同的线
        line_vals=['triton'],      # 目前只测我们写的 triton 算子
        line_names=['Triton Gather'], 
        styles=[('blue', '-')],
        ylabel='Bandwidth (GB/s)', # Y轴：显存带宽
        plot_name='Paged-KV-Gather-Performance',
        args={'BATCH': 16, 'PAGE_SIZE': 128, 'HEAD_DIM': 1},
    )
)
def benchmark(BATCH, PAGE_SIZE, HEAD_DIM, seq_len, provider):
    device = 'cuda' # 在昇腾上调优时请改为 'npu'
    dtype = torch.float16
    
    # 1. 模拟底层巨大的全局 KV Cache 池子
    # 假设池子有 100,000 个 Token 的容量
    total_slots = 100000 
    # 为了模拟真实场景，我们把 head_dim 展平，把原来的 [x, 1] 扩充为模拟单头特征的 [x, head_dim]
    # 注意：测试脚本中我们需要改一下 cache 形状以产生足够的数据量压测带宽
    cache = torch.randn((total_slots, HEAD_DIM), device=device, dtype=dtype)
    
    # 2. 构造 block_table 和实际长度
    max_num_blocks = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    block_table = torch.randint(0, total_slots // PAGE_SIZE, (BATCH, max_num_blocks), device=device, dtype=torch.int32)
    
    # 模拟实际长度 (为了简单，这里假设所有请求都跑满了 seq_len)
    actual_seq_len_kv = torch.full((BATCH,), seq_len, device=device, dtype=torch.int32)

    # 3. 包装测试函数
    # 这里我们对之前的 gather_kv_cache_triton 稍微做一点适配，让它能处理 [x, head_dim] 的数据流
    # (如果不想改算子，把 HEAD_DIM 设为 1，测纯标量搬运也可以，但数据量太小可能测不准带宽)
    def test_fn():
        # 如果你的算子还是严格的 [x, 1]，请将上面 cache 设为 [total_slots, 1]
        gather_kv_cache_triton(cache, actual_seq_len_kv, block_table, PAGE_SIZE)

    # 4. 执行 Benchmark
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(test_fn, quantiles=quantiles)

    # 5. 计算有效显存带宽 (GB/s)
    # 计算公式：(读出数据量 + 写入数据量) / 执行时间
    # 读 Block Table: BATCH * max_num_blocks * 4 bytes (int32)
    # 读 Actual Len: BATCH * 4 bytes (int32)
    # 读 Cache: BATCH * seq_len * HEAD_DIM * 2 bytes (fp16)
    # 写 Output: BATCH * seq_len * HEAD_DIM * 2 bytes (fp16)
    
    num_bytes = (BATCH * max_num_blocks * 4) + \
                (BATCH * 4) + \
                (2 * BATCH * seq_len * HEAD_DIM * 2) # *2 for Read + Write
                
    gbps = num_bytes / (ms * 1e6)
    max_gbps = num_bytes / (min_ms * 1e6)
    min_gbps = num_bytes / (max_ms * 1e6)
    
    return gbps, max_gbps, min_gbps

if __name__ == '__main__':
    # 运行压测，结果会打印在控制台，并且如果你环境里有 pandas/matplotlib，还能保存出 csv 和 png 图表
    benchmark.run(print_data=True, show_plots=False)