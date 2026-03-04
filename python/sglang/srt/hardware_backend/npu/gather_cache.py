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

# =====================================================================
# 2. PyTorch Ground Truth (基准精度实现)
# =====================================================================
def gather_kv_cache_pytorch(cache, actual_seq_len_kv, block_table, page_size):
    b, n = block_table.shape
    max_seq_len = n * page_size
    output = torch.zeros((b, max_seq_len), device=cache.device, dtype=cache.dtype)
    
    for i in range(b):
        seq_len = actual_seq_len_kv[i].item()
        for j in range(seq_len):
            logical_block = j // page_size
            block_offset = j % page_size
            physical_block = block_table[i, logical_block].item()
            cache_idx = physical_block * page_size + block_offset
            output[i, j] = cache[cache_idx, 0]
            
    return output

# =====================================================================
# 3. 精度测试模块 (Unit Test)
# =====================================================================
def test_precision():
    print(">>> [1/2] 开始执行精度测试 (Precision Test)...")
    device = 'cuda' # 在昇腾上调优时请改为 'npu'
    dtype = torch.float16
    
    b, n, page_size = 16, 100, 128
    total_slots = 10000 
    
    cache = torch.randn((total_slots, 1), device=device, dtype=dtype)
    actual_seq_len_kv = torch.tensor([15, 32, 100, 5], device=device, dtype=torch.int32)
    block_table = torch.randint(0, total_slots // page_size, (b, n), device=device, dtype=torch.int32)

    out_ref = gather_kv_cache_pytorch(cache, actual_seq_len_kv, block_table, page_size)
    out_tri = gather_kv_cache_triton(cache, actual_seq_len_kv, block_table, page_size)

    try:
        torch.testing.assert_close(out_tri, out_ref, rtol=0.0, atol=0.0)
        print("✅ 精度测试通过！Triton 算子位级一致 (Bit-exact match)。\n")
        return True
    except AssertionError as e:
        print("❌ 精度测试失败！")
        print(e)
        return False

# =====================================================================
# 4. 性能压测模块 (Benchmark)
# =====================================================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],       
        x_vals=[2**i for i in range(8, 15)], # 长度从 256 到 16384
        line_arg='provider',       
        line_vals=['triton'],      
        line_names=['Triton Gather'], 
        styles=[('blue', '-')],
        ylabel='Bandwidth (GB/s)', 
        plot_name='Paged-KV-Gather-Performance',
        args={'BATCH': 16, 'PAGE_SIZE': 16}, 
    )
)
def benchmark(BATCH, PAGE_SIZE, seq_len, provider):
    device = 'npu' # 在昇腾上调优时请改为 'npu'
    dtype = torch.float16
    total_slots = 500000 # 保证池子够大
    
    cache = torch.randn((total_slots, 1), device=device, dtype=dtype)
    max_num_blocks = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    block_table = torch.randint(0, total_slots // PAGE_SIZE, (BATCH, max_num_blocks), device=device, dtype=torch.int32)
    actual_seq_len_kv = torch.full((BATCH,), seq_len, device=device, dtype=torch.int32)

    def test_fn():
        gather_kv_cache_triton(cache, actual_seq_len_kv, block_table, PAGE_SIZE)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(test_fn, quantiles=quantiles)

    # 计算有效显存带宽 (GB/s)
    # 对于 [x, 1] 的 fp16 张量，每个 Token 数据量为 2 Bytes
    bytes_per_token = 2
    
    num_bytes = (BATCH * max_num_blocks * 4) + \
                (BATCH * 4) + \
                (2 * BATCH * seq_len * bytes_per_token) # *2 因为是一次读 + 一次写
                
    gbps = num_bytes / (ms * 1e6)
    max_gbps = num_bytes / (min_ms * 1e6)
    min_gbps = num_bytes / (max_ms * 1e6)
    
    return gbps, max_gbps, min_gbps

# =====================================================================
# 5. 主程序入口
# =====================================================================
if __name__ == '__main__':
    # 第一步：拦截至关重要的精度错误
    is_accurate = test_precision()
    
    # 第二步：如果精度正确，则启动多维度的性能摸底
    if is_accurate:
        print(">>> [2/2] 开始执行性能压测 (Benchmark)...")
        print("正在 Warmup 并测试不同 Sequence Length 下的显存带宽...")
        # 自动调优并打印表格，你可以将 show_plots 设为 True 来保存趋势图
        benchmark.run(print_data=True, show_plots=False)