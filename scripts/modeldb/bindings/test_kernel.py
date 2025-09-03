import argparse
import time
import torch
from importlib import import_module
from pathlib import Path
import sys
import nvtx
# 使得可以导入 scripts.utils.pq_utils
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from scripts.utils.pq_utils import sa_decode_4d


def build_kernel(Ns: int, d: int, M: int, C: int):
    bindings = import_module("bindings")
    fname = f"flash_decoding_allocated_buffer_f16u8_Ns{Ns}Lt{d}d{d}M{M}C{C}"
    try:
        return getattr(bindings, fname)
    except AttributeError:
        raise RuntimeError(f"Kernel {fname} 未编译。请在 scripts/modeldb/bindings/setup.py 中添加该组合并重新编译安装。")
def build_paged_kernel(Ns: int, d: int, M: int, C: int):
    bindings = import_module("bindings")
    fname = f"flash_decoding_allocated_paged_buffer_f16u8_Ns{Ns}Lt{d}d{d}M{M}C{C}"
    try:
        return getattr(bindings, fname)
    except AttributeError:
        raise RuntimeError(f"Kernel {fname} 未编译。请在 scripts/modeldb/bindings/setup.py 中添加该组合并重新编译安装。")


def main():
    p = argparse.ArgumentParser(description="Kernel micro-benchmark & correctness check")
    p.add_argument("--Ns", type=int, default=16, help="并行块数，与编译组合一致")
    p.add_argument("--d", type=int, default=128, help="头维度 d，与编译组合一致")
    p.add_argument("--M", type=int, default=64, help="PQ 子空间数 M，与编译组合一致")
    p.add_argument("--C", type=int, default=256, help="码本大小 C=2**nbits，与编译组合一致")
    p.add_argument("--T", type=int, default=1000, help="历史 token 数（非残差部分长度）")
    p.add_argument("--r", type=int, default=17, help="residual 有效 token 数，0<=r<=d")
    p.add_argument("--iters", type=int, default=100, help="计时迭代次数")
    p.add_argument("--paged", action="store_true", help="使用页式内核")
    p.add_argument("--page_size", type=int, default=64, help="每页包含的token数")
    args = p.parse_args()

    bs, nh = 1, 32
    d, M, C = args.d, args.M, args.C
    Lt = d
    T = args.T
    Ns = args.Ns
    r = args.r

    scalar_t = torch.float16
    code_t = torch.uint8
    device = "cuda"

    kernel = build_kernel(Ns, d, M, C) if not args.paged else build_paged_kernel(Ns, d, M, C)

    # 输入张量构造（形状需严格匹配）
    query = torch.randn(bs, nh, 1, d, dtype=scalar_t, device=device)
    key_codes   = torch.randint(0, C, (bs, nh, T, M), dtype=code_t, device=device)
    value_codes = torch.randint(0, C, (bs, nh, T, M), dtype=code_t, device=device)
    key_cents   = torch.randn(M, C, d // M, dtype=scalar_t, device=device)
    value_cents = torch.randn(M, C, d // M, dtype=scalar_t, device=device)
    key_residuals   = torch.randn(bs, nh, Lt, d, dtype=scalar_t, device=device)
    value_residuals = torch.randn(bs, nh, Lt, d, dtype=scalar_t, device=device)

    # 预先分配的分块累加缓冲（大小 (bs, nh, Ns+1, d) 与 (bs, nh, Ns+1)）
    partial_out = torch.empty(bs, nh, Ns+1, d, dtype=scalar_t, device=device)
    partial_lse = torch.empty(bs, nh, Ns+1, dtype=scalar_t, device=device)

    default = nvtx.start_range(message="sdpa_mode", color="yellow")
    # print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    
    if not args.paged:
        # 正确性：与 PyTorch SDPA 对比
        out = kernel(query, key_codes, value_codes, key_cents, value_cents,
                    key_residuals, value_residuals, r, partial_out, partial_lse)
    else:
        # 正确性：与 PyTorch SDPA 对比
        out = kernel(query, key_codes, value_codes, key_cents, value_cents,
                    key_residuals, value_residuals, r, partial_out, partial_lse)
    nvtx.end_range(default)
        
        
    # print(out)
    K_hat = sa_decode_4d(key_codes, key_cents)
    V_hat = sa_decode_4d(value_codes, value_cents)
    K_full = torch.cat([K_hat, key_residuals[:, :, :r, :]], dim=2)
    V_full = torch.cat([V_hat, value_residuals[:, :, :r, :]], dim=2)

    out_ref = torch.nn.functional.scaled_dot_product_attention(
        query, K_full, V_full, is_causal=True
    )

    mae = (out - out_ref).abs().mean().item()
    mxe = (out - out_ref).abs().max().item()
    print(f"shape: {tuple(out.shape)}, MAE: {mae:.4e}, MaxAbsErr: {mxe:.4e}")

    # 性能：预热 + 多次计时
    for _ in range(10):
        kernel(query, key_codes, value_codes, key_cents, value_cents,
                key_residuals, value_residuals, r, partial_out, partial_lse)
           
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        kernel(query, key_codes, value_codes, key_cents, value_cents,
               key_residuals, value_residuals, r, partial_out, partial_lse)
       
    torch.cuda.synchronize()
    avg_ms = (time.time() - t0) / args.iters * 1e3
    print(f"avg time per call: {avg_ms:.3f} ms  (Ns={Ns}, d={d}, M={M}, C={C}, T={T}, r={r})")


if __name__ == "__main__":
    main()