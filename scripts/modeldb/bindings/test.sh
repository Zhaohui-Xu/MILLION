
# 效果会好一点的参数
python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 
python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active /root/miniconda3/envs/million/bin/python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# ncu --metrics matmul_wmma_kernel python3 test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
pkill dcgm
pkill dcgm
pkill dcgm
pkill dcgm
# ncu --devices 0 --nvtx --nvtx-include "sdpa_mode" --set roofline --force-overwrite --csv  --print-details all --print-metric-name name --log-file $dir/$output python3 atten_ncu_prof.py --mode ${sdpa_mode}
ncu --devices 0 --nvtx --nvtx-include "sdpa_mode" --set roofline --force-overwrite -o test --print-details all --print-metric-name name python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged


#!/bin/bash
# simplified_benchmark.sh

# echo "🚀 Starting simplified kernel benchmark (identical input format)"

# # 快速测试
# echo "Running quick test..."
# python benchmark_kernels.py --quick

# # 测试标准配置组合
# echo "Testing standard configurations..."
# python benchmark_kernels.py \
#     --Ns_list 8 16 32 \
#     --d_list 64 128 \
#     --M_list 32 64 \
#     --T_list 500 1000 2000 \
#     --r_list 16 32 \
#     --warmup 10 \
#     --iters 100

# # 测试高并行度场景
# echo "Testing high parallelism scenarios..."
# python benchmark_kernels.py \
#     --Ns_list 16 32 \
#     --d_list 128 \
#     --M_list 64 \
#     --T_list 2000 5000 \
#     --r_list 32 \
#     --output_dir results_high_parallelism

# echo "✅ All benchmarks completed!"