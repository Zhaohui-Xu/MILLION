

# python test_kernel.py --Ns 32 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 
# python test_kernel.py --Ns 32 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# python test_kernel.py --Ns 16 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 
# python test_kernel.py --Ns 16 --d 128 --M 16 --C 256 --T 1000 --r 17 --iters 100 
# python test_kernel.py --Ns 16 --d 128 --M 16 --C 256 --T 1000 --r 17 --iters 100 --paged
# python test_kernel.py --Ns 16 --d 128 --M 1 --C 256 --T 1000 --r 17 --iters 100 --paged
# python test_kernel.py --Ns 16 --d 128 --M 1 --C 256 --T 1000 --r 17 --iters 100 --paged


# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active /root/miniconda3/envs/million/bin/python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# ncu --metrics matmul_wmma_kernel python3 test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
pkill dcgm
pkill dcgm
pkill dcgm
pkill dcgm
# ncu --devices 0 --nvtx --nvtx-include "sdpa_mode" --set roofline --force-overwrite --csv  --print-details all --print-metric-name name --log-file $dir/$output python3 atten_ncu_prof.py --mode ${sdpa_mode}
# ncu --devices 0 --nvtx --nvtx-include "unroll_paged" --set roofline --force-overwrite -o unroll_kernel --print-details all --print-metric-name name python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
ncu --devices 0 --nvtx --nvtx-include "unroll_paged" --set roofline --force-overwrite -o antiquant_kernel --print-details all --print-metric-name name python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100


