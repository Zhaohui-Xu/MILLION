
pkill dcgm
pkill dcgm
pkill dcgm
# ncu --devices 0 --nvtx --nvtx-include "sdpa_mode" --set roofline --force-overwrite --csv  --print-details all --print-metric-name name --log-file $dir/$output python3 atten_ncu_prof.py --mode ${sdpa_mode}
# ncu --devices 0 --nvtx --nvtx-include "unroll_paged" --set roofline --force-overwrite -o unroll_kernel --print-details all --print-metric-name name python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100 --paged
# ncu --devices 0 --nvtx --nvtx-include "unroll_paged" --set roofline --force-overwrite -o antiquant_kernel --print-details all --print-metric-name name python test_kernel.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100

# ncu --devices 0 --nvtx --nvtx-include "unroll_paged" --set roofline --force-overwrite -o unroll_paged --print-details all --print-metric-name name python kernel_sample.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100

get_ncu() {
    ncu --devices 0 --nvtx --nvtx-include "unroll_paged"\
    --set $1 --force-overwrite -o unroll_paged \
    --print-details all --print-metric-name name \
    python kernel_sample.py --Ns 8 --d 128 --M 32 --C 256 --T 1000 --r 17 --iters 100
}
# get_ncu roofline
get_ncu full