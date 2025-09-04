pkill dcgm
pkill dcgm
pkill dcgm


# 自动找到 libtorch 库所在的目录
# 确保你的 conda 环境已激活
LIBTORCH_LIB_DIR=$(python -c 'import torch; from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])')

if [ -z "$LIBTORCH_LIB_DIR" ]; then
    echo "Error: Could not find PyTorch library path. Please ensure PyTorch is installed."
    exit 1
fi
# LIBTORCH_LIB_DIR=/root/miniconda3/envs/million/lib/python3.12/site-packages/torch/lib
echo "Using LD_LIBRARY_PATH: $LIBTORCH_LIB_DIR"

# 将 libtorch 路径导出到环境变量，这样后续命令才能看到它
export LD_LIBRARY_PATH=$LIBTORCH_LIB_DIR:$LD_LIBRARY_PATH

# get_ncu() {
#     ncu --devices 0 --nvtx --nvtx-include  "Kernel_Test"\
#     --set $1 --force-overwrite -o unroll_paged \
#     --print-details all --print-metric-name name \
#     --import-source on --source-folders ".,core,.."\
#     ./kernel_test.bin --Ns 16 --d 128 --M 64 --T 1000 --r 17 --iters 1
# }
# # get_ncu roofline
# get_ncu full

# 定义一个更通用的分析函数
run_profiling() {
    local ncu_set="$1"
    local nvtx_filter="$2"
    local report_name="$3"
    local M_Size="$4"

    echo "================================================================="
    echo "Running NCU with set: '$ncu_set' for NVTX range: '$nvtx_filter'"
    echo "================================================================="

    local app_args="--Ns 16 --d 128 --M $M_Size --T 1000 --r 17 --iters 1"

    # 定义源代码所在的目录
    # '.' 表示当前目录 (Kernel_Test/)
    # '..' 表示上一级目录 (bindings/)
    # '../core' 表示上一级目录下的core目录
    local source_dirs=".,..,../core"

    ncu --devices 0 \
        --nvtx \
        --nvtx-include "$nvtx_filter" \
        --set "$ncu_set" \
        --force-overwrite \
        -o "$report_name" \
        --print-details all --print-metric-name name \
        --import-source on \
        --source-folders "$source_dirs" \
        ./kernel_test.bin $app_args

    if [ $? -eq 0 ]; then
        echo "✅ NCU profiling successful. Report saved to ${report_name}.ncu-rep"
        echo "   You can now open it with: ncu-ui ${report_name}.ncu-rep"
    else
        echo "❌ NCU profiling failed for set '$ncu_set' on range '$nvtx_filter'."
    fi
    echo "================================================================="
    echo
}

# --- 在这里选择你要运行的分析任务 ---
# run_profiling "full" "Kernel_Test" "comparison_report_M_32" 32
./kernel_test.bin --Ns 16 --d 128 --M 32 --T 1000 --r 17 --iters 100 # 性能和Baseline持平
./kernel_test.bin --Ns 16 --d 128 --M 64 --T 1000 --r 17 --iters 100 # 因为Bank冲突的问题，导致了性能比Baseline差
# Bank冲突的问题 是否能够解决

echo "All selected profiling tasks completed."



