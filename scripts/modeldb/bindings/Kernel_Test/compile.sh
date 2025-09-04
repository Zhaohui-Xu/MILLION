#!/bin/bash

# --- 配置 ---
# 自动查找 libtorch 路径
# 确保你的 python 环境和 pytorch 已被激活
PYTHON_EXE=$(which python)

echo "Finding PyTorch paths..."
# 修改下面这两行
LIBTORCH_PATH=$($PYTHON_EXE -c 'import torch; from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])')
LIBTORCH_INCLUDE_PATH=$($PYTHON_EXE -c 'import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])')
PYTHON_INCLUDE_PATH=$($PYTHON_EXE -c 'from sysconfig import get_paths; print(get_paths()["include"])')

if [ -z "$LIBTORCH_PATH" ] || [ -z "$LIBTORCH_INCLUDE_PATH" ]; then
    echo "Error: Could not find PyTorch library or include paths."
    echo "Please ensure PyTorch is installed and accessible in your current environment."
    exit 1
fi

echo "LibTorch Path: $LIBTORCH_PATH"
echo "LibTorch Include Path: $LIBTORCH_INCLUDE_PATH"
echo "Python Include Path: $PYTHON_INCLUDE_PATH" # 打印出来以供检查
# NVTX 路径 (如果已安装)
NVTX_INCLUDE_PATH="/root/xzh_codebase/NVTX/c/include" # 默认路径，根据你的系统修改

# --- 定义文件名 ---
OUTPUT_BIN="kernel_test.bin"
OUTPUT_CUBIN="kernel_test.cubin"
OBJECT_FILE="main.o"

# --- 编译步骤 ---
# -lineinfo \
# 步骤 1: 编译 .cu 文件为目标文件 (.o)
echo "Compiling main.cu to an object file..."
nvcc main.cu \
    -c -o ${OBJECT_FILE} \
    -I./ \
    -I./core \
    -I${LIBTORCH_INCLUDE_PATH} \
    -I${LIBTORCH_INCLUDE_PATH}/torch/csrc/api/include \
    -I${NVTX_INCLUDE_PATH} \
    -I${PYTHON_INCLUDE_PATH} \
    -arch=sm_80 \
    -O3 \
    --use_fast_math \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -lineinfo

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "❌ Compilation to object file failed."
    exit 1
fi
echo "✅ Object file created: ${OBJECT_FILE}"

# (可选) 步骤 1.5: 单独生成 .cubin 文件用于检查
echo "Generating CUBIN file for inspection..."
nvcc main.cu \
    -cubin -o ${OUTPUT_CUBIN} \
    -I./ \
    -I./core \
    -I${LIBTORCH_INCLUDE_PATH} \
    -I${LIBTORCH_INCLUDE_PATH}/torch/csrc/api/include \
    -I${NVTX_INCLUDE_PATH} \
    -I${PYTHON_INCLUDE_PATH} \
    -arch=sm_80 \
    -O3 \
    --use_fast_math \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -D_GLIBCXX_USE_CXX11_ABI=0
echo "✅ CUBIN file created: ${OUTPUT_CUBIN}"


# 步骤 2: 链接目标文件和库，生成最终的可执行文件
echo "Linking object file to create executable..."
nvcc ${OBJECT_FILE} \
    -o ${OUTPUT_BIN} \
    -L${LIBTORCH_PATH} \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

# 检查链接是否成功
if [ $? -eq 0 ]; then
    echo "✅ Linking successful. Executable created: ${OUTPUT_BIN}"
    echo "To run, use for example: ./${OUTPUT_BIN} --Ns 16 --d 128"
else
    echo "❌ Linking failed."
fi