#!/bin/bash
# 测试 PagedPQCache 的 prefill 方法

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🧪 测试 PagedPQCache 的 prefill 方法${NC}"
echo "=============================================="

# 设置环境
export LD_LIBRARY_PATH="/root/miniconda3/envs/million/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
PYTHON_PATH="/root/miniconda3/envs/million/bin/python"

echo -e "${YELLOW}🔧 测试配置:${NC}"
echo "  模型: llama-2-7b.json"
echo "  数据集: _synthetic"
echo "  参数: -M 32 --nbits 8 --paged --page_size 64 --extended_residual 128 --max_pages 3000"
echo "  目标: 验证 prefill 方法是否正确触发页面分配"
echo "  修复: 增加max_pages到3000，避免页面池耗尽"
echo

# 检查环境
if [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}❌ Python环境不存在: $PYTHON_PATH${NC}"
    exit 1
fi

echo -e "${CYAN}🚀 开始测试...${NC}"

# 运行测试
$PYTHON_PATH -m scripts.modeldb.main_pq \
    -f llama-2-7b.json \
    -p evaluation \
    -d _synthetic \
    --merged_training \
    --half \
    -M 32 \
    --nbits 8 \
    --paged \
    --page_size 64 \
    --extended_residual 128 \
    --max_pages 3000

echo
echo -e "${GREEN}🎉 测试完成！${NC}"
echo -e "${CYAN}💡 检查输出中的页面分配情况：${NC}"
echo "• 应该看到 'Total pages allocated: > 0'"
echo "• 应该看到 'Layer X: Y tokens, Z pages'"
echo "• 应该看到 prefill 相关的调试信息"

