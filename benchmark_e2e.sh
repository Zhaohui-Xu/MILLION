#!/usr/bin/env bash
# MILLION 端到端性能基准测试脚本
# 基于 test.sh 的实现方式，用于测试 PagedPQCache vs Baseline vs Standard PQ 的性能对比

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置参数
PYTHON_CMD="/root/miniconda3/envs/million/bin/python"
export LD_LIBRARY_PATH="/root/miniconda3/envs/million/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# 默认参数
MODEL="llama-2-7b"
DATASET="_synthetic"
M=32  # 改为32以匹配可用内核
NBITS=8

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --m)
            M="$2"
            shift 2
            ;;
        --nbits)
            NBITS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Model name (default: llama-2-7b)"
            echo "  --m M                Number of PQ sub-spaces (default: 64)"
            echo "  --nbits NBITS        Quantization bits (default: 8)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# 测试函数定义（参考 test.sh）
run_baseline() {
    echo -e "${CYAN}运行 Baseline (无量化) 测试...${NC}"
    $PYTHON_CMD -m scripts.modeldb.main_pq \
        -f "${MODEL}.json" \
        -p baseline \
        -d "$DATASET" \
        --merged_training \
        --half
}

run_standard_pq() {
    echo -e "${CYAN}运行 Standard PQ 测试...${NC}"
    $PYTHON_CMD -m scripts.modeldb.main_pq \
        -f "${MODEL}.json" \
        -p evaluation \
        -d "$DATASET" \
        --merged_training \
        --half \
        -M "$M" \
        --nbits "$NBITS"
}

run_paged_pq() {
    echo -e "${CYAN}运行 PagedPQCache 测试...${NC}"
    $PYTHON_CMD -m scripts.modeldb.main_pq \
        -f "${MODEL}.json" \
        -p evaluation \
        -d "$DATASET" \
        --merged_training \
        --half \
        -M "$M" \
        --nbits "$NBITS" \
        --paged \
        --page_size 128 \
        --extended_residual 128 \
        --max_pages 1000
}

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         MILLION 端到端性能基准测试                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}测试配置:${NC}"
echo -e "  模型: ${GREEN}${MODEL}${NC}"
echo -e "  数据集: ${GREEN}${DATASET}${NC}"
echo -e "  PQ子空间数(M): ${GREEN}${M}${NC}"
echo -e "  量化位数: ${GREEN}${NBITS}${NC}"
echo ""

# 创建结果目录
RESULTS_DIR="benchmark_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${YELLOW}📊 开始性能测试...${NC}"
echo ""

# 执行测试
echo -e "${CYAN}[1/3] Baseline (无量化) 测试${NC}"
run_baseline 2>&1 | tee "$RESULTS_DIR/baseline.log"
echo -e "${GREEN}✅ Baseline 测试完成${NC}"
echo ""

echo -e "${CYAN}[2/3] Standard PQ 测试${NC}"
run_standard_pq 2>&1 | tee "$RESULTS_DIR/standard_pq.log"
echo -e "${GREEN}✅ Standard PQ 测试完成${NC}"
echo ""

echo -e "${CYAN}[3/3] PagedPQCache 测试${NC}"
run_paged_pq 2>&1 | tee "$RESULTS_DIR/paged_pq.log"
echo -e "${GREEN}✅ PagedPQCache 测试完成${NC}"
echo ""

# 生成简化的测试报告
REPORT_FILE="$RESULTS_DIR/performance_report.txt"
echo -e "${YELLOW}📈 生成性能报告...${NC}"

cat > "$REPORT_FILE" << EOF
════════════════════════════════════════════════════════════════════
                    MILLION 性能基准测试报告
════════════════════════════════════════════════════════════════════

测试时间: $(date '+%Y-%m-%d %H:%M:%S')

【测试配置】
• 模型: $MODEL
• 数据集: $DATASET
• PQ子空间数: $M
• 量化位数: $NBITS

【测试内容】
1. Baseline (无量化): 直接运行原生模型
2. Standard PQ: 使用标准产品量化
3. PagedPQCache: 使用页式缓存优化的产品量化

【日志文件】
• Baseline: $RESULTS_DIR/baseline.log
• Standard PQ: $RESULTS_DIR/standard_pq.log
• PagedPQCache: $RESULTS_DIR/paged_pq.log

【使用说明】
请查看各日志文件中的性能统计信息：
- 执行时间 (Evaluation completed in X.XX seconds)
- 缓存统计 (PagedPQCache Statistics)
- 内存使用情况

════════════════════════════════════════════════════════════════════
EOF

# 显示报告
cat "$REPORT_FILE"

echo ""
echo -e "${GREEN}🎉 性能测试完成！${NC}"
echo -e "${CYAN}报告已保存至: ${RESULTS_DIR}/performance_report.txt${NC}"