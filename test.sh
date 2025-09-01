#!/usr/bin/env bash
#!/usr/bin/env bash
# claude -c 是接着之前的聊 /clean

ddbug_pq() {
    python3 scripts/modeldb/main_pq.py \
    -f llama-2-7b.json \
    -p evaluation \
    -d $1 \
    --merged_training \
    --half -M 32 --nbits 8
}

run_pq() {
    python3 scripts/modeldb/main_pq.py \
    -f llama-2-7b.json \
    -p sampling training evaluation \
    -d wikitext-2-raw-v1 \
    --merged_training \
    --half -M 32 --nbits 8
}

debug_pq_paged() {
    python3 scripts/modeldb/main_pq.py \
    -f llama-2-7b.json \
    -p evaluation \
    -d $1 \
    --merged_training \
    --half -M 32 --nbits 8 --paged --page_size 64 --extended_residual 128 --max_pages 100
}

# run_pq
# ddbug_pq _synthetic # 只能测PPL，不能测速度
debug_pq_paged _synthetic 


















