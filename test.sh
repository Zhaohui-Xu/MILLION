#!/usr/bin/env bash
#!/usr/bin/env bash

ddbug_pq() {
    python3 -m ipdb scripts/modeldb/main_pq.py \
    -f llama-2-7b.json \
    -p sampling training evaluation \
    -d wikitext-2-raw-v1 \
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

run_pq


















