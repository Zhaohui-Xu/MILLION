# MILLION

This repository presents the source code for the paper "MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization" (DAC'25).

## News

- [2025-02] Our paper is accepted by DAC'25! 🎆

## Introduction

Large language models (LLMs) are increasingly utilized for complex tasks requiring longer context lengths, with some models supporting up to 128K or 1M tokens. This trend, however, presents significant challenges in inference speed and memory management.
The primary bottleneck in long-context LLM inference is the quadratic computational complexity of attention mechanisms, causing substantial slowdowns as sequence length increases. KV cache mechanism alleviates this issue by storing pre-computed data, but introduces memory requirements that scale linearly with context length, hindering efficient LLM deployment. Quantization emerges as a promising approach to address the widening gap between LLM size and memory capacity. However, traditional quantization schemes often yield suboptimal compression results for KV caches due to two key factors:
i) On-the-fly quantization and de-quantization, causing significant performance overhead;
ii) Prevalence of outliers in KV values, challenging low-bitwidth uniform quantization.
To this end, we propose **MILLION**, a novel quantization framework achieving low-bitwidth KV cache through product quantization. First, we conduct a thorough analysis of KV cache distribution, revealing the limitations of existing quantization schemes. Second, we introduce a non-uniform quantization algorithm based on product quantization, which efficiently compresses data while preserving accuracy. Third, we develop a high-performance GPU inference framework with efficient attention kernel and pipeline design for **MILLION** that leverages sparse computation and asynchronous quantization, significantly enhancing inference speed. 
Comprehensive evaluation results demonstrate that **MILLION** can achieve 4 bits quantization with trivial perplexity and accuracy loss, and achieve 2.09x end-to-end performance gains at 32K context length.


## Setup

1. Create a new conda environment and install the required packages:
```bash
conda create -n million python=3.12
conda activate million
pip install -r requirements.txt
```

2. Create directories for or links to static data:
```bash
mkdir models # model weights
mkdir kv_samples # sampled kv vectors, used for training PQ
mkdir centroids # trained PQ codebook
mkdir datasets
```
```bash
ln -s /path/to/models models
ln -s /path/to/kv_samples kv_samples
ln -s /path/to/centroids centroids
ln -s /path/to/datasets datasets
```

Currently, MILLION supports all models from LLaMA family, including those using GQA(e.g. LLaMA-3.1), in **huggingface format**. You can refer to [the conversion script from transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) to convert the weights released by META team to huggingface format. For example, the `models/Meta-Llama-3.1-8B-hf` directory should look similar to:
```text
Meta-Llama-3.1-8B-hf/
┣ config.json
┣ generation_config.json
┣ model-00001-of-00004.safetensors
┣ model-00002-of-00004.safetensors
┣ model-00003-of-00004.safetensors
┣ model-00004-of-00004.safetensors
┣ model.safetensors.index.json
┣ special_tokens_map.json
┣ tokenizer.json
┗ tokenizer_config.json
```

3. Compile the CUDA extension:
```bash
make bindings
```

## Quick start

**Perplexity**:
```bash
make ppl
```
Supported datasets are `wikitext-2-raw-v1`, `wikitext-103-v1`, `ptb_text_only` and `wikitext-103-raw-v1`.

**Longbench**:
```bash
make longbench
```
Check `scripts/benchmarks/longbench.py` for more details, including the supported datasets and the corresponding metrics.

**End to End performance**:
```bash
make e2e
```

**Performance breakdowns**:
```bash
make breakdown
```

Results will be saved to `scripts/modeldb/results.jsonl` for possible further analysis.

## Citation

This repository is maintained by Zongwu Wang and Peng Xu from the IMPACT Lab at SJTU, under the supervision of Professor Li Jiang and Fangxin Liu. If you find MILLION useful or relevant to your research, please kindly cite our paper:

> Zongwu Wang=, Peng Xu=, Fangxin Liu*, Yiwei Hu, Qingxiao Sun, Gezi Li, Cheng Li, Xuan Wang, and Li Jiang*, MILLION: MasterIng Long-Context LLM InferenceVia Outlier-Immunized KV Product OuaNtization. In Proceedings of the 61st ACM/IEEE Design Automation Conference 2024.