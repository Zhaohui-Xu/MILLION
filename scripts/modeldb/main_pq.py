import argparse
import json
import itertools
from tqdm import tqdm
import pathlib
import os
import importlib
import sys
from pathlib import Path
import time

# 使得可以导入 scripts.utils.*（脚本直接运行时）
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
try:
    from ..utils.Namespace import UniConfig, load_config
    from ..utils.Timer import tprint, Timer
except ImportError:
    from scripts.utils.Namespace import UniConfig, load_config
    from scripts.utils.Timer import tprint, Timer

import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
    Timer('').start()
    # ================== Argument Parsing ==================
    parser = argparse.ArgumentParser(description="ModelDB")
    parser.add_argument("-f", "--file", type=str, help="Relative path to config.json. Relative to scripts/modeldb/configs/", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", required=False)
    parser.add_argument("-M", type=int, help="PQ config, number of sub-sections", required=False)
    parser.add_argument("--nbits", type=int, help="PQ config, number of bits per sub-section", required=False)
    parser.add_argument("-m", "--merged_training", action="store_true", help="Train a merged PQ", required=False)
    parser.add_argument("--opq", action="store_true", help="Prepand LT before PQ to maximize squred error across dimensions", required=False)
    parser.add_argument("--seed", type=int, help="Random seed", required=False, default=42)
    parser.add_argument("--half", action="store_true", help="Use half precision", required=False)
    parser.add_argument("--breakdown", action="store_true", help="Breakdown timing, could lead to overhead due to additional torch.cuda.synchronize", required=False)
    
    # Phase 3: 页式缓存相关选项
    parser.add_argument("--paged", action="store_true", help="Enable paged attention with transposed V storage optimization", required=False)
    parser.add_argument("--page_size", type=int, default=64, help="Page size in tokens for paged attention (default: 64)", required=False)
    parser.add_argument("--extended_residual", type=int, default=128, help="Extended residual cache size in tokens (default: 128)", required=False)
    parser.add_argument("--max_pages", type=int, default=1000, help="Maximum number of pages in page pool (default: 1000)", required=False)
    
    parser.add_argument(
        "-p", "--pipeline",
        nargs='+',
        choices=["baseline", "sampling", "training", "evaluation"],
        help="List of pipeline stages to execute"
    )

    args = parser.parse_args()

    if args.opq:
        raise NotImplementedError("OPQ is not implemented for GPU yet.")
    if args.merged_training is False:
        raise NotImplementedError("Only merged training is supported😈. Use --merged_training")
    
    # ================== Config ==================
    config = UniConfig()
    config.device = 'cuda' # TODO: support multi-gpu

    config.root = pathlib.Path(__file__).parent.parent.parent
    config.config_root = config.root / "scripts" / "modeldb" / "configs"
    config.config_path = config.config_root / args.file

    # Load config
    config += load_config(config.config_root / "default.json")
    config += load_config(config.config_path)

    if args.M is not None:
        config.M = args.M
    if args.nbits is not None:
        config.nbits = args.nbits
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.pipeline is not None:
        config.pipeline = args.pipeline
    if args.seed is not None:
        config.seed = args.seed
    if args.half is not None:
        config.half = args.half
    if args.breakdown is not None:
        config.breakdown = args.breakdown
    
    # 页式缓存配置
    if args.paged is not None:
        config.paged = args.paged
    if args.page_size is not None:
        config.page_size = args.page_size
    if args.extended_residual is not None:
        config.extended_residual = args.extended_residual
    if args.max_pages is not None:
        config.max_pages = args.max_pages

    config.scalar_t = torch.float16 if config.half else torch.float32

    config.model_root = config.root / "models"
    config.datasets_root = config.root / "datasets" 

    config.model_path = config.model_root / config.folder
    config.sample_root = config.root / "kv_samples" / config.model_name / config.dataset
    config.cent_root = config.root / "centroids" / config.model_name / config.dataset

    config.opq = args.opq
    config.merged_training = args.merged_training

    from transformers import AutoConfig
    try:
        from .models.ModelContext import get_context
    except ImportError:
        from scripts.modeldb.models.ModelContext import get_context

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    config.context = get_context(config.model_config.model_type)

    try:
        from ..utils.pq_utils import nbits2dtype
    except ImportError:
        from scripts.utils.pq_utils import nbits2dtype
    config.cache_dtype = nbits2dtype(config.nbits)

    # ================== Seed ==================
    seed_everything(config.seed)

    # ================== Load Model ==================
    if not (len(config.pipeline) == 1 and "training" in config.pipeline):
        tprint(f"Loading model {config.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with config.context.init_context:
            model = AutoModelForCausalLM.from_pretrained(config.model_path).to(config.device)
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            if config.half:
                model = model.half()

    # ================== baseline ==================
    if "baseline" in config.pipeline:
        tprint("Baseline")
        try:
            from ..benchmarks import dataset2benchmark
        except ImportError:
            from scripts.benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        with config.context.baseline_context:
            score_baseline = benchmark(model, tokenizer, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score_baseline, "model_name": config.model_name, "dataset": config.dataset, "baseline": True, "half": config.half}))
            f.write("\n")

    # ================== sampling ==================
    if "sampling" in config.pipeline and config.dataset != '_synthetic':
        tprint("Sampling")

        key_sampled_path = config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs'
        value_sampled_path = config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs'

        if key_sampled_path.exists() or value_sampled_path.exists():
            # ask for confirmation
            tprint(f"Sampling files already exist at {key_sampled_path} or {value_sampled_path}")
            tprint(f"Overwrite? (y/n)")
            if input().lower() != 'y':
                tprint("Exit")
                exit()
            else:
                os.remove(key_sampled_path)
                os.remove(value_sampled_path)

        try:
            from .Errors import SamplingComplete
        except ImportError:
            from scripts.modeldb.Errors import SamplingComplete

        try:
            from ..benchmarks import dataset2benchmark
        except ImportError:
            from scripts.benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        config.sampled_nums = 0
        config.expected_sample_nums = 256 * 2**config.nbits
        config.expected_vecs_per_sample = config.model_config.num_key_value_heads
        config.threshold = 1 / config.expected_vecs_per_sample / 5

        with config.context.sampling_context:
            try:
                benchmark(model, tokenizer, **(config.to_dict()))
            except SamplingComplete as e:
                tprint(e)

    # ================== training ==================
    if "training" in config.pipeline and config.dataset != '_synthetic':
        tprint("Training")
        try:
            from ..utils.fvecio import read_fvecs
            from ..utils.pq_utils import train_pq
            from ..utils.pq_utils import train_opq
        except ImportError:
            from scripts.utils.fvecio import read_fvecs
            from scripts.utils.pq_utils import train_pq
            from scripts.utils.pq_utils import train_opq
        from torch import save

        os.makedirs(config.cent_root, exist_ok=True)

        key = read_fvecs(config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs')
        if config.opq is False:
            key_cent = train_pq(key, config.M, config.nbits)
            save(key_cent, config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt')
            del key, key_cent
        else:
            key_A, key_cent = train_opq(key, config.M, config.nbits)
            save(key_A, config.cent_root / f'key_A_{config.M}_{config.nbits}.opq.pt')
            save(key_cent, config.cent_root / f'key_cent_{config.M}_{config.nbits}.opq.pt')
            del key, key_A, key_cent

        val = read_fvecs(config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs')
        if config.opq is False:
            val_cent = train_pq(val, config.M, config.nbits)
            save(val_cent, config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt')
            del val, val_cent
        else:
            val_A, val_cent = train_opq(val, config.M, config.nbits)
            save(val_A, config.cent_root / f'val_A_{config.M}_{config.nbits}.opq.pt')
            save(val_cent, config.cent_root / f'val_cent_{config.M}_{config.nbits}.opq.pt')
            del val, val_A, val_cent

    if "evaluation" not in config.pipeline:
        tprint("Exit")
        exit()

    # 主要看这个 
    
    # ================== PQ Config ==================
    if "evaluation" in config.pipeline:
        if config.dataset == '_synthetic':
            tprint("Using synthetic centroids for speed evaluation")
            key_cent = torch.randn(config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
            val_cent = torch.randn(config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
        else:
            key_cent = torch.load(config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True)
            key_cent = key_cent.to(config.device).to(model.dtype) # Load 质心
            val_cent = torch.load(config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True)
            val_cent = val_cent.to(config.device).to(model.dtype) # Load 质心

            if config.opq is True:
                tprint("OPQ patching model weights")
                key_A = torch.load(config.cent_root / f'key_A_{config.M}_{config.nbits}.opq.pt', weights_only=True)
                key_A = key_A.to(config.device).to(model.dtype)
                val_A = torch.load(config.cent_root / f'val_A_{config.M}_{config.nbits}.opq.pt', weights_only=True)
                val_A = val_A.to(config.device).to(model.dtype)

                # # OPQ prepends Linear Transformation before PQ: 
                # # key_codes = opq_encode(K) = pq_encode(K @ A.T) = pq_encode(X @ Wk.T @ A.T) = pq_encode(X @ (A @ Wk).T)

                # # Insight: A @ Wk is a fixed transformation, so we can precompute it as new model weights to avoid online overhead

                # # In practice: k_proj, v_proj are instances of nn.Linear
                # # k_proj = x -> x @ Wk.T where k_proj.weight = Wk, sized (num_key_value_heads * head_dim, hidden_size)
                # # We are assuming bias is not used in k_proj, v_proj. Fortunately this is generally true for most models.
                # # Merged training uses the same opq for all heads, A.shape = (head_dim, head_dim). To apply A to all heads, 
                # # we should fit A to diagonal blocks of (num_key_value_heads * head_dim, num_key_value_heads * head_dim) matrix

                # head_dim = config.model_config.hidden_size // config.model_config.num_key_value_heads
                # head_dim_sum = head_dim * config.model_config.num_key_value_heads
                # key_A_expanded = torch.zeros(head_dim_sum, head_dim_sum, dtype=model.dtype, device=model.device)
                # val_A_expanded = torch.zeros(head_dim_sum, head_dim_sum, dtype=model.dtype, device=model.device)
                # for i in range(config.model_config.num_key_value_heads):
                #     start = i * head_dim
                #     end = start + head_dim
                #     key_A_expanded[start:end, start:end] = key_A
                #     val_A_expanded[start:end, start:end] = val_A

                # # The following code is tuned for llama. For other models, we may need to change the component names.
                # for layer in model.model.layers:
                #     Wk = layer.self_attn.k_proj.weight.data
                #     Wv = layer.self_attn.v_proj.weight.data
                #     layer.self_attn.k_proj.weight = torch.nn.Parameter(key_A_expanded @ Wk)
                #     layer.self_attn.v_proj.weight = torch.nn.Parameter(val_A_expanded @ Wv)
                # del key_A_expanded, val_A_expanded

                del key_A, val_A, 
                torch.cuda.empty_cache()
        # import ipdb; ipdb.set_trace()
        # 智能缓存初始化 - 根据配置选择合适的缓存类型
        if hasattr(config, 'paged') and config.paged:
            try:
                from ..utils.paged_pq_utils import PagedPQCache
            except ImportError:
                from scripts.utils.paged_pq_utils import PagedPQCache
            
            tprint("Initializing PagedPQCache with enhanced performance optimizations")
            # 使用PagedPQCache（转置存储+页面管理优化）
            cache = PagedPQCache(
                bs=1,
                num_key_value_heads=config.model_config.num_key_value_heads,
                nh=config.model_config.num_attention_heads, 
                M=config.M,
                layer_num=config.model_config.num_hidden_layers,
                dtype=config.cache_dtype,
                nbits=config.nbits,
                d=config.d,
                scalar_t=config.scalar_t,
                # 页式缓存专用参数
                page_size=getattr(config, 'page_size', 64),
                extended_residual_size=getattr(config, 'extended_residual', 128)
            )
            tprint(f"PagedPQCache initialized: {cache.page_size} tokens/page, {cache.extended_residual_size} residual tokens")
            
            # 添加性能监控
            if hasattr(config, 'breakdown') and config.breakdown:
                tprint("🔍 Performance monitoring enabled")
        else:
            try:
                from ..utils.pq_utils import DynamicPQCache
            except ImportError:
                from scripts.utils.pq_utils import DynamicPQCache
            
            tprint("Initializing standard DynamicPQCache")
            # init cache
            cache = DynamicPQCache(
                bs = 1, # TODO: support batch size?
                num_key_value_heads=config.model_config.num_key_value_heads,
                nh = config.model_config.num_attention_heads,
                M = config.M, # 子空间数
                layer_num=config.model_config.num_hidden_layers,
                dtype=config.cache_dtype,
                nbits=config.nbits,
                d=config.d, # model_head_dim
                scalar_t=config.scalar_t,
            )
        
        cache.set_cent(key_cent, val_cent)

        tprint("Evaluation")
        
        try:
            from ..benchmarks import dataset2benchmark
        except ImportError:
            from scripts.benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        # 添加性能监控
        evaluation_start_time = time.time()
        
        with config.context.evaluation_context:
            cache_clear_func = cache.init_cache
            score = benchmark(model, tokenizer, cache_clear_func=cache_clear_func, **(config.to_dict()))
        
        evaluation_time = time.time() - evaluation_start_time
        
        # 输出详细的性能统计
        tprint(f"⏱️  Evaluation completed in {evaluation_time:.2f} seconds")
        
        # 如果是PagedPQCache，输出详细的缓存统计
        if hasattr(config, 'paged') and config.paged:
            try:
                cache_stats = cache.get_cache_stats()
                tprint("📊 PagedPQCache Statistics:")
                tprint(f"  Total pages allocated: {cache_stats['total_pages_allocated']}")
                tprint(f"  Total memory usage: {cache_stats['total_memory_usage_mb']:.2f} MB")
                
                # 输出每层的统计信息
                for layer_stat in cache_stats['layer_stats']:
                    tprint(f"  Layer {layer_stat['layer_idx']}: {layer_stat['seen_tokens']} tokens, {layer_stat['value_pages_count']} pages")
            except Exception as e:
                tprint(f"⚠️  Could not retrieve cache statistics: {e}")

        # write to jsonl with performance data
        result_data = {
            "score": score, 
            "evaluation_time_seconds": evaluation_time,
            "cache_type": "PagedPQCache" if (hasattr(config, 'paged') and config.paged) else "DynamicPQCache"
        }
        
        # 添加缓存统计信息
        if hasattr(config, 'paged') and config.paged:
            try:
                cache_stats = cache.get_cache_stats()
                result_data.update({
                    "total_pages_allocated": cache_stats['total_pages_allocated'],
                    "total_memory_usage_mb": cache_stats['total_memory_usage_mb'],
                    "page_size": getattr(config, 'page_size', 64),
                    "extended_residual_size": getattr(config, 'extended_residual', 128)
                })
            except:
                pass
        
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({**result_data, **config.to_serializable_dict()}))
            f.write("\n")

    tprint("Exit")
