from ..utils.LazyLoader import LazyLoader

def select_benchmark(benchmark_name):
    if benchmark_name == 'perplexity':
        from .perplexity import perplexity
        return perplexity
    elif benchmark_name == 'speedtest':
        from .speedtest import speedtest
        return speedtest
    elif benchmark_name == 'lm_eval_simple':
        from .lm_eval_simple import lm_eval_simple
        return lm_eval_simple
    elif benchmark_name == 'longbench':
        from .longbench import pred_long_bench
        return pred_long_bench
    else:
        raise ValueError(f'Unknown benchmark name {benchmark_name}')

benchmark2dataset = {
    'perplexity': [
        'wikitext-2-raw-v1',
        'wikitext-103-v1',
        'ptb-text-only',
        'wikitext-103-raw-v1',
    ],
    'speedtest': [
        '_synthetic',
    ],
    'lm_eval_simple': [
        'mmlu',
        'arc_challenge',
        'arc_easy',
        'logiqa',
        'piqa',
        'sciq',
        'winogrande',
        'wsc',
    ],
    'longbench': [
        'qasper',   # Single-Document QA, avg len 3619
        '2wikimqa', # Multi-Document QA, avg len 4887
        'multi_news', # Summarization, avg len 2113
        'trec', # Few-shot Learning, avg len 5177
        'passage_retrieval_en', # Synthetic, avg len 9289
        'lcc', # Code Completion, avg len 1235

        'multifieldqa_en', # Single-Document QA, avg len 4559 
        'hotpotqa', # Multi-Document QA, avg len 9151
        'gov_report', # Summarization, avg len 8734
        'samsum', # Few-shot Learning, avg len 6278
        'passage_count', # Synthetic, avg len 11141
        'repobench-p', # Code Completion, avg len 4206   

        'narrativeqa', # Single-Document QA, avg len 18409
        'musique', # Multi-Document QA, avg len 11214
        'qmsum', # Summarization, avg len 10614
        'triviaqa', # Few-shot Learning, avg len 8209
    ],
}

dataset2benchmark = {
    dataset: LazyLoader(select_benchmark, benchmark)
    for benchmark, datasets in benchmark2dataset.items()
    for dataset in datasets
}