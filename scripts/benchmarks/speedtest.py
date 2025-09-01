from transformers import TextStreamer
from ..utils.Timer import Ticker, tprint, Timer
from ..utils.Injector import PreSuffixInjector
from ..utils.ContextList import ContextList

from itertools import product

from ..utils.escape_codes import BLUE, END, escape_code_factory as ecf

# 快速验证模式：只测试1024长度，重复1次
prefill_lengths = [256]  # 减少 prefill 长度，便于测试页面分配
# prefill_lengths = [4096]  # 只测试1024
# prefill_lengths = [1024, 2048, 4096, 8192, 16384]  # 原始完整测试
decoding_lengths = [128]  # 增加 decoding 长度，确保触发页面分配

def IgnoreOOM(clear_func=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    tprint("OOM")
                    if clear_func is not None:
                        clear_func()
                else:
                    raise e
        return wrapper
    return decorator

def synthetic_input_ids(tokenizer, prefill_length):
    from torch import randint
    return randint(0, len(tokenizer), (1, prefill_length))

def speedtest(model, tokenizer, *, verbose=True, cache_clear_func=None, niter=1, breakdown=False, **kwargs):  # 改为1次重复
    ticker = Ticker('_speedtest')
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # inject ticker.tick before streamer.put to record the timestamp of each token generation
    put_injector = PreSuffixInjector(
        streamer,
        'put',
        prefix=ticker.tick,
        suffix=None
    )
    # colorize streamer output
    print_injector = PreSuffixInjector(
        streamer,
        'on_finalized_text',
        prefix=ecf(BLUE),
        suffix=ecf(END)
    )

    streaming_context = ContextList([put_injector, print_injector])


    @IgnoreOOM(clear_func=cache_clear_func)
    def synthetic_generation_task(
        prefill_length,
        decoding_length,
    ):
        input_ids = synthetic_input_ids(tokenizer, prefill_length).to(model.device)

        ticker.clear()
        if cache_clear_func is not None:
            cache_clear_func()
        model.generate(
            input_ids=input_ids,
            streamer=streamer,
            max_length=prefill_length + decoding_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,  # Disable early stopping
            do_sample=False,            # 新增
            # temperature=0.0,
            repetition_penalty=1.0,
            num_return_sequences=1,
        )
        streamer.end()

        if len(ticker.intervals) <= 5:
            tprint("Not enough intervals. May lead to incorrect timing results.")

    result = []
    with streaming_context:
        for pl, dl in product(prefill_lengths, decoding_lengths):
            tpot = [] # time per output token
            ttft = [] # time to first token
            res = {
                'prefill_length': pl,
                'decoding_length': dl,
            }
            synthetic_generation_task(pl, dl) # warmup
            for _ in range(niter):  # 只重复1次
                synthetic_generation_task(pl, dl)

                intervals = ticker.intervals
                res = {
                    'prefill_length': pl,
                    'decoding_length': dl,
                }

                if verbose is True:
                    print(f"prefill_length: {pl}, decoding_length: {dl}, intervals: {intervals}")
                tpot.append(sum(intervals[1:]) / (dl - 1)) # skip the first 1 intervals as it's prefilling
                ttft.append(intervals[0])
            
            res['time_per_output_token'] = sum(tpot) / niter / 1e6 # in milliseconds
            res['time_to_first_token'] = sum(ttft) / niter / 1e6 # in milliseconds
            
            if breakdown is True:
                breakdown_times = {}
                for k, t in Timer._instances.items():
                    if k.startswith('LlamaSdpaAttention.forward'):
                        print(t)
                        breakdown_times[k] = t.duration
                        t.reset()
                res['breakdown_times'] = breakdown_times
            result.append(res)

    if verbose:
        print(result)

    return result
                
            


            
