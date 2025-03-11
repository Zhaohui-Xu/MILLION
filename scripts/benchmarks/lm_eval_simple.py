from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from ..utils.Timer import tprint
from ..utils.Injector import PreSuffixInjector
from ..utils.ContextList import ContextList

def lm_eval_simple(model, tokenizer, *, dataset, verbose=True, cache_clear_func=None, **kwargs):
    hflm = HFLM(model)

    eval_context = ContextList([])
    if cache_clear_func is not None:
        # for prefill-only benchmarks, hflm._model_call is called per request
        # for prefifll-and-decode benchmarks, hflm._model_generate is called per request
        # the two methods should be prefixed with cache_clear_func to ensure a clean start
        eval_context.append(
            PreSuffixInjector(
                hflm,
                '_model_call',
                prefix=cache_clear_func,
                suffix=None
            )
        )
        eval_context.append(
            PreSuffixInjector(
                hflm,
                '_model_generate',
                prefix=cache_clear_func,
                suffix=None
            )
        )
    
    tprint("Starting evaluation")
    res = simple_evaluate(
        model=hflm,
        tasks=[dataset],
    )['results'][dataset]

    tprint(res)

    return res


