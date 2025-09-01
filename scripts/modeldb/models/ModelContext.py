from ...utils.ContextList import ContextList
from dataclasses import dataclass
import importlib

@dataclass(frozen=True)
class ModelContext:
    init_context: ContextList # 模型初始化时的上下文 
    baseline_context: ContextList # 基线评测时的上下文  
    sampling_context: ContextList # 采样阶段的上下文
    evaluation_context: ContextList # 评测阶段的上下文

registery = {}

def get_context(name) -> ModelContext:
    if name in registery:
        return registery[name]
    
    try:
        module_name = f".modeling_{name}"
        module = importlib.import_module(module_name, package=__package__)
        model_context = getattr(module, 'model_context')
        registery[name] = model_context

        return model_context
    
    except ModuleNotFoundError:
        raise ImportError(f"modeling_{name}.py not found.")
    
    except AttributeError:
        raise ImportError(f"'model_context' not defined in modeling_{name}.py")