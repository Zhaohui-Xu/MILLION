from ...utils.Injector import Injector
from ...utils.Namespace import UniConfig
from ...utils.ContextList import ContextList
from ...utils.Timer import Timer
from ...utils.pq_utils import DynamicPQCache
from ...utils.paged_pq_utils import PagedPQCache
from transformers.models.llama.modeling_llama import (
    LlamaSdpaAttention,
    LlamaForCausalLM,
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    Cache
)

import torch
from typing import Optional, Tuple
from .ModelContext import ModelContext
from ..Errors import SamplingComplete

def save_forward(
    self: LlamaSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 在注意力计算过程中采样 KV 向量
    # 保存到 .fvecs 文件用于训练 PQ 码本
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # save to disk
    bs, nh, _, head_size = key_states.shape
    config = UniConfig()
    root = config.sample_root
    threshold = config.threshold

    import os
    import random 
    os.makedirs(root, exist_ok=True)
    from ...utils.fvecio import write_fvecs
    for b in range(bs):
        for h in range(nh):
            if random.random() > threshold:
                continue
            key_ = key_states[b, h].view(-1, head_size).cpu().detach().numpy()
            write_fvecs(config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs', key_, 'ab')
            del key_
            value_ = value_states[b, h].view(-1, head_size).cpu().detach().numpy()
            write_fvecs(config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs', value_, 'ab')
            del value_
            if self.layer_idx == 0: 
                config.sampled_nums += 1
            if config.sampled_nums > config.expected_sample_nums:
                raise SamplingComplete(f"Sampled {config.sampled_nums} vectors, expected {config.expected_sample_nums}, stopping...")

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def prepare_inputs_for_generation(
    self: LlamaForCausalLM,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=True,
    **kwargs,
):
    
    # 参考原始实现：手动构造 model_inputs（不要调用 super()）
    past_length = 0
    
    # 移除强制设置，使用正确的逻辑
    # 支持多种缓存类型
    if PagedPQCache.has_instance():
        # 若需要，可从实例获取已见长度；这里按0处理，保持行为与原始类似
        try:
            # 由于PagedPQCache使用Singleton，我们需要通过类字典获取实例
            cache = PagedPQCache._instances.get(PagedPQCache, None)
            if cache is not None:
                past_length = cache.seen_tokens[0] if hasattr(cache, 'seen_tokens') else 0
            else:
                past_length = 0
        except Exception as e:
            past_length = 0
        cache_length = past_length
        max_cache_length = None
    elif DynamicPQCache.has_instance():
        try:
            cache = DynamicPQCache()
            past_length = cache.seen_tokens[0]
        except Exception:
            past_length = 0
        cache_length = past_length
        max_cache_length = None
    
    # Keep only the unprocessed tokens
    if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
    elif past_length < input_ids.shape[1]:
        input_ids = input_ids[:, past_length:]
    elif past_length >= input_ids.shape[1]:
        # 关键修复：当past_length >= input_ids.shape[1]时，说明这是decode阶段
        # 此时input_ids应该只包含新生成的token，通常是1个token
        if past_length > 0:  # 确保不是第一次调用
            # 如果input_ids长度大于1，说明有问题，强制截断为1个token
            if input_ids.shape[1] > 1:
                input_ids = input_ids[:, -1:]  # 只保留最后一个token
    
    if (
        'max_cache_length' in locals()
        and max_cache_length is not None
        and attention_mask is not None
        and cache_length + input_ids.shape[1] > max_cache_length
    ):
        attention_mask = attention_mask[:, -max_cache_length:]
    
    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if PagedPQCache.has_instance() or DynamicPQCache.has_instance():
            position_ids = position_ids[:, -input_ids.shape[1] :]
    
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # 保持与原实现一致的 contiguous 处理
        model_inputs = {"input_ids": input_ids.contiguous()}
    
    input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    if cache_position is None:
        cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
    elif use_cache:
        cache_position = cache_position[-input_length:]
    
    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # attn_forward 是 PyTorch 版本的 PQ 实现，当 nbits != 8 时使用。
    # 没有做计算融合优化 就是单纯的KV正常量化+反量化 的计算 
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    distort_recent = UniConfig().distort_recent
    with Timer("LlamaSdpaAttention.forward"):
        with Timer("qkv_proj"):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            torch.cuda.synchronize()

        with Timer("rotary_emb"):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            torch.cuda.synchronize()

        with Timer("update_cache"): # 模块计时工具
            if use_cache:
                assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
                # 获取已存在的实例，需要提供必要的参数
                config = UniConfig()
                if not hasattr(config, 'd') or config.d is None:
                    config.d = config.model_config.hidden_size // config.model_config.num_key_value_heads
                if not hasattr(config, 'M') or config.M is None:
                    config.M = 16
                if not hasattr(config, 'nbits') or config.nbits is None:
                    config.nbits = 8
                if not hasattr(config, 'cache_dtype') or config.cache_dtype is None:
                    config.cache_dtype = torch.uint8
                if not hasattr(config, 'scalar_t') or config.scalar_t is None:
                    config.scalar_t = torch.float16 if getattr(config, 'half', False) else torch.float32
                
                cache = DynamicPQCache(
                    bs=1,
                    nh=config.model_config.num_attention_heads,
                    num_key_value_heads=config.model_config.num_key_value_heads,
                    M=config.M,
                    layer_num=config.model_config.num_hidden_layers,
                    dtype=config.cache_dtype,
                    nbits=config.nbits,
                    d=config.d,
                    scalar_t=config.scalar_t
                )
                # 传入的是FP16的KV
                # 这里调用 cache.update() 返回的是反量化后的 KV
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx, distort_recent=distort_recent)
            torch.cuda.synchronize()
        
        with Timer("repeat_kv"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            torch.cuda.synchronize()

        with Timer("causal_mask"):
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            torch.cuda.synchronize()

        with Timer("contiguous"):
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            torch.cuda.synchronize()

        with Timer("sdpa"):
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
            # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
            is_causal = True if causal_mask is None and q_len > 1 else False
            
            # 使用 PyTorch 的 SDPA 进行注意力计算
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states, # 反量化后的类型
                value_states, # 反量化后的类型
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            torch.cuda.synchronize()

        with Timer("contiguous"):
            attn_output = attn_output.transpose(1, 2).contiguous()
            torch.cuda.synchronize()

        with Timer("o_proj"):
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            torch.cuda.synchronize()

        return attn_output, None, past_key_value

def baseline_forward(
    self: LlamaSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 原生 LLaMA 注意力实现
    # 用于计算基线性能，不使用 PQ Cache
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()
    Timer.disabled = True if q_len != 1 else False
    breakdown = UniConfig().breakdown
    with Timer("LlamaSdpaAttention.forward"):
        with Timer("LlamaSdpaAttention.forward.qkv_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.rotary_emb"):
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.cat"):    
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.repeat_kv"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.causal_mask"):
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            if breakdown is True: torch.cuda.synchronize()
        
        with Timer("LlamaSdpaAttention.forward.contiguous"):
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.sdpa"):
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.o_proj"):
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            if breakdown is True: torch.cuda.synchronize()

            attn_output = self.o_proj(attn_output)
            if breakdown is True: torch.cuda.synchronize()

    return attn_output, None, past_key_value

def attn_forward_custom_kernel(      
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # attn_forward_custom_kernel - 自定义核注意力实现
    # 当 nbits == 8 时使用。
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
    
    bsz, q_len, _ = hidden_states.size()
    Timer.disabled = True if q_len != 1 else False
    breakdown = UniConfig().breakdown
    distort_recent = UniConfig().distort_recent

    with Timer("LlamaSdpaAttention.forward"):
        with Timer("LlamaSdpaAttention.forward.qkv_proj"):

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.rotary_emb"):
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if breakdown is True: torch.cuda.synchronize()

        assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
        # 获取已存在的实例，需要提供必要的参数
        config = UniConfig()
        if not hasattr(config, 'd') or config.d is None:
            config.d = config.model_config.hidden_size // config.model_config.num_key_value_heads
        if not hasattr(config, 'M') or config.M is None:
            config.M = 16
        if not hasattr(config, 'nbits') or config.nbits is None:
            config.nbits = 8
        if not hasattr(config, 'cache_dtype') or config.cache_dtype is None:
            config.cache_dtype = torch.uint8
        if not hasattr(config, 'scalar_t') or config.scalar_t is None:
            config.scalar_t = torch.float16 if getattr(config, 'half', False) else torch.float32
        
        cache = DynamicPQCache(
            bs=1,
            nh=config.model_config.num_attention_heads,
            num_key_value_heads=config.model_config.num_key_value_heads,
            M=config.M,
            layer_num=config.model_config.num_hidden_layers,
            dtype=config.cache_dtype,
            nbits=config.nbits,
            d=config.d,
            scalar_t=config.scalar_t
        )
        if q_len > 1:
            attn_output = cache.prefill(
                query_states, key_states, value_states, self.layer_idx, distort_recent=distort_recent
            )
        else:
            attn_output = cache.decoding(
                query_states, key_states, value_states, self.layer_idx
            )

        with Timer("LlamaSdpaAttention.forward.o_proj"):
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            if breakdown is True: torch.cuda.synchronize()

    return attn_output, None, past_key_value

def attn_forward_paged_kernel(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """页式attention前向传播 - 使用PagedPQCache和转置存储优化"""
    if output_attentions:
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
    
    bsz, q_len, _ = hidden_states.size()
    Timer.disabled = True if q_len != 1 else False
    breakdown = UniConfig().breakdown
    distort_recent = UniConfig().distort_recent

    with Timer("LlamaSdpaAttention.forward.paged"):
        with Timer("LlamaSdpaAttention.forward.qkv_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.rotary_emb"):
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if breakdown is True: torch.cuda.synchronize()
        # 使用PagedPQCache进行attention计算
        assert PagedPQCache.has_instance(), 'PagedPQCache must be set before using it.'
        # 获取已存在的实例，需要提供必要的参数
        config = UniConfig()
        if not hasattr(config, 'd') or config.d is None:
            config.d = config.model_config.hidden_size // config.model_config.num_key_value_heads
        if not hasattr(config, 'M') or config.M is None:
            config.M = 16
        if not hasattr(config, 'nbits') or config.nbits is None:
            config.nbits = 8
        if not hasattr(config, 'cache_dtype') or config.cache_dtype is None:
            config.cache_dtype = torch.uint8
        if not hasattr(config, 'scalar_t') or config.scalar_t is None:
            config.scalar_t = torch.float16 if getattr(config, 'half', False) else torch.float32
        
        cache = PagedPQCache(
            bs=1,
            nh=config.model_config.num_attention_heads,
            num_key_value_heads=config.model_config.num_key_value_heads,
            M=config.M,
            layer_num=config.model_config.num_hidden_layers,
            dtype=config.cache_dtype,
            nbits=config.nbits,
            d=config.d,
            scalar_t=config.scalar_t,
            page_size=getattr(config, 'page_size', 64),
            extended_residual_size=getattr(config, 'extended_residual', 128)
        )
        
        with Timer("PagedPQCache.attention"):
            
            if q_len > 1:
                # Prefill阶段：批量处理输入tokens
                attn_output = cache.prefill(
                    query_states, key_states, value_states, self.layer_idx, distort_recent=distort_recent
                )
               
            else:
                # import ipdb; ipdb.set_trace()
                # Decoding阶段：使用页式优化attention
                attn_output = cache.decoding_with_pages(
                    query_states, key_states, value_states, self.layer_idx
                )
              
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.o_proj"):
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            if breakdown is True: torch.cuda.synchronize()

    return attn_output, None, past_key_value

def _create_evaluation_context():
    """创建评估上下文 - 智能选择合适的attention实现"""
    config = UniConfig()
    
    try:
        # 页式缓存优先级最高  
        if hasattr(config, 'paged') and config.paged:
            # 验证页式缓存的先决条件
            if config.nbits == 8:
                logger.info("Using PagedPQCache with transposed V storage optimization")
                return ContextList([
                    Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation),
                    Injector(LlamaSdpaAttention, 'forward', attn_forward_paged_kernel)
                ])
            else:
                logger.warning(f"PagedPQCache requires nbits=8, but got {config.nbits}. Falling back to standard implementation.")
                # 降级到合适的实现
                if config.nbits == 8:
                    return ContextList([
                        Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation), 
                        Injector(LlamaSdpaAttention, 'forward', attn_forward_custom_kernel)
                    ])
                else:
                    return ContextList([
                        Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation),
                        Injector(LlamaSdpaAttention, 'forward', attn_forward)
                    ])
        
        # nbits=8时使用custom kernel
        elif config.nbits == 8:
            logger.info("Using DynamicPQCache with custom CUDA kernels")
            return ContextList([
                Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation), 
                Injector(LlamaSdpaAttention, 'forward', attn_forward_custom_kernel)
            ])
        
        # 其他情况使用torch实现
        else:
            logger.info(f"Using DynamicPQCache with PyTorch implementation (nbits={config.nbits})")
            return ContextList([
                Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation),
                Injector(LlamaSdpaAttention, 'forward', attn_forward)
            ])
            
    except Exception as e:
        logger.error(f"Failed to create evaluation context: {e}. Using fallback torch implementation.")
        return ContextList([
            Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation),
            Injector(LlamaSdpaAttention, 'forward', attn_forward)
        ])

model_context = ModelContext(
    init_context=ContextList([]),
    baseline_context=ContextList([
        Injector(LlamaSdpaAttention, 'forward', baseline_forward)
    ]),
    sampling_context=ContextList([
        Injector(LlamaSdpaAttention, 'forward', save_forward)
    ]),
    evaluation_context=_create_evaluation_context()
)