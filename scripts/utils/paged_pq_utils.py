import torch
from typing import Dict
from .pq_utils import sa_encode_4d_keops, sa_decode_4d, DynamicPQCache
from .Timer import Timer
import logging

logger = logging.getLogger(__name__)


class PagedPQCache(DynamicPQCache):
    """
    简化的页式PQ缓存，采用连续存储策略
    
    主要特性：
    1. 扩展残差缓存从64到128 tokens  
    2. V矩阵采用按页转置的连续存储
    3. 简化内存管理，无需动态页面分配
    """
    
    def __init__(self, *, bs, nh, num_key_value_heads, M, layer_num, 
                 dtype=torch.uint8, nbits=8, d=128, scalar_t=torch.float32,
                 page_size=64, extended_residual_size=128, max_pages_per_layer=None):
        """
        初始化简化的页式PQ缓存
        
        Args:
            page_size: 页面大小（tokens），默认64，仅用于转置逻辑
            extended_residual_size: 扩展残差缓存大小，默认128
            max_pages_per_layer: 保留参数，兼容性考虑
        """
        # 页式转置参数
        self.page_size = page_size  # 64 tokens per page，仅用于转置逻辑
        self.extended_residual_size = extended_residual_size  # 128 tokens

        # 手动设置父类的属性，避免调用init_cache
        self.bs = bs
        self.nh = nh
        self.num_key_value_heads = num_key_value_heads
        self.M = M
        self.layer_num = layer_num
        self.dtype = dtype
        self.nbits = nbits
        self.d = d
        self.scalar_t = scalar_t
        
        # 设置父类的其他属性
        self.max_residual_length = extended_residual_size  # 直接设置为扩展大小
        
        # 导入KernelRegistry
        from .pq_utils import KernelRegistry
        self.registery = KernelRegistry(M=M, d=d, nbits=nbits, nh=nh, scalar_t=scalar_t)
        
        # 重新初始化缓存（覆盖父类的初始化）
        self.init_cache()
        
        logger.info(f"PagedPQCache initialized with contiguous storage: "
                   f"{layer_num} layers, {extended_residual_size} residual tokens, "
                   f"page_size={page_size} (for transpose logic only)")
        
        # 检查页式内核可用性
        # self._check_paged_kernel_availability()
        
        logger.info(f"PagedPQCache initialized: {layer_num} layers, {extended_residual_size} residual tokens, "
                   f"{page_size} tokens/page")
        
    
    
    def init_cache(self):
        """初始化缓存结构"""
        # K矩阵：使用传统存储（row-wise），优化attention计算
        self.key_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M), dtype=self.dtype, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        # V矩阵：使用转置存储格式 (bs, nh_k, M, 0)
        self.value_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, self.M, 0), dtype=self.dtype, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        # 改进：优化残差缓存大小，根据实际使用场景动态调整
        # 预填充阶段：使用较小的残差缓存（64 tokens）
        # 解码阶段：使用较大的残差缓存（128 tokens）
        prefill_residual_size = self.page_size  # 64 tokens
        decoding_residual_size = self.extended_residual_size  # 128 tokens
        
        # 预填充残差缓存（较小，节省内存）
        self.key_prefill_residual = [
            torch.zeros((self.bs, self.num_key_value_heads, prefill_residual_size, self.d), 
                       dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        self.value_prefill_residual = [
            torch.zeros((self.bs, self.num_key_value_heads, prefill_residual_size, self.d), 
                       dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        # 解码残差缓存（较大，支持长序列）
        self.key_residual_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, decoding_residual_size, self.d), 
                       dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        self.value_residual_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, decoding_residual_size, self.d), 
                       dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        # 重置计数器
        self.seen_tokens = [0 for _ in range(self.layer_num)]
        self.residualed_tokens = [0 for _ in range(self.layer_num)]
        
        # 改进：添加内存使用统计
        total_memory_mb = (
            sum(cache.numel() * cache.element_size() for cache in self.key_cache + self.value_cache) +
            sum(cache.numel() * cache.element_size() for cache in self.key_prefill_residual + self.value_prefill_residual) +
            sum(cache.numel() * cache.element_size() for cache in self.key_residual_cache + self.value_residual_cache)
        ) / (1024 * 1024)
        
        logger.info(f"PagedPQCache memory initialized: {total_memory_mb:.2f} MB "
                   f"(K_traditional: {len(self.key_cache)} layers, "
                   f"V_contiguous: {len(self.value_cache)} layers, "
                   f"residual: {prefill_residual_size + decoding_residual_size} tokens)")
    
    def flush_to_pages(self, layer_idx: int):
        """
        将残差缓存的前64个token flush到存储
        
        改进：K矩阵使用传统存储，V矩阵使用按页转置后连续存储
        
        Args:
            layer_idx: 层索引
        """
        logger.info(f"flush_to_pages被调用: layer_idx={layer_idx}, residualed_tokens={self.residualed_tokens[layer_idx]}, page_size={self.page_size}")
        
        if self.residualed_tokens[layer_idx] < self.page_size:
            logger.debug(f"Layer {layer_idx}: Not enough tokens to flush ({self.residualed_tokens[layer_idx]}/{self.page_size})")
            return  # 不足64个token，无需flush
        
        logger.debug(f"Layer {layer_idx}: Flushing {self.page_size} tokens to pages")
        
        # 添加调试信息
        logger.debug(f"flush_to_pages开始时的缓存形状:")
        logger.debug(f"  key_residual_cache[{layer_idx}]: {self.key_residual_cache[layer_idx].shape}")
        logger.debug(f"  value_residual_cache[{layer_idx}]: {self.value_residual_cache[layer_idx].shape}")
        
        # 提取前64个token的KV
        k_to_flush = self.key_residual_cache[layer_idx][:, :, :self.page_size, :]  # (bs, nh_k, 64, d)
        v_to_flush = self.value_residual_cache[layer_idx][:, :, :self.page_size, :]  # (bs, nh_k, 64, d)
        
        logger.debug(f"提取的KV形状:")
        logger.debug(f"  k_to_flush: {k_to_flush.shape}")
        logger.debug(f"  v_to_flush: {v_to_flush.shape}")
        
        # 改进：K矩阵使用传统存储（row-wise），优化attention计算
        k_codes = sa_encode_4d_keops(k_to_flush, self.key_cent, target_dtype=self.dtype)  # (bs, nh_k, 64, M)
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_codes], dim=2)
        
        logger.debug(f"K矩阵使用传统存储，更新后的形状: {self.key_cache[layer_idx].shape}")
        
        # V矩阵：使用连续存储（与K一致），按页转置后存储
        v_codes = sa_encode_4d_keops(v_to_flush, self.value_cent, target_dtype=self.dtype)  # (bs, nh_k, 64, M)
        
        logger.debug(f"V编码后的形状: {v_codes.shape}")
        
        # 按页转置后连续存储到value_cache
        # 对每个页面进行转置，然后拼接
        transposed_v_codes = v_codes.permute(0, 1, 3, 2).contiguous()  # (bs, nh_k, M, 64)
        # 直接存储转置后的数据
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], transposed_v_codes], dim=3)
        
        logger.debug(f"Layer {layer_idx}: V matrix stored continuously in value_cache "
                   f"(shape: {self.value_cache[layer_idx].shape}) after page-wise transposition")
        
        # 移动残差缓存：保留后面的token
        remaining_tokens = self.residualed_tokens[layer_idx] - self.page_size
        
        logger.debug(f"移动残差缓存前的状态:")
        logger.debug(f"  remaining_tokens: {remaining_tokens}")
        logger.debug(f"  key_residual_cache[{layer_idx}]形状: {self.key_residual_cache[layer_idx].shape}")
        logger.debug(f"  value_residual_cache[{layer_idx}]形状: {self.value_residual_cache[layer_idx].shape}")
        
        if remaining_tokens > 0:
            # 将后面的token移到前面
            # 使用clone()避免引用问题
            key_src = self.key_residual_cache[layer_idx][:, :, self.page_size:self.residualed_tokens[layer_idx], :].clone()
            value_src = self.value_residual_cache[layer_idx][:, :, self.page_size:self.residualed_tokens[layer_idx], :].clone()
            
            # 清空目标区域
            self.key_residual_cache[layer_idx][:, :, :remaining_tokens, :].fill_(0)
            self.value_residual_cache[layer_idx][:, :, :remaining_tokens, :].fill_(0)
            
            # 复制数据
            self.key_residual_cache[layer_idx][:, :, :remaining_tokens, :] = key_src
            self.value_residual_cache[layer_idx][:, :, :remaining_tokens, :] = value_src
        else:
            # 如果remaining_tokens <= 0，清空残差缓存
            self.key_residual_cache[layer_idx].fill_(0)
            self.value_residual_cache[layer_idx].fill_(0)
        
        # 更新计数
        self.residualed_tokens[layer_idx] = remaining_tokens
        self.seen_tokens[layer_idx] += self.page_size
        
        logger.debug(f"Layer {layer_idx}: Flush completed. Remaining residual tokens: {remaining_tokens}")
    
    def test_method(self):
        print("DEBUG: PagedPQCache.test_method called")
        return "test"
    
    def prefill(self, query_states, key_states, value_states, layer_idx, distort_recent=False):
        """
        预填充方法 - 优化预填充阶段的存储
        
        改进：K矩阵使用传统存储（优化attention计算），V矩阵使用按页转置后连续存储（优化内存访问）
        
        Args:
            query_states: Query张量 (bs, nh, prefill_length, d)
            key_states: Key张量 (bs, nh_k, prefill_length, d)
            value_states: Value张量 (bs, nh_k, prefill_length, d)
            layer_idx: 层索引
            distort_recent: 是否在prefill阶段使用量化数据
            
        Returns:
            attention_output: 注意力输出
        """
        
        # 标记当前处于prefill阶段，避免页面覆盖
        self._is_prefill_phase = True
        
        with Timer("PagedPQCache.prefill"):
            prefill_length = key_states.size(2)
            
            # 总是进行量化编码和存储
            with Timer("PagedPQCache.prefill.encode"):
                key_codes = sa_encode_4d_keops(key_states, self.key_cent, target_dtype=self.dtype)
                value_codes = sa_encode_4d_keops(value_states, self.value_cent, target_dtype=self.dtype)
                # 减少同步调用，只在必要时同步
                # torch.cuda.synchronize()
            
            # 改进：K矩阵使用传统存储（row-wise），优化attention计算
            with Timer("PagedPQCache.prefill.store_k_traditional"):
                # K矩阵直接存储到key_cache，保持传统格式
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
                # 减少同步调用
                # torch.cuda.synchronize()
                
                logger.info(f"Layer {layer_idx}: K matrix stored traditionally in key_cache "
                           f"(shape: {self.key_cache[layer_idx].shape})")
            
            # import ipdb; ipdb.set_trace()
            # V矩阵：按页转置后，连续存储（与K一致）
            with Timer("PagedPQCache.prefill.store_v_contiguous"):
                # 按页粒度进行转置，然后连续存储
                batch_size = self.page_size
                num_batches = (prefill_length + batch_size - 1) // batch_size
                
                # 存储转置后的所有chunk
                transposed_chunks = []
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, prefill_length)
                    chunk = value_codes[:, :, start_idx:end_idx, :]  # (bs, nh_k, cur_page_tokens, M)
                    
                    # 页内转置：(bs, nh_k, cur_page_tokens, M) -> (bs, nh_k, M, cur_page_tokens)
                    transposed_chunk = chunk.permute(0, 1, 3, 2).contiguous()  # (bs, nh_k, M, cur_page_tokens)
                    
                    # # 为了与K保持一致的存储格式，再转回：(bs, nh_k, M, cur_page_tokens) -> (bs, nh_k, cur_page_tokens, M)
                    # final_chunk = transposed_chunk.permute(0, 1, 3, 2).contiguous()  # (bs, nh_k, cur_page_tokens, M)
                    
                    transposed_chunks.append(transposed_chunk)
                
                # 连续存储所有转置后的chunk
                if transposed_chunks:
                    # 转置后的chunk形状是 (bs, nh_k, M, cur_page_tokens)
                    # 直接拼接转置后的数据，保持转置格式
                    all_transposed = torch.cat(transposed_chunks, dim=3)  # (bs, nh_k, M, total_tokens)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], all_transposed], dim=3)
                
                logger.info(f"Layer {layer_idx}: V matrix stored contiguously after page-wise transposition "
                           f"(shape: {self.value_cache[layer_idx].shape}, {num_batches} pages processed)")

            # 如果需要distort_recent，反量化用于attention计算
            if distort_recent:
                with Timer("PagedPQCache.prefill.decode"):
                    key_states = sa_decode_4d(key_codes, self.key_cent)
                    value_states = sa_decode_4d(value_codes, self.value_cent)
                    
            
            # 执行注意力计算
            with Timer("PagedPQCache.prefill.attention"):
                from torch.nn.functional import scaled_dot_product_attention as sdpa
                from transformers.models.llama.modeling_llama import repeat_kv

                num_heads = query_states.size(1)
                num_kv_heads = key_states.size(1)
                nrep = num_heads // num_kv_heads

                key_states = repeat_kv(key_states, nrep)
                value_states = repeat_kv(value_states, nrep)
                
                attention_output = sdpa(query_states, key_states, value_states, is_causal=True)
                
                logger.debug(f"PagedPQCache prefill completed: layer={layer_idx}, "
                           f"K_traditional={self.key_cache[layer_idx].size(2)} tokens, "
                           f"V_transposed={self.value_cache[layer_idx].size(3)} tokens")
                
                # 更新seen_tokens计数
                self.seen_tokens[layer_idx] += prefill_length
        
                # 清除prefill阶段标记
                self._is_prefill_phase = False
                
                return attention_output
    
    def update(self, key_states, value_states, layer_idx, distort_recent=False):
        """
        重写父类的update方法，确保正确管理存储
        
        Args:
            key_states: Key张量 (bs, nh_k, update_length, d)
            value_states: Value张量 (bs, nh_k, update_length, d)
            layer_idx: 层索引
            distort_recent: 是否在prefill阶段使用量化数据
            
        Returns:
            key_states, value_states: 更新后的KV状态
        """
        # 直接调用父类的update方法，因为prefill阶段现在直接进行存储
        # 不需要复杂的残差缓存管理
        key_states, value_states = super().update(key_states, value_states, layer_idx, distort_recent)
        
        return key_states, value_states
    
    def decoding_with_pages(self, query_states, key_states, value_states, layer_idx):
        """
        使用存储的解码attention - 调用真正的页式CUDA内核
        
        Args:
            query_states: Query张量 (bs, nh, 1, d)
            key_states: Key张量 (bs, nh_k, 1, d)  
            value_states: Value张量 (bs, nh_k, 1, d)
            layer_idx: 层索引
            
        Returns:
            attention_output: 注意力输出
        """
        # 确保当前处于decoding阶段
        self._is_prefill_phase = False
        
        try:
            # 检查是否需要flush
            if self.residualed_tokens[layer_idx] >= self.extended_residual_size:
                logger.debug(f"Layer {layer_idx}: Residual cache full, flushing to pages")
                self.flush_to_pages(layer_idx)
            
            # 添加新token到残差缓存
            r = self.residualed_tokens[layer_idx]
            n = key_states.size(2)  # 新token数量
            
            # 安全检查：确保残差缓存有足够空间
            if r + n > self.extended_residual_size:
                logger.warning(f"Layer {layer_idx}: Residual cache overflow detected: {r + n} > {self.extended_residual_size}. Force flushing.")
                self.flush_to_pages(layer_idx)
                r = self.residualed_tokens[layer_idx]
            
            # 安全检查：残差缓存状态
            if r + n > self.extended_residual_size:
                logger.warning(f"Layer {layer_idx}: Residual cache near full: {r + n}/{self.extended_residual_size}")
            
            self.key_residual_cache[layer_idx][:, :, r:r+n, :] = key_states
            self.value_residual_cache[layer_idx][:, :, r:r+n, :] = value_states
            self.residualed_tokens[layer_idx] += n
            self.seen_tokens[layer_idx] += n
            
            logger.debug(f"Layer {layer_idx}: Added {n} tokens, total residual: {self.residualed_tokens[layer_idx]}")
            
            # 调用真正的页式CUDA内核
            logger.info(f"Layer {layer_idx}: 🎯 About to call paged kernel with residual_length={r + n}")
            return self._call_paged_kernel(query_states, layer_idx, r + n)
            
        except Exception as e:
            logger.error(f"PagedPQCache decoding failed at layer {layer_idx}: {e}. Falling back to standard decoding.")
            # 如果页式处理失败，回退到标准处理
            # 确保状态一致性
            try:
                self.seen_tokens[layer_idx] += key_states.size(2)
                return super().decoding(query_states, key_states, value_states, layer_idx)
            except Exception as fallback_e:
                logger.critical(f"Fallback decoding also failed: {fallback_e}")
                raise
    
    def _call_paged_kernel(self, query_states, layer_idx, residual_length):
        """
        调用页式CUDA内核进行attention计算 - 适配连续存储的V矩阵
        
        Args:
            query_states: Query张量 (bs, nh, 1, d)
            layer_idx: 层索引
            residual_length: 残差缓存长度
            
        Returns:
            attention_output: 注意力输出
        """
        try:
            # 获取当前层的缓存数据
            current_device = query_states.device
            bs = query_states.size(0)
            nh = query_states.size(1)
            nh_k = self.num_key_value_heads
            
            # 准备Key数据：量化缓存 + 残差缓存
            key_codes = self.key_cache[layer_idx].to(current_device)  # (bs, nh_k, nk, M)
            key_cents = self.key_cent.to(current_device)  # (M, C, d/M)
            key_residuals = self.key_residual_cache[layer_idx][:, :, :residual_length, :].to(current_device)  # (bs, nh_k, r, d)
            
            # 准备V数据：连续存储的转置缓存 + 残差缓存
            value_codes = self.value_cache[layer_idx].to(current_device)  # (bs, nh_k, M, tokens) - 连续存储的转置格式
            value_cents = self.value_cent.to(current_device)  # (M, C, d/M)
            value_residuals = self.value_residual_cache[layer_idx][:, :, :residual_length, :].to(current_device)  # (bs, nh_k, r, d)
            
            # 计算页面相关参数
            total_tokens = value_codes.size(3)  # 总token数量
            n_pages = (total_tokens + self.page_size - 1) // self.page_size  # 向上取整
            page_size = self.page_size
            
            # 确保页面数量与页面池匹配
            if n_pages == 0:
                logger.warning(f"Layer {layer_idx}: No pages available, using fallback")
                return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
            
            # 构建虚拟页面ID（连续存储，页面ID就是连续的）
            # 确保页面ID在有效范围内，避免内核访问越界
            value_page_ids = torch.arange(n_pages, dtype=torch.int64, device=current_device)
            value_page_ids = value_page_ids.unsqueeze(0).unsqueeze(0).expand(bs, nh_k, -1)  # (bs, nh_k, n_pages)
            
            logger.debug(f"Layer {layer_idx}: value_page_ids shape: {value_page_ids.shape}, max_page_id: {value_page_ids.max()}")
            
            logger.debug(f"Layer {layer_idx}: value_page_ids shape: {value_page_ids.shape}, max_page_id: {value_page_ids.max()}")
            
            # 构建页面池 - 直接reshape连续存储的V数据
            # V的当前形状：(bs, nh_k, M, tokens) - 连续存储的转置格式
            # 页式内核期望的形状：(max_pages, M, page_size)
            
            # 计算实际需要的页面数（向上取整）
            actual_pages_needed = (total_tokens + page_size - 1) // page_size
            max_pages = actual_pages_needed
            
            # 直接reshape V数据以适配页式内核接口
            # 取第一个batch和head的数据：(M, tokens)
            v_data = value_codes[0, 0, :, :]  # (M, tokens)
            
            # 确保数据在正确的设备上
            v_data = v_data.to(current_device)
            
            logger.debug(f"Layer {layer_idx}: V data shape: {v_data.shape}, total_tokens: {total_tokens}, page_size: {page_size}")
            
            # 如果tokens数量不是page_size的整数倍，需要padding
            if total_tokens % page_size != 0:
                padding_size = page_size - (total_tokens % page_size)
                # 在最后一个维度添加padding
                v_data = torch.cat([v_data, torch.zeros((self.M, padding_size), dtype=self.dtype, device=current_device)], dim=1)
                total_tokens_padded = total_tokens + padding_size
                logger.debug(f"Layer {layer_idx}: Added padding: {padding_size}, new shape: {v_data.shape}")
            else:
                total_tokens_padded = total_tokens
            
            # Reshape为页式内核期望的形状：(M, tokens) -> (M, pages, page_size) -> (pages, M, page_size)
            # 注意：tokens维度需要重新组织为(pages, page_size)
            try:
                # 确保数据大小正确
                expected_size = self.M * max_pages * page_size
                actual_size = v_data.numel()
                
                if actual_size != expected_size:
                    logger.error(f"Layer {layer_idx}: Data size mismatch! expected={expected_size}, actual={actual_size}")
                    logger.error(f"Layer {layer_idx}: v_data.shape={v_data.shape}, max_pages={max_pages}, page_size={page_size}")
                    raise ValueError(f"Data size mismatch: {actual_size} != {expected_size}")
                
                v_data_reshaped = v_data.view(self.M, max_pages, page_size).transpose(0, 1).contiguous()  # (pages, M, page_size)
                logger.debug(f"Layer {layer_idx}: Reshaped V data: {v_data_reshaped.shape}")
                
                # 验证reshape结果
                if v_data_reshaped.shape != (max_pages, self.M, page_size):
                    logger.error(f"Layer {layer_idx}: Reshape result shape mismatch! expected=({max_pages}, {self.M}, {page_size}), actual={v_data_reshaped.shape}")
                    raise ValueError(f"Reshape result shape mismatch: {v_data_reshaped.shape}")
                    
            except Exception as e:
                logger.error(f"Layer {layer_idx}: Reshape failed: {e}")
                logger.error(f"Layer {layer_idx}: v_data.shape={v_data.shape}, max_pages={max_pages}, page_size={page_size}")
                raise
            
            # 创建页面池张量
            value_page_pool = v_data_reshaped  # (max_pages, M, page_size)
            
            # 验证页面池的有效性
            if value_page_pool.shape[0] != max_pages:
                logger.error(f"Layer {layer_idx}: Page pool shape mismatch! expected_pages={max_pages}, actual_pages={value_page_pool.shape[0]}")
                raise ValueError(f"Page pool shape mismatch: {value_page_pool.shape[0]} != {max_pages}")
            
            # 验证页面ID的有效性
            if value_page_ids.max() >= max_pages:
                logger.error(f"Layer {layer_idx}: Invalid page IDs detected! max_page_id={value_page_ids.max()}, max_pages={max_pages}")
                raise ValueError(f"Page ID out of bounds: {value_page_ids.max()} >= {max_pages}")
            
            logger.debug(f"Layer {layer_idx}: value_page_pool shape: {value_page_pool.shape}, dtype: {value_page_pool.dtype}")
            
            # 确保数据类型匹配
            if value_page_pool.dtype != torch.uint8:
                logger.warning(f"Layer {layer_idx}: Converting value_page_pool from {value_page_pool.dtype} to uint8")
                value_page_pool = value_page_pool.to(torch.uint8)
            
            # 获取预分配的缓冲区
            kernel_registry = self.registery
            l = self.seen_tokens[layer_idx]
            
            # 从pq_utils导入l2Ns函数
            from .pq_utils import l2Ns
            Ns = l2Ns(l)
            
            # 确保必要的缓冲区已经存在
            if Ns not in kernel_registry.partial_out_buffers:
                logger.warning(f"Layer {layer_idx}: Buffer for Ns={Ns} not found, creating it")
                # 创建必要的缓冲区
                bs = 1  # typical for inference
                nh = self.nh
                d = self.d
                scalar_t = self.scalar_t
                
                partial_out_buffer = torch.empty(bs, nh, Ns+1, d, dtype=scalar_t, device='cuda')
                partial_lse_buffer = torch.empty(bs, nh, Ns+1, dtype=scalar_t, device='cuda')
                
                kernel_registry.partial_out_buffers[Ns] = partial_out_buffer
                kernel_registry.partial_lse_buffers[Ns] = partial_lse_buffer
            
            partial_out_buffer = kernel_registry.partial_out_buffers[Ns]
            partial_lse_buffer = kernel_registry.partial_lse_buffers[Ns]
            
            # 构建页式内核函数名
            # 根据当前参数选择合适的内核
            paged_func_name = f"flash_decoding_paged_v_f16u8_Ns{Ns}Lt{self.extended_residual_size}d{self.d}M{self.M}C256"
            
            logger.info(f"Layer {layer_idx}: 🚀 Calling paged kernel: {paged_func_name}")
            
            try:
                # 导入bindings并获取页式内核函数
                import sys
                import importlib.util
                from pathlib import Path
                
                # 获取项目根目录
                current_file = Path(__file__).resolve()
                project_root = current_file.parent
                while project_root.parent != project_root:
                    if (project_root / "scripts").exists():
                        break
                    project_root = project_root.parent
                
                local_bindings_path = str(project_root / "scripts" / "modeldb" / "bindings")
                local_bindings_so = Path(local_bindings_path) / "bindings.cpython-312-x86_64-linux-gnu.so"
                
                if not local_bindings_so.exists():
                    raise FileNotFoundError(f"Local bindings.so not found: {local_bindings_so}")
                
                # 强制加载本地bindings模块
                spec = importlib.util.spec_from_file_location("bindings", local_bindings_so)
                local_bindings = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_bindings)
                
                logger.info(f"✅ Forced loading local bindings from: {local_bindings_so}")
                
                # 验证内核函数是否存在
                if not hasattr(local_bindings, paged_func_name):
                    logger.warning(f"Layer {layer_idx}: Paged kernel {paged_func_name} not found, trying alternative kernels")
                    # 尝试找到可用的页式内核
                    available_kernels = [f for f in dir(local_bindings) if 'flash_decoding_paged' in f]
                    if available_kernels:
                        # 选择第一个可用的页式内核
                        paged_func_name = available_kernels[0]
                        logger.info(f"Layer {layer_idx}: Using alternative kernel: {paged_func_name}")
                    else:
                        raise AttributeError("No paged kernels available")
                
                paged_kernel_func = getattr(local_bindings, paged_func_name)
                
                # 输入数据健康检查
                def check_tensor_health(tensor, name):
                    if torch.isnan(tensor).any():
                        logger.error(f"Layer {layer_idx}: {name} contains NaN values")
                        return False
                    if torch.isinf(tensor).any():
                        logger.error(f"Layer {layer_idx}: {name} contains Inf values")
                        return False
                    return True
                
                # 检查所有输入张量
                input_health_checks = [
                    check_tensor_health(query_states, "query_states"),
                    check_tensor_health(key_codes, "key_codes"),
                    check_tensor_health(key_cents, "key_cents"),
                    check_tensor_health(key_residuals, "key_residuals"),
                    check_tensor_health(value_page_ids, "value_page_ids"),
                    check_tensor_health(value_page_pool, "value_page_pool"),
                    check_tensor_health(value_cents, "value_cents"),
                    check_tensor_health(value_residuals, "value_residuals")
                ]
                
                if not all(input_health_checks):
                    logger.error(f"Layer {layer_idx}: Input data health check failed")
                    logger.info(f"Layer {layer_idx}: Falling back to standard decoding due to input data issues")
                    return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
                
                # 调用页式内核 - 按照Interface.cu中的参数顺序
                try:
                    attn_output = paged_kernel_func(
                        query_states,  # (bs, nh, 1, d)
                        key_codes,     # (bs, nh_k, nk, M)
                        key_cents,     # (M, C, d/M)
                        key_residuals, # (bs, nh_k, r, d)
                        value_page_ids, # (bs, nh_k, n_pages)
                        value_page_pool, # (max_pages, M, page_size)
                        value_cents,   # (M, C, d/M)
                        value_residuals, # (bs, nh_k, r, d)
                        residual_length, # r
                        n_pages,       # number of pages per batch-head
                        page_size,     # tokens per page
                        partial_out_buffer,  # partial_out_buffer
                        partial_lse_buffer   # partial_lse_buffer
                    )
                except RuntimeError as e:
                    if "illegal memory access" in str(e):
                        logger.error(f"Layer {layer_idx}: CUDA memory access error in paged kernel: {e}")
                        logger.error(f"Layer {layer_idx}: This suggests page pool or page ID access issues")
                        return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
                    else:
                        raise
                
                # 检查输出结果是否包含NaN值
                if torch.isnan(attn_output).any():
                    nan_count = torch.isnan(attn_output).sum().item()
                    total_elements = attn_output.numel()
                    nan_percentage = (nan_count / total_elements) * 100
                    
                    logger.warning(f"Layer {layer_idx}: ⚠️  PAGED KERNEL PRODUCED NaN VALUES")
                    logger.warning(f"  NaN count: {nan_count} / {total_elements} ({nan_percentage:.4f}%)")
                    
                    # 显示前几个NaN位置
                    nan_positions = torch.where(torch.isnan(attn_output))
                    if len(nan_positions[0]) > 0:
                        first_nan_positions = []
                        for i in range(min(5, len(nan_positions[0]))):
                            pos = [nan_positions[j][i].item() for j in range(len(nan_positions))]
                            first_nan_positions.append(pos)
                        logger.warning(f"  First few NaN positions: {first_nan_positions}")
                    
                    logger.warning(f"Layer {layer_idx}: Preserving NaN output for debugging purposes")
                    logger.warning(f"  NOTE: This NaN output will propagate to subsequent layers")
                    logger.warning(f"  RECOMMENDATION: Use CUDA_LAUNCH_BLOCKING=1 for detailed kernel debugging")
                    
                    # 返回NaN输出，让上层处理
                    return attn_output
                
                logger.info(f"Layer {layer_idx}: Paged kernel executed successfully")
                return attn_output
                
            except (AttributeError, ImportError) as e:
                logger.warning(f"Layer {layer_idx}: Paged kernel {paged_func_name} not available: {e}")
                return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
                
        except Exception as e:
            logger.error(f"Layer {layer_idx}: Failed to call paged kernel: {e}")
            logger.error(f"Layer {layer_idx}: Stack trace:", exc_info=True)
            
            # 最后的fallback
            return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
    
    def _call_continuous_kernel(self, query_states, layer_idx, residual_length):
        """
        使用连续存储的标准内核进行attention计算
        
        Args:
            query_states: Query张量 (bs, nh, 1, d)
            layer_idx: 层索引
            residual_length: 残差缓存长度
            
        Returns:
            attention_output: 注意力输出
        """
        try:
            # 获取当前层的缓存数据
            current_device = query_states.device
            bs = query_states.size(0)
            nh = query_states.size(1)
            nh_k = self.num_key_value_heads
            
            # 准备Key数据：量化缓存 + 残差缓存
            key_codes = self.key_cache[layer_idx].to(current_device)  # (bs, nh_k, nk, M)
            key_cents = self.key_cent.to(current_device)  # (M, C, d/M)
            key_residuals = self.key_residual_cache[layer_idx][:, :, :residual_length, :].to(current_device)  # (bs, nh_k, r, d)
            
            # 准备V数据：连续存储的转置缓存 + 残差缓存
            value_codes = self.value_cache[layer_idx].to(current_device)  # (bs, nh_k, M, tokens) - 连续存储的转置格式
            value_cents = self.value_cent.to(current_device)  # (M, C, d/M)
            value_residuals = self.value_residual_cache[layer_idx][:, :, :residual_length, :].to(current_device)  # (bs, nh_k, r, d)
            
            # 使用标准的allocated_buffer内核，但传入连续存储的V数据
            # 需要将V数据转置回标准格式
            value_codes_standard = value_codes.transpose(2, 3).contiguous()  # (bs, nh_k, tokens, M)
            
            # 获取预分配的缓冲区
            kernel_registry = self.registery
            l = self.seen_tokens[layer_idx]
            
            # 从pq_utils导入l2Ns函数
            from .pq_utils import l2Ns
            Ns = l2Ns(l)
            
            # 确保必要的缓冲区已经存在
            if Ns not in kernel_registry.partial_out_buffers:
                logger.warning(f"Layer {layer_idx}: Buffer for Ns={Ns} not found, creating it")
                # 创建必要的缓冲区
                bs = 1  # typical for inference
                nh = self.nh
                d = self.d
                scalar_t = self.scalar_t
                
                partial_out_buffer = torch.empty(bs, nh, Ns+1, d, dtype=scalar_t, device='cuda')
                partial_lse_buffer = torch.empty(bs, nh, Ns+1, dtype=scalar_t, device='cuda')
                
                kernel_registry.partial_out_buffers[Ns] = partial_out_buffer
                kernel_registry.partial_lse_buffers[Ns] = partial_lse_buffer
            
            partial_out_buffer = kernel_registry.partial_out_buffers[Ns]
            partial_lse_buffer = kernel_registry.partial_lse_buffers[Ns]
            
            # 构建标准内核函数名
            standard_func_name = f"flash_decoding_allocated_buffer_f16u8_Ns{Ns}Lt{self.extended_residual_size}d{self.d}M{self.M}C256"
            
            logger.debug(f"Layer {layer_idx}: Calling standard kernel: {standard_func_name}")
            
            try:
                # 导入bindings并获取标准内核函数
                import sys
                import importlib.util
                from pathlib import Path
                
                # 获取项目根目录
                current_file = Path(__file__).resolve()
                project_root = current_file.parent
                while project_root.parent != project_root:
                    if (project_root / "scripts").exists():
                        break
                    project_root = project_root.parent
                
                local_bindings_path = str(project_root / "scripts" / "modeldb" / "bindings")
                local_bindings_so = Path(local_bindings_path) / "bindings.cpython-312-x86_64-linux-gnu.so"
                
                if not local_bindings_so.exists():
                    raise FileNotFoundError(f"Local bindings.so not found: {local_bindings_so}")
                
                # 强制加载本地bindings模块
                spec = importlib.util.spec_from_file_location("bindings", local_bindings_so)
                local_bindings = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_bindings)
                
                logger.info(f"✅ Forced loading local bindings from: {local_bindings_so}")
                
                # 验证内核函数是否存在
                if not hasattr(local_bindings, standard_func_name):
                    logger.warning(f"Layer {layer_idx}: Standard kernel {standard_func_name} not found, trying alternative kernels")
                    # 尝试找到可用的标准内核
                    available_kernels = [f for f in dir(local_bindings) if 'flash_decoding_allocated_buffer' in f]
                    if available_kernels:
                        # 选择第一个可用的标准内核
                        standard_func_name = available_kernels[0]
                        logger.info(f"Layer {layer_idx}: Using alternative kernel: {standard_func_name}")
                    else:
                        raise AttributeError("No standard kernels available")
                
                standard_kernel_func = getattr(local_bindings, standard_func_name)
                
                # 调用标准内核
                attn_output = standard_kernel_func(
                    query_states,  # (bs, nh, 1, d)
                    key_codes,     # (bs, nh_k, nk, M)
                    value_codes_standard,  # (bs, nh_k, tokens, M) - 转置回标准格式
                    key_cents,     # (M, C, d/M)
                    value_cents,   # (M, C, d/M)
                    key_residuals, # (bs, nh_k, r, d)
                    value_residuals, # (bs, nh_k, r, d)
                    residual_length, # r
                    partial_out_buffer,  # partial_out_buffer
                    partial_lse_buffer   # partial_lse_buffer
                )
                
                logger.info(f"Layer {layer_idx}: Standard kernel executed successfully with continuous storage")
                return attn_output
                
            except (AttributeError, ImportError) as e:
                logger.warning(f"Layer {layer_idx}: Standard kernel {standard_func_name} not available: {e}")
                return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
                
        except Exception as e:
            logger.error(f"Layer {layer_idx}: Failed to call continuous kernel: {e}")
            logger.error(f"Layer {layer_idx}: Stack trace:", exc_info=True)
            
            # 最后的fallback
            return self._fallback_to_standard_decoding(query_states, layer_idx, residual_length)
    
    def _fallback_to_standard_decoding(self, query_states, layer_idx, residual_length):
        """回退到标准decoding实现"""
        try:
            logger.info(f"Layer {layer_idx}: Using fallback to standard decoding")
            
            # 使用真实的key_states和value_states进行fallback
            real_key_states = self.key_residual_cache[layer_idx][:, :, :residual_length, :]
            real_value_states = self.value_residual_cache[layer_idx][:, :, :residual_length, :]
            
            # 调用父类的标准decoding方法
            return super().decoding(query_states, real_key_states, real_value_states, layer_idx)
            
        except Exception as fallback_e:
            logger.critical(f"Layer {layer_idx}: Fallback decoding failed: {fallback_e}")
            raise
    
    def _fallback_to_standard_decoding(self, query_states, layer_idx, residual_length):
        """
        回退到标准解码方法
        
        Args:
            query_states: Query张量 (bs, nh, 1, d)
            layer_idx: 层索引
            residual_length: 残差缓存长度
            
        Returns:
            attention_output: 注意力输出
        """
        logger.warning(f"Layer {layer_idx}: Using fallback standard decoding")
        
        try:
            # 获取当前层的缓存数据
            current_device = query_states.device
            
            # 准备Key数据：量化缓存 + 残差缓存
            key_codes = self.key_cache[layer_idx].to(current_device)  # (bs, nh_k, nk, M)
            key_cents = self.key_cent.to(current_device)  # (M, C, d/M)
            key_residuals = self.key_residual_cache[layer_idx][:, :, :residual_length, :].to(current_device)  # (bs, nh_k, r, d)
            
            # 准备V数据：连续存储的转置缓存 + 残差缓存
            value_codes = self.value_cache[layer_idx].to(current_device)  # (bs, nh_k, M, tokens) - 连续存储的转置格式
            value_cents = self.value_cent.to(current_device)  # (M, C, d/M)
            value_residuals = self.value_residual_cache[layer_idx][:, :, :residual_length, :].to(current_device)  # (bs, nh_k, r, d)
            
            # 解码Key和Value
            from .pq_utils import sa_decode_4d
            
            # 解码Key
            key_states = sa_decode_4d(key_codes, key_cents)
            # 拼接残差Key
            if residual_length > 0:
                key_states = torch.cat([key_states, key_residuals], dim=2)
            
            # 解码Value（注意：value_codes是转置存储的）
            # 需要先转置回来，然后解码
            value_codes_transposed = value_codes.transpose(2, 3).contiguous()  # (bs, nh_k, tokens, M)
            value_states = sa_decode_4d(value_codes_transposed, value_cents)
            # 拼接残差Value
            if residual_length > 0:
                value_states = torch.cat([value_states, value_residuals], dim=2)
            
            # 执行注意力计算
            from torch.nn.functional import scaled_dot_product_attention as sdpa
            from transformers.models.llama.modeling_llama import repeat_kv
            
            num_heads = query_states.size(1)
            num_kv_heads = key_states.size(1)
            nrep = num_heads // num_kv_heads
            
            key_states = repeat_kv(key_states, nrep)
            value_states = repeat_kv(value_states, nrep)
            
            attention_output = sdpa(query_states, key_states, value_states, is_causal=True)
            
            logger.debug(f"Layer {layer_idx}: Fallback attention computed successfully")
            return attention_output
            
        except Exception as e:
            logger.error(f"Layer {layer_idx}: Fallback decoding failed: {e}")
            raise
    
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        stats = {
            'layer_stats': [],
            'total_memory_usage_mb': 0,
            'memory_breakdown': {}
        }
        
        for layer_idx in range(self.layer_num):
            layer_stat = {
                'layer_idx': layer_idx,
                'seen_tokens': self.seen_tokens[layer_idx],
                'residual_tokens': self.residualed_tokens[layer_idx],
                'key_cache_tokens': self.key_cache[layer_idx].size(2),
                'value_cache_tokens': self.value_cache[layer_idx].size(3)  # 转置存储，tokens在dim=3
            }
            stats['layer_stats'].append(layer_stat)
        
        # 内存使用明细
        total_cache_memory = sum(
            cache.numel() * cache.element_size() 
            for cache in self.key_cache + self.value_cache
        ) / (1024 * 1024)
        
        total_residual_memory = sum(
            cache.numel() * cache.element_size() 
            for cache in self.key_residual_cache + self.value_residual_cache
        ) / (1024 * 1024)
        
        total_prefill_residual_memory = sum(
            cache.numel() * cache.element_size() 
            for cache in getattr(self, 'key_prefill_residual', []) + getattr(self, 'value_prefill_residual', [])
        ) / (1024 * 1024)
        
        stats['memory_breakdown'] = {
            'cache_memory_mb': total_cache_memory,
            'residual_memory_mb': total_residual_memory,
            'prefill_residual_memory_mb': total_prefill_residual_memory,
            'total_memory_mb': total_cache_memory + total_residual_memory + total_prefill_residual_memory
        }
        
        return stats
    
    def get_performance_stats(self) -> Dict:
        """获取详细的性能统计信息"""
        stats = self.get_cache_stats()
        
        # 计算内存使用效率
        total_tokens = sum(layer['key_cache_tokens'] + layer['value_cache_tokens'] for layer in stats['layer_stats'])
        memory_efficiency = (stats['memory_breakdown']['total_memory_mb'] / 
                           (total_tokens * self.M * 2 / (1024 * 1024)) * 100) if total_tokens > 0 else 0
        
        # 性能评分（0-100）
        performance_score = 0
        if memory_efficiency > 80:
            performance_score += 50
        elif memory_efficiency > 60:
            performance_score += 40
        else:
            performance_score += 30
            
        # 基于缓存使用情况的评分
        cache_utilization = total_tokens / (self.layer_num * 1000) * 100  # 假设每层1000个token为基准
        if cache_utilization > 80:
            performance_score += 50
        elif cache_utilization > 60:
            performance_score += 40
        else:
            performance_score += 30
        
        performance_stats = {
            'memory_efficiency_percent': memory_efficiency,
            'cache_utilization_percent': cache_utilization,
            'performance_score': performance_score,
            'recommendations': []
        }
        
        # 生成改进建议
        if memory_efficiency < 60:
            performance_stats['recommendations'].append("内存使用效率较低，建议优化缓存大小和内存管理")
        
        if cache_utilization < 60:
            performance_stats['recommendations'].append("缓存利用率较低，建议优化序列长度和缓存策略")
        
        if performance_score < 50:
            performance_stats['recommendations'].append("整体性能较差，建议全面优化缓存管理和内存使用")
        
        stats['performance_analysis'] = performance_stats
        return stats
    
    def print_performance_summary(self):
        """打印性能统计摘要"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*60)
        print("🚀 PagedPQCache 性能统计摘要")
        print("="*60)
        
        # 缓存使用情况
        print(f"📊 缓存使用情况:")
        total_k_tokens = sum(layer['key_cache_tokens'] for layer in stats['layer_stats'])
        total_v_tokens = sum(layer['value_cache_tokens'] for layer in stats['layer_stats'])
        print(f"   K矩阵tokens: {total_k_tokens}")
        print(f"   V矩阵tokens: {total_v_tokens}")
        print(f"   总tokens: {total_k_tokens + total_v_tokens}")
        
        # 内存使用情况
        memory_stats = stats['memory_breakdown']
        print(f"\n💾 内存使用情况:")
        print(f"   缓存内存: {memory_stats['cache_memory_mb']:.2f} MB")
        print(f"   残差缓存: {memory_stats['residual_memory_mb']:.2f} MB")
        print(f"   总内存: {memory_stats['total_memory_mb']:.2f} MB")
        
        # 性能分析
        perf_stats = stats['performance_analysis']
        print(f"\n⚡ 性能分析:")
        print(f"   内存使用效率: {perf_stats['memory_efficiency_percent']:.1f}%")
        print(f"   缓存利用率: {perf_stats['cache_utilization_percent']:.1f}%")
        print(f"   性能评分: {perf_stats['performance_score']}/100")
        
        # 改进建议
        if perf_stats['recommendations']:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(perf_stats['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("="*60)
    
    def show_current_status(self):
        """显示当前缓存状态"""
        stats = self.get_cache_stats()
        
        print("\n" + "="*50)
        print("📊 PagedPQCache 当前状态")
        print("="*50)
        
        # 基本信息
        print(f"🔧 基本信息:")
        print(f"   层数: {self.layer_num}")
        print(f"   批次大小: {self.bs}")
        print(f"   KV头数: {self.num_key_value_heads}")
        print(f"   子空间数: {self.M}")
        print(f"   页面大小: {self.page_size} tokens")
        
        # 缓存使用情况
        print(f"\n📄 缓存使用情况:")
        for layer_idx in range(self.layer_num):
            layer_stats = stats['layer_stats'][layer_idx]
            print(f"   层 {layer_idx}: K={layer_stats['key_cache_tokens']} tokens, V={layer_stats['value_cache_tokens']} tokens, "
                  f"残差={layer_stats['residual_tokens']} tokens")
        
        # 内存使用情况
        memory_stats = stats['memory_breakdown']
        print(f"\n💾 内存使用情况:")
        print(f"   缓存内存: {memory_stats['cache_memory_mb']:.2f} MB")
        print(f"   残差缓存: {memory_stats['residual_memory_mb']:.2f} MB")
        print(f"   总内存: {memory_stats['total_memory_mb']:.2f} MB")
        
        # 性能指标
        if 'performance_analysis' in stats:
            perf_stats = stats['performance_analysis']
            print(f"\n⚡ 性能指标:")
            print(f"   内存使用效率: {perf_stats['memory_efficiency_percent']:.1f}%")
            print(f"   缓存利用率: {perf_stats['cache_utilization_percent']:.1f}%")
            print(f"   性能评分: {perf_stats['performance_score']}/100")
        
        print("="*50)
    
    def get_memory_usage_summary(self) -> str:
        """获取内存使用摘要"""
        stats = self.get_cache_stats()
        memory_stats = stats['memory_breakdown']
        
        summary = (
            f"内存使用摘要:\n"
            f"  📊 总内存: {memory_stats['total_memory_mb']:.2f} MB\n"
            f"  🔧 缓存: {memory_stats['cache_memory_mb']:.2f} MB\n"
            f"  📝 残差: {memory_stats['residual_memory_mb']:.2f} MB\n"
            f"  ⚡ 预填充: {memory_stats['prefill_residual_memory_mb']:.2f} MB"
        )
        return summary
    

    
    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up PagedPQCache resources")
        
        # 统计清理前的资源使用情况
        stats_before = self.get_cache_stats()
        total_memory_before = stats_before['memory_breakdown']['total_memory_mb']
        
        logger.info(f"Before cleanup: {total_memory_before:.2f} MB")
        
        # 清理缓存数据
        for layer_idx in range(self.layer_num):
            # 清空K和V缓存
            self.key_cache[layer_idx] = torch.zeros((self.bs, self.num_key_value_heads, 0, self.M), dtype=self.dtype, device='cuda')
            self.value_cache[layer_idx] = torch.zeros((self.bs, self.num_key_value_heads, self.M, 0), dtype=self.dtype, device='cuda')
            
            # 重置计数器
            self.seen_tokens[layer_idx] = 0
            self.residualed_tokens[layer_idx] = 0
            
            # 清空残差缓存
            self.key_residual_cache[layer_idx].fill_(0)
            self.value_residual_cache[layer_idx].fill_(0)
        
        # 统计清理后的资源使用情况
        stats_after = self.get_cache_stats()
        total_memory_after = stats_after['memory_breakdown']['total_memory_mb']
        
        # 计算释放的资源
        memory_freed = total_memory_before - total_memory_after
        
        logger.info(f"After cleanup: {total_memory_after:.2f} MB")
        logger.info(f"Cleanup completed: Freed {memory_freed:.2f} MB")
        
        # 输出性能统计摘要
        # self.print_performance_summary()
    
    def _compute_attention_fallback(self, query_states, key_states, value_states, layer_idx):
        """
        fallback attention计算
        
        Args:
            query_states: Query张量
            key_states: Key张量
            value_states: Value张量
            layer_idx: 层索引
            
        Returns:
            attention_output: 注意力输出
        """
        logger.info(f"Layer {layer_idx}: Using fallback attention computation")
        
        # 如果需要distort_recent，反量化用于attention计算
        if hasattr(self, 'distort_recent') and self.distort_recent:
            with Timer("PagedPQCache.prefill.decode_fallback"):
                key_states = sa_decode_4d(self.key_cache[layer_idx], self.key_cent)
                value_states = sa_decode_4d(self.value_cache[layer_idx], self.value_cent)
                torch.cuda.synchronize()
        
        # 执行注意力计算
        with Timer("PagedPQCache.prefill.attention_fallback"):
            from torch.nn.functional import scaled_dot_product_attention as sdpa
            from transformers.models.llama.modeling_llama import repeat_kv

            num_heads = query_states.size(1)
            num_kv_heads = key_states.size(1)
            nrep = num_heads // num_kv_heads

            key_states = repeat_kv(key_states, nrep)
            value_states = repeat_kv(value_states, nrep)
            
            attention_output = sdpa(query_states, key_states, value_states, is_causal=True)
            
            logger.debug(f"PagedPQCache fallback attention completed: layer={layer_idx}")
            
            return attention_output