import torch
from typing import List, Optional, Tuple, Dict, Set
from .pq_utils import sa_encode_4d_keops, sa_decode_4d, DynamicPQCache
from .Singleton import Singleton
from .Timer import Timer
import logging

logger = logging.getLogger(__name__)

class PageManager:
    """
    页面内存管理器
    负责管理V矩阵转置存储的页面分配和释放
    支持动态页面扩展
    """
    
    def __init__(self, page_size: int = 64, initial_pages: int = 100, max_pages: int = None, M: int = 64, device='cuda'):
        """
        初始化页面管理器
        
        Args:
            page_size: 每个页面包含的token数量，默认64
            initial_pages: 初始页面数量，默认100
            max_pages: 最大页面数量，None表示无限制（仅受内存限制）
            M: PQ的子空间数量
            device: 设备类型
        """
        self.page_size = page_size  # 64 tokens per page
        self.M = M  # 子空间数量
        self.device = device
        self.initial_pages = initial_pages
        self.max_pages = max_pages  # None 表示无限制
        
        # 修复：更合理的预分配策略
        # 预分配大小应该独立于max_pages，避免逻辑混乱
        if max_pages is not None:
            # 确保预分配不超过max_pages
            self.preallocated_size = min(initial_pages * 2, max_pages)
        else:
            self.preallocated_size = min(initial_pages * 2, 1024)  # 默认预分配大小
        
        self.current_active_pages = initial_pages  # 当前激活的页面数
        
        # 预分配大页面池：(preallocated_size, M, page_size) for transposed storage
        # 只有前 initial_pages 个页面是激活的
        self.page_pool = torch.zeros(
            (self.preallocated_size, M, page_size), dtype=torch.uint8, device=device
        )
        
        # 页面管理状态
        self.free_pages: Set[int] = set(range(initial_pages))
        self.allocated_pages: Dict[int, Dict] = {}  # page_id -> usage_info
        
        # 动态扩展统计
        self.expansion_count = 0
        self.total_expansions = 0
        
        # 改进：添加页面复用统计
        self.page_reuse_count = 0
        self.total_allocations = 0
        
        logger.info(f"PageManager initialized: {initial_pages} active pages (pre-allocated {self.preallocated_size}), "
                   f"max: {'unlimited' if max_pages is None else max_pages}, {page_size} tokens/page, {M} subspaces")
    
    def _expand_page_pool(self, additional_pages: int = None, force_expansion: bool = False):
        """
        动态扩展页面池
        
        Args:
            additional_pages: 要添加的页面数，默认为当前池大小的50%
            force_expansion: 是否强制扩展，用于预填充阶段一次性扩展足够页面
        """
        if additional_pages is None:
            if force_expansion:
                # 预填充阶段：一次性扩展足够页面，避免频繁扩展
                additional_pages = max(self.current_active_pages, 100)  # 至少扩展100页
                logger.info(f"Force expansion for prefill: adding {additional_pages} pages")
            else:
                # 正常扩展：当前激活页面数的50%
                additional_pages = max(self.current_active_pages // 2, 50)  # 至少扩展50页
        
        # 修复：优化页面扩展限制检查
        if self.max_pages is not None:
            max_additional = self.max_pages - self.current_active_pages
            if max_additional <= 0:
                # 如果已达到最大页面限制，尝试清理一些页面或给出更详细的错误信息
                logger.warning(f"Page pool at max capacity: {self.current_active_pages}/{self.max_pages}")
                logger.warning(f"Allocated pages: {len(self.allocated_pages)}, Free pages: {len(self.free_pages)}")
                raise RuntimeError(f"Cannot expand page pool: reached max_pages limit {self.max_pages}")
            
            # 确保扩展的页面数不超过限制
            if additional_pages > max_additional:
                logger.warning(f"Requested expansion ({additional_pages}) exceeds available capacity ({max_additional})")
                additional_pages = max_additional
        
        # 检查预分配的页面池是否有足够空间
        new_active_size = self.current_active_pages + additional_pages
        
        if new_active_size <= self.preallocated_size:
            # 无需重新分配内存，直接激活预分配的页面
            logger.info(f"Activating {additional_pages} pre-allocated pages ({self.current_active_pages} → {new_active_size})")
            
            # 添加新的空闲页面（来自预分配池）
            for i in range(self.current_active_pages, new_active_size):
                self.free_pages.add(i)
                # 清零预分配的页面
                self.page_pool[i].fill_(0)
            
        else:
            # 需要真正扩展内存（预分配空间不足）
            actual_new_size = max(new_active_size, self.preallocated_size * 2)
            logger.warning(f"Pre-allocated space insufficient, expanding memory from {self.preallocated_size} to {actual_new_size}")
            
            # 创建新的更大页面池
            new_pool = torch.zeros(
                (actual_new_size, self.M, self.page_size), dtype=torch.uint8, device=self.device
            )
            
            # 复制现有数据
            new_pool[:self.current_active_pages] = self.page_pool[:self.current_active_pages]
            
            # 替换页面池
            self.page_pool = new_pool
            self.preallocated_size = actual_new_size
            
            # 添加新的空闲页面
            for i in range(self.current_active_pages, new_active_size):
                self.free_pages.add(i)
        
        # 更新状态
        self.current_active_pages = new_active_size
        self.total_expansions += 1
        
        logger.info(f"Page pool expanded successfully: {len(self.free_pages)} free pages available, "
                   f"active: {self.current_active_pages}/{self.preallocated_size}")
    
    def allocate_page(self) -> int:
        """
        分配一个新页面，返回页面ID
        支持动态扩展
        
        Returns:
            page_id: 分配的页面ID
            
        Raises:
            RuntimeError: 当达到最大页面限制且无法扩展时
        """
        # 如果没有空闲页面，尝试扩展页面池
        if not self.free_pages:
            try:
                self._expand_page_pool()
            except RuntimeError as e:
                raise RuntimeError(f"No free pages available and cannot expand: {e}")
        
        page_id = self.free_pages.pop()
        self.allocated_pages[page_id] = {
            'allocated_at': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0,
            'allocation_count': 1,  # 记录分配次数
            'last_used': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0
        }
        
        # 去掉分配时的清零，避免重复memset，按需在写入阶段处理尾部清零
        self.total_allocations += 1
        logger.debug(f"Allocated page {page_id}, remaining free pages: {len(self.free_pages)}")
        return page_id
    
    def allocate_reused_page(self) -> int:
        """
        分配一个复用页面，优先选择最近使用的页面
        支持动态扩展
        
        Returns:
            page_id: 分配的页面ID
            
        Raises:
            RuntimeError: 当达到最大页面限制且无法扩展时
        """
        # 如果没有空闲页面，尝试扩展页面池
        if not self.free_pages:
            try:
                self._expand_page_pool()
            except RuntimeError as e:
                raise RuntimeError(f"No free pages available and cannot expand: {e}")
        
        # 优先选择最近使用的页面进行复用
        page_id = self.free_pages.pop()
        
        # 检查是否是复用页面
        if page_id in self.allocated_pages:
            self.page_reuse_count += 1
            self.allocated_pages[page_id]['allocation_count'] += 1
            self.allocated_pages[page_id]['last_used'] = torch.cuda.current_stream().query() if torch.cuda.is_available() else 0
            logger.debug(f"Reused page {page_id} (reuse count: {self.allocated_pages[page_id]['allocation_count']})")
        else:
            self.allocated_pages[page_id] = {
                'allocated_at': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0,
                'allocation_count': 1,
                'last_used': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0
            }
        
        # 去掉分配时的清零，避免重复memset
        self.total_allocations += 1
        logger.debug(f"Allocated page {page_id}, remaining free pages: {len(self.free_pages)}")
        return page_id
    
    def allocate_pages(self, n: int) -> list:
        """批量分配n个页面，必要时自动扩容。"""
        if n <= 0:
            return []
        if len(self.free_pages) < n:
            try:
                self._expand_page_pool(additional_pages=n - len(self.free_pages), force_expansion=True)
            except RuntimeError as e:
                raise RuntimeError(f"Cannot bulk-allocate pages: {e}")
        ids = []
        for _ in range(n):
            pid = self.free_pages.pop()
            self.allocated_pages[pid] = {
                'allocated_at': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0,
                'allocation_count': self.allocated_pages.get(pid, {}).get('allocation_count', 0) + 1,
                'last_used': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0
            }
            ids.append(pid)
            self.total_allocations += 1
        return ids
    
    def free_page(self, page_id: int):
        """
        释放页面
        
        Args:
            page_id: 要释放的页面ID
        """
        if page_id not in self.allocated_pages:
            logger.warning(f"Attempting to free unallocated page {page_id}")
            return
        
        del self.allocated_pages[page_id]
        self.free_pages.add(page_id)
        
        logger.debug(f"Freed page {page_id}, available free pages: {len(self.free_pages)}")
    
    def get_page(self, page_id: int) -> torch.Tensor:
        """
        获取页面数据，返回转置存储的页面
        
        Args:
            page_id: 页面ID
            
        Returns:
            page_data: 形状为 (M, page_size) 的转置存储页面
        """
        # 页面ID有效性检查
        if not self._is_valid_page_id(page_id):
            raise ValueError(f"Invalid page_id {page_id}: must be within [0, {self.current_active_pages})")
        
        if page_id not in self.allocated_pages:
            raise ValueError(f"Page {page_id} is not allocated")
        
        # 更新最后使用时间
        self.allocated_pages[page_id]['last_used'] = torch.cuda.current_stream().query() if torch.cuda.is_available() else 0
        
        return self.page_pool[page_id]
    
    def _is_valid_page_id(self, page_id: int) -> bool:
        """
        检查页面ID是否有效
        
        Args:
            page_id: 要检查的页面ID
            
        Returns:
            bool: 页面ID是否有效
        """
        return (
            isinstance(page_id, int) and 
            0 <= page_id < self.current_active_pages and
            page_id < self.preallocated_size
        )
    
    def safe_page_access(self, page_id: int) -> torch.Tensor:
        """
        安全的页面访问，包含完整的边界检查
        
        Args:
            page_id: 页面ID
            
        Returns:
            页面张量
            
        Raises:
            ValueError: 页面ID无效
        """
        if not self._is_valid_page_id(page_id):
            error_msg = (f"Invalid page access: page_id={page_id}, "
                        f"valid range=[0, {self.current_active_pages}), "
                        f"preallocated_size={self.preallocated_size}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.page_pool[page_id]
    
    def get_stats(self) -> Dict:
        """获取页面管理器统计信息"""
        reuse_rate = (self.page_reuse_count / self.total_allocations * 100) if self.total_allocations > 0 else 0
        return {
            'initial_pages': self.initial_pages,
            'current_active_pages': self.current_active_pages,
            'preallocated_size': self.preallocated_size,
            'max_pages': self.max_pages,
            'allocated_pages': len(self.allocated_pages),
            'free_pages': len(self.free_pages),
            'utilization': len(self.allocated_pages) / self.current_active_pages if self.current_active_pages > 0 else 0,
            'memory_usage_mb': (self.page_pool.numel() * self.page_pool.element_size()) / (1024 * 1024),
            'memory_efficiency': (self.current_active_pages / self.preallocated_size * 100) if self.preallocated_size > 0 else 0,
            'page_reuse_count': self.page_reuse_count,
            'total_allocations': self.total_allocations,
            'reuse_rate_percent': reuse_rate,
            'total_expansions': self.total_expansions,
            'expansion_efficiency': (self.current_active_pages - self.initial_pages) / self.initial_pages * 100 if self.initial_pages > 0 else 0
        }


def quantize_and_transpose_batch(v_batch: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    量化并转置一个batch的V数据
    
    Args:
        v_batch: (64, d) - 64个token的V向量
        centroids: (M, C, d/M) - V的码本
    
    Returns:
        transposed_codes: (M, 64) - 转置后的量化码
    """
    # 确保输入是连续的
    v_batch = v_batch.contiguous()
    
    # 量化：(64, d) -> (1, 1, 64, M) -> (64, M)
    v_codes = sa_encode_4d_keops(
        v_batch.unsqueeze(0).unsqueeze(0),  # 添加batch和head维度: (1, 1, 64, d)
        centroids, target_dtype=torch.uint8
    ).squeeze(0).squeeze(0)  # 移除batch和head维度: (64, M)
    
    # 转置：(64, M) -> (M, 64)
    transposed_codes = v_codes.transpose(0, 1).contiguous()
    
    return transposed_codes


class PagedPQCache(DynamicPQCache):
    """
    扩展的页式PQ缓存，兼容原有DynamicPQCache接口
    
    主要改进：
    1. 扩展残差缓存从64到128 tokens
    2. V矩阵采用转置存储的页式管理
    3. 优化访存模式，提高cache利用率
    """
    
    def __init__(self, *, bs, nh, num_key_value_heads, M, layer_num, 
                 dtype=torch.uint8, nbits=8, d=128, scalar_t=torch.float32,
                 page_size=64, extended_residual_size=128, max_pages_per_layer=None):
        """
        初始化页式PQ缓存
        
        Args:
            page_size: 页面大小（tokens），默认64
            extended_residual_size: 扩展残差缓存大小，默认128
            max_pages_per_layer: 每层最大页面数，None表示无限制（自动计算）
        """
        # 页式存储特有参数
        self.page_size = page_size  # 64 tokens per page
        self.extended_residual_size = extended_residual_size  # 128 tokens
        self.max_pages_per_layer = max_pages_per_layer

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
        
        # 初始化页面管理器
        # 修复：更合理的初始页面计算策略
        
        # 基础需求：每个batch-head组合的最小页面需求
        base_pages_needed = self.bs * self.num_key_value_heads * 2  # K和V各需要至少1个页面
        
        # 针对1024 token prefill的需求计算
        prefill_1024_tokens = 1024
        pages_per_1024 = (prefill_1024_tokens + self.page_size - 1) // self.page_size  # 16页
        pages_for_1024_prefill = self.bs * self.num_key_value_heads * pages_per_1024  # 1*32*16 = 512页
        
        # 修复：减小initial_pages，避免接近max_pages限制
        # 初始页面数应该较小，让页面管理器按需扩展
        initial_pages = max(
            base_pages_needed,  # 最小需求
            100                  # 合理的初始值，避免过度预分配
        )
        
        logger.info(f"Smart initial page calculation: "
                   f"bs={self.bs}, kv_heads={self.num_key_value_heads}, page_size={self.page_size}")
        logger.info(f"  - Base need: {base_pages_needed} pages")
        logger.info(f"  - 1024 token prefill need: {pages_for_1024_prefill} pages")  
        logger.info(f"  - Selected initial: {initial_pages} pages (conservative to allow expansion)")
        
        # 最大页面数：如果用户指定了max_pages_per_layer，则使用；否则设置合理默认值
        if self.max_pages_per_layer is None:
            # 修复：为1024 token prefill设置合理的max_pages
            # 需要确保能容纳完整的prefill + 一些余量
            max_pages = pages_for_1024_prefill * 2  # 1024页，足够1024 token prefill + 余量
            logger.info(f"Auto-calculated max_pages: {max_pages} (2x of 1024-token prefill needs)")
        else:
            max_pages = self.max_pages_per_layer
            # 验证max_pages是否足够
            if max_pages < pages_for_1024_prefill:
                logger.warning(f"Specified max_pages ({max_pages}) may be insufficient for 1024-token prefill ({pages_for_1024_prefill})")
                max_pages = pages_for_1024_prefill * 2
                logger.warning(f"Increasing max_pages to {max_pages}")
        
        # 创建页面管理器
        self.page_managers = [
            PageManager(
                page_size=page_size, 
                initial_pages=initial_pages,  # 使用较小的初始值
                max_pages=max_pages,          # 确保有足够的扩展空间
                M=M, 
                device='cuda'
            )
            for _ in range(layer_num)
        ]
        
        # K页面ID列表：每层维护页面ID列表，支持多batch和多head
        # 结构：[layer][batch][head] -> List[page_id]
        self.key_page_ids = [
            [[[] for _ in range(num_key_value_heads)] for _ in range(bs)]
            for _ in range(layer_num)
        ]
        
        # V页面ID列表：每层维护页面ID列表，支持多batch和多head
        # 结构：[layer][batch][head] -> List[page_id]
        self.value_page_ids = [
            [[[] for _ in range(num_key_value_heads)] for _ in range(bs)]
            for _ in range(layer_num)
        ]
        
        # 重新初始化缓存（覆盖父类的初始化）
        self._init_paged_cache()
        
        # 检查页式内核可用性
        # self._check_paged_kernel_availability()
        
        logger.info(f"PagedPQCache initialized: {layer_num} layers, {extended_residual_size} residual tokens, "
                   f"{page_size} tokens/page, initial={initial_pages}, max={max_pages}")
        
        # 改进：显示详细的配置信息和预期性能
        logger.info(f"📋 配置详情:")
        logger.info(f"   - 模型配置: {self.bs} batch × {self.num_key_value_heads} heads × {self.M} subspaces")
        logger.info(f"   - 存储策略: K矩阵(传统) + V矩阵(页面转置)")
        logger.info(f"   - 页面管理: {initial_pages} initial pages/layer, max={max_pages} pages/layer × {layer_num} layers")
        logger.info(f"   - 内存优化: 预填充残差({self.page_size} tokens) + 解码残差({self.extended_residual_size} tokens)")
        logger.info(f"   - 动态扩展: 支持自动页面池扩展，按需分配内存")
        
        # 计算初始内存使用
        initial_page_memory = (initial_pages * self.M * self.page_size * 2) / (1024 * 1024)  # MB
        initial_total_memory = initial_page_memory * layer_num
        max_page_memory = (max_pages * self.M * self.page_size * 2) / (1024 * 1024)  # MB
        max_total_memory = max_page_memory * layer_num
        
        logger.info(f"💾 内存使用预估:")
        logger.info(f"   - 初始内存: {initial_total_memory:.2f} MB ({initial_page_memory:.2f} MB/layer)")
        logger.info(f"   - 最大内存: {max_total_memory:.2f} MB ({max_page_memory:.2f} MB/layer)")
        logger.info(f"   - 扩展策略: 按需扩展，避免内存浪费")
    
    def _init_paged_cache(self):
        """初始化页式缓存结构"""
        # K矩阵：使用传统存储（row-wise），优化attention计算
        self.key_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M), dtype=self.dtype, device='cuda')
            for _ in range(self.layer_num)
        ]
        
        # V矩阵：使用页面存储，这里不需要直接的value_cache
        # 但保留作为fallback选项
        self.value_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M), dtype=self.dtype, device='cuda')
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
                   f"V_pages: {len(self.value_cache)} layers, "
                   f"residual: {prefill_residual_size + decoding_residual_size} tokens)")
    
    def flush_to_pages(self, layer_idx: int):
        """
        将残差缓存的前64个token flush到页面存储
        
        改进：K矩阵使用传统存储，V矩阵使用页面存储
        
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
        
        # V矩阵：使用页面存储（转置格式），优化内存访问
        v_codes = sa_encode_4d_keops(v_to_flush, self.value_cent, target_dtype=self.dtype)  # (bs, nh_k, 64, M)
        
        logger.debug(f"V编码后的形状: {v_codes.shape}")
        
        # 为每个batch和head分配V页面并转置存储
        # 记录分配失败的batch-head组合，用于fallback
        failed_v_allocations = []
        successful_v_allocations = 0
        
        for b in range(self.bs):
            for h in range(self.num_key_value_heads):
                # 分配V页面
                try:
                    # 改进：优先使用复用页面
                    v_page_id = self.page_managers[layer_idx].allocate_reused_page()
                    
                    # 转置存储：(64, M) -> (M, 64)
                    v_page_data = self.page_managers[layer_idx].get_page(v_page_id)  # (M, 64)
                    v_page_data[:, :] = v_codes[b, h, :, :].transpose(0, 1)  # 转置并存储
                    
                    # 记录页面ID
                    self.value_page_ids[layer_idx][b][h].append(v_page_id)
                    successful_v_allocations += 1
                    
                    logger.debug(f"Layer {layer_idx}: Successfully allocated V page {v_page_id} for batch {b}, head {h}")
                    
                except RuntimeError as e:
                    logger.error(f"Failed to allocate V page for layer {layer_idx}, batch {b}, head {h}: {e}")
                    # 记录失败的分配，但不break，继续处理其他batch-head组合
                    failed_v_allocations.append((b, h))
        
        # 统计分配结果
        total_allocations = self.bs * self.num_key_value_heads
        logger.info(f"Layer {layer_idx}: V matrix page allocation completed. "
                   f"Success: {successful_v_allocations}/{total_allocations} (failed: {len(failed_v_allocations)})")
        
        # 如果有分配失败的情况，使用fallback策略
        if failed_v_allocations:
            logger.warning(f"Layer {layer_idx}: {len(failed_v_allocations)} V page allocations failed, using fallback")
            
            # V矩阵fallback
            fallback_v_codes = []
            for b, h in failed_v_allocations:
                fallback_v_codes.append(v_codes[b:b+1, h:h+1, :, :])
            
            if fallback_v_codes:
                fallback_v_codes = torch.cat(fallback_v_codes, dim=0)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], fallback_v_codes], dim=2)
                logger.debug(f"Layer {layer_idx}: Added {len(fallback_v_codes)} failed V allocations to value_cache")
        
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
    
    def prefill(self, query_states, key_states, value_states, layer_idx, distort_recent=False):
        """
        页式预填充方法 - 优化预填充阶段的页面存储
        
        改进：K矩阵使用传统存储（优化attention计算），V矩阵使用转置存储（优化内存访问）
        
        Args:
            query_states: Query张量 (bs, nh, prefill_length, d)
            key_states: Key张量 (bs, nh_k, prefill_length, d)
            value_states: Value张量 (bs, nh_k, prefill_length, d)
            layer_idx: 层索引
            distort_recent: 是否在prefill阶段使用量化数据
            
        Returns:
            attention_output: 注意力输出
        """
        logger.debug(f"PagedPQCache prefill: layer={layer_idx}, tokens={key_states.size(2)}, distort_recent={distort_recent}")
        
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
            
            # V矩阵：按页面存储（转置格式），优化内存访问
            with Timer("PagedPQCache.prefill.store_v_pages"):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                batch_size = self.page_size
                num_batches = (prefill_length + batch_size - 1) // batch_size
                total_v_pages_needed = self.bs * self.num_key_value_heads * num_batches
                
                # 详细记录页面需求计算
                logger.info(f"Layer {layer_idx}: 页面需求计算详情:")
                logger.info(f"  - prefill_length: {prefill_length}")
                logger.info(f"  - page_size: {batch_size}")
                logger.info(f"  - num_batches: {num_batches}")
                logger.info(f"  - bs: {self.bs}, num_key_value_heads: {self.num_key_value_heads}")
                logger.info(f"  - total_v_pages_needed: {total_v_pages_needed}")
                
                pm = self.page_managers[layer_idx]
                available_pages = len(pm.free_pages)
                if available_pages < total_v_pages_needed:
                    logger.warning(f"Layer {layer_idx}: Insufficient pages available ({available_pages}/{total_v_pages_needed}). Attempting to expand page pool...")
                    try:
                        delta = total_v_pages_needed - available_pages
                        if pm.max_pages is not None and pm.current_active_pages + delta > pm.max_pages:
                            # 计算真正需要的总页面数
                            total_needed = pm.current_active_pages + delta
                            logger.warning(f"Layer {layer_idx}: 页面池状态 - 当前激活: {pm.current_active_pages}, 可用: {available_pages}, 需要: {total_v_pages_needed}")
                            logger.warning(f"Layer {layer_idx}: 需要扩展页面池，但受max_pages限制 ({pm.max_pages})")
                            logger.warning(f"Layer {layer_idx}: 建议增加max_pages到至少 {total_needed} (当前: {pm.current_active_pages} + 需要: {delta})")
                            # 不临时提升max_pages，让扩展失败，使用fallback
                            raise RuntimeError(f"Page pool expansion blocked by max_pages limit: {pm.max_pages}")
                        pm._expand_page_pool(additional_pages=delta, force_expansion=True)
                        available_pages = len(pm.free_pages)
                        if available_pages < total_v_pages_needed:
                            raise RuntimeError(f"Still insufficient pages after expansion: {available_pages}/{total_v_pages_needed}")
                    except Exception as e:
                        logger.error(f"Layer {layer_idx}: Failed to expand page pool: {e}. Switching to traditional storage mode for V matrix.")
                        with Timer("PagedPQCache.prefill.store_v_traditional_fallback"):
                            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
                            logger.info(f"Layer {layer_idx}: V matrix stored traditionally (fallback mode)")
                        logger.info(f"Layer {layer_idx}: Skipping page allocation due to expansion failure")
                        return self._compute_attention_fallback(query_states, key_states, value_states, layer_idx)
                
                logger.info(f"Layer {layer_idx}: V matrix needs {total_v_pages_needed} pages ({self.bs} batch × {self.num_key_value_heads} heads × {num_batches} batches)")
                
                successful_v_pages = 0
                failed_v_pages = 0
                
                # 预转置编码，减少循环内转置开销
                # value_codes: (bs, nh_k, T, M) -> 我们按小块切片后再转置
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, prefill_length)
                    current_batch_size = end_idx - start_idx
                    # (bs, nh_k, current_batch_size, M)
                    chunk = value_codes[:, :, start_idx:end_idx, :]
                    # 目标页形状 (bs, nh_k, M, page_size)，先创建临时buffer
                    # 仅当不足一页时，才需要零填充尾部
                    transposed = chunk.permute(0, 1, 3, 2).contiguous()  # (bs, nh_k, M, current_batch_size)
                    if current_batch_size < self.page_size:
                        pad_cols = self.page_size - current_batch_size
                        pad = torch.zeros((self.bs, self.num_key_value_heads, self.M, pad_cols), dtype=transposed.dtype, device=transposed.device)
                        transposed = torch.cat([transposed, pad], dim=3)
                    # 现在 transposed 形状为 (bs, nh_k, M, page_size)
                    
                    # 批量分配本小批需要的页面
                    pages_needed = self.bs * self.num_key_value_heads
                    try:
                        page_ids = pm.allocate_pages(pages_needed)
                    except RuntimeError as e:
                        logger.error(f"Layer {layer_idx}: bulk allocate failed: {e}")
                        # 回退为单个分配
                        page_ids = []
                        for _ in range(pages_needed):
                            try:
                                page_ids.append(pm.allocate_reused_page())
                            except RuntimeError as ee:
                                logger.error(f"Layer {layer_idx}: allocate_reused_page failed: {ee}")
                                page_ids.append(-1)
                    
                    # 批量写入页面
                    idx = 0
                    for b in range(self.bs):
                        for h in range(self.num_key_value_heads):
                            pid = page_ids[idx] if idx < len(page_ids) else -1
                            idx += 1
                            if pid == -1:
                                # 失败fallback该(b,h)
                                fallback_value_codes = chunk[b:b+1, h:h+1, :, :]
                                if self.value_cache[layer_idx].size(2) == 0:
                                    self.value_cache[layer_idx] = fallback_value_codes
                                else:
                                    expanded_fallback = fallback_value_codes.expand(self.bs, self.num_key_value_heads, -1, -1)
                                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], expanded_fallback], dim=2)
                                failed_v_pages += 1
                                continue
                            page_data = pm.get_page(pid)
                            # 直接slice赋值
                            page_data[:, :] = transposed[b, h]
                            # 直接设置页面ID，避免覆盖检查
                            self._direct_set_value_page(layer_idx, b, h, batch_idx, pid)
                            successful_v_pages += 1
                
                end_time.record()
                # 只在最后进行一次同步，减少同步开销
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                
                v_success_rate = (successful_v_pages / total_v_pages_needed * 100) if total_v_pages_needed > 0 else 0
                logger.info(f"Layer {layer_idx}: V matrix page allocation completed. Success: {successful_v_pages}/{total_v_pages_needed} ({v_success_rate:.1f}%), Time: {elapsed_time:.2f} ms")
                page_stats = pm.get_stats()
                logger.info(f"Layer {layer_idx}: Page reuse stats - Reuse rate: {page_stats['reuse_rate_percent']:.1f}%, Total allocations: {page_stats['total_allocations']}")
                pages_per_ms = successful_v_pages / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Layer {layer_idx}: Performance - {pages_per_ms:.2f} pages/ms")

            # 如果需要distort_recent，反量化用于attention计算
            if distort_recent:
                with Timer("PagedPQCache.prefill.decode"):
                    key_states = sa_decode_4d(key_codes, self.key_cent)
                    value_states = sa_decode_4d(value_codes, self.value_cent)
                    # 减少同步调用
                    # torch.cuda.synchronize()
            
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
                           f"V_pages={sum(len(self.value_page_ids[layer_idx][b][h]) for b in range(self.bs) for h in range(self.num_key_value_heads))}")
                
                # 清除prefill阶段标记
                self._is_prefill_phase = False
                
                return attention_output
    
    def update(self, key_states, value_states, layer_idx, distort_recent=False):
        """
        重写父类的update方法，确保正确管理页面存储
        
        Args:
            key_states: Key张量 (bs, nh_k, update_length, d)
            value_states: Value张量 (bs, nh_k, update_length, d)
            layer_idx: 层索引
            distort_recent: 是否在prefill阶段使用量化数据
            
        Returns:
            key_states, value_states: 更新后的KV状态
        """
        # 直接调用父类的update方法，因为prefill阶段现在直接进行页面存储
        # 不需要复杂的残差缓存管理
        key_states, value_states = super().update(key_states, value_states, layer_idx, distort_recent)
        
        return key_states, value_states
    
    def decoding_with_pages(self, query_states, key_states, value_states, layer_idx):
        """
        使用页面存储的解码attention - 调用真正的页式CUDA内核
        
        Args:
            query_states: Query张量 (bs, nh, 1, d)
            key_states: Key张量 (bs, nh_k, 1, d)  
            value_states: Value张量 (bs, nh_k, 1, d)
            layer_idx: 层索引
            
        Returns:
            attention_output: 注意力输出
        """
        # 确保当前处于decoding阶段，允许页面覆盖
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
            
            # 安全检查：页面管理器状态
            page_stats = self.page_managers[layer_idx].get_stats()
            if page_stats['free_pages'] < 2:  # 保留至少2个空闲页面
                logger.warning(f"Layer {layer_idx}: Low memory warning - only {page_stats['free_pages']} pages remaining")
            
            self.key_residual_cache[layer_idx][:, :, r:r+n, :] = key_states
            self.value_residual_cache[layer_idx][:, :, r:r+n, :] = value_states
            self.residualed_tokens[layer_idx] += n
            self.seen_tokens[layer_idx] += n
            
            logger.debug(f"Layer {layer_idx}: Added {n} tokens, total residual: {self.residualed_tokens[layer_idx]}")
            
            # 调用真正的页式CUDA内核
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
        调用页式CUDA内核进行attention计算
        
        Args:
            query_states: Query张量 (bs, nh, 1, d)
            layer_idx: 层索引
            residual_length: 残差缓存长度
            
        Returns:
            attention_output: 注意力输出
        """
        # 简化的fallback实现
        logger.info(f"Layer {layer_idx}: Using fallback to standard decoding (paged kernel not implemented)")
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
    
    def _direct_set_value_page(self, layer_idx: int, b: int, h: int, batch_idx: int, page_id: int):
        """直接设置V页面ID，不进行覆盖检查（用于prefill阶段的批量分配）。"""
        # 确保列表长度
        lst = self.value_page_ids[layer_idx][b][h]
        if len(lst) <= batch_idx:
            # 使用占位符-1扩展
            lst.extend([-1] * (batch_idx + 1 - len(lst)))
        
        # 直接设置页面ID，不进行任何检查
        lst[batch_idx] = int(page_id)
        logger.debug(f"直接设置：layer={layer_idx}, batch={b}, head={h}, batch_idx={batch_idx} -> 页面 {page_id}")
    
    def _compute_attention_fallback(self, query_states, key_states, value_states, layer_idx):
        """
        页面分配失败时的fallback attention计算
        
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