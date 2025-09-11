#include "Kernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using u8 = uint8_t;
using f16 = core::Scalar::AtScalar::f16;

template
<
	typename scalar_t,
	typename code_t,
	const int Ns,
	const int Lt,
	const int d,
	const int M,
	const int C
>
torch::Tensor flash_decoding_allocated_buffer
(
	const torch::Tensor query, // shaped (bs, nh, 1, d)
	const torch::Tensor key_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor value_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor key_cents, // shaped (M, C, d/M)
	const torch::Tensor value_cents, // shaped (M, C, d/M)
	const torch::Tensor key_residuals, // shaped (bs, nh_k, Lt, d)
	const torch::Tensor value_residuals, // shaped (bs, nh_k, Lt, d)
	const int r, // residual length, 0 < r <= Lt
	const torch::Tensor partial_out_buffer, // shaped (bs, nh, Ns+1, d)
	const torch::Tensor partial_lse_buffer // shaped (bs, nh, Ns+1)
)
{
	const int bs = query.size(0);
	const int nh = query.size(1);
	const int nh_k = key_codes.size(1);
	const int nk = key_codes.size(2);

	const int Ls = (nk + Ns - 1) / Ns; // number of key_codes and value_codes inside a split.

	// compute ad_lut, shaped (bs, nh, M, 1, C). Could also be implemented as a custom CUDA kernel.
	// but come on, this is just a simple matmul. torch handles it well.
	auto query_reshaped = query.reshape({bs, nh, 1, M, d/M}).transpose(2, 3);
	auto ad_lut = at::matmul(query_reshaped, key_cents.transpose(1, 2));
	query_reshaped = torch::Tensor(); // release memory

	// convert torch type to cuda type
	using cuscalar_t = typename cs::at2cu<scalar_t>::type;

	// get pointer to partial_out, partial_lse
	cuscalar_t *partial_out = reinterpret_cast<cuscalar_t*>(partial_out_buffer.data_ptr<scalar_t>());
	cuscalar_t *partial_lse = reinterpret_cast<cuscalar_t*>(partial_lse_buffer.data_ptr<scalar_t>());

	// Fill partial_out[:, :, :Ns] and partial_lse[:, :, :Ns]
	{
		dim3 split_grid(bs, nh, Ns);
		dim3 split_block(Lt, 1, 1);
		
		flash_decoding_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M, C><<<split_grid, split_block>>>(
			reinterpret_cast<const cuscalar_t*>(ad_lut.data_ptr<scalar_t>()),
			key_codes.data_ptr<code_t>(),
			value_codes.data_ptr<code_t>(),
			reinterpret_cast<const cuscalar_t*>(value_cents.data_ptr<scalar_t>()),
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k,
			nk,
			Ls
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	ad_lut = torch::Tensor(); // release memory

	// Fill partial_out[:, :, -1] and partial_lse[:, :, -1]
	{
		dim3 residual_grid(bs, nh);
		dim3 residual_block(d, 1, 1);
		// constexpr int DEN = sizeof(float) / sizeof(cuscalar_t); // scalar density, for example, float -> 1, half -> 2
		flash_decoding_residual_kernel<cuscalar_t, Ns, Lt, d><<<residual_grid, residual_block>>>(
			reinterpret_cast<const cuscalar_t*>(query.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(key_residuals.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(value_residuals.data_ptr<scalar_t>()),
			r,
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k
		);

		gpuErrchk(cudaPeekAtLastError());
	}
		
	// Reduce partial_out and partial_lse to out
	torch::Tensor out = torch::zeros({bs, nh, 1, d}, query.options());
	{
		dim3 reduce_grid(bs, nh);
		dim3 reduce_block(d, 1, 1);
		
		flash_decoding_reduce_kernel<cuscalar_t, Ns, d><<<reduce_grid, reduce_block>>>(
			partial_out,
			partial_lse,
			reinterpret_cast<cuscalar_t*>(out.data_ptr<scalar_t>()),
			bs,
			nh
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	return out;
}

#define register_flash_decoding_allocated_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
	torch::Tensor flash_decoding_allocated_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv( \
		const torch::Tensor query, \
		const torch::Tensor key_codes, \
		const torch::Tensor value_codes, \
		const torch::Tensor key_cents, \
		const torch::Tensor value_cents, \
		const torch::Tensor key_residuals, \
		const torch::Tensor value_residuals, \
		const int r, \
		const torch::Tensor partial_out_buffer, \
		const torch::Tensor partial_lse_buffer \
	) { \
		return flash_decoding_allocated_buffer<T, code_alias, Nsv, Ltv, dv, Mv, Cv>( \
			query, \
			key_codes, \
			value_codes, \
			key_cents, \
			value_cents, \
			key_residuals, \
			value_residuals, \
			r, \
			partial_out_buffer, \
			partial_lse_buffer \
		); \
}

template
<
	typename scalar_t,
	typename code_t,
	const int Ns,
	const int Lt,
	const int d,
	const int M,
	const int C
>
torch::Tensor flash_decoding_allocated_paged_buffer
(
	const torch::Tensor query, // shaped (bs, nh, 1, d)
	const torch::Tensor key_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor value_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor key_cents, // shaped (M, C, d/M)
	const torch::Tensor value_cents, // shaped (M, C, d/M)
	const torch::Tensor key_residuals, // shaped (bs, nh_k, Lt, d)
	const torch::Tensor value_residuals, // shaped (bs, nh_k, Lt, d)
	const int r, // residual length, 0 < r <= Lt
	const torch::Tensor partial_out_buffer, // shaped (bs, nh, Ns+1, d)
	const torch::Tensor partial_lse_buffer // shaped (bs, nh, Ns+1)
)
{
	const int bs = query.size(0);
	const int nh = query.size(1);
	const int nh_k = key_codes.size(1);// key头数
	const int nk = key_codes.size(2);// Seq长度
	
	// 切的一份seqlen的大小
	const int Ls = (nk + Ns - 1) / Ns; // number of key_codes and value_codes inside a split. 

	// compute ad_lut, shaped (bs, nh, M, 1, C). Could also be implemented as a custom CUDA kernel.
	// but come on, this is just a simple matmul. torch handles it well.
	auto query_reshaped = query.reshape({bs, nh, 1, M, d/M}).transpose(2, 3);
	auto ad_lut = at::matmul(query_reshaped, key_cents.transpose(1, 2));
	query_reshaped = torch::Tensor(); // release memory

	// convert torch type to cuda type
	using cuscalar_t = typename cs::at2cu<scalar_t>::type;

	// get pointer to partial_out, partial_lse
	cuscalar_t *partial_out = reinterpret_cast<cuscalar_t*>(partial_out_buffer.data_ptr<scalar_t>());
	cuscalar_t *partial_lse = reinterpret_cast<cuscalar_t*>(partial_lse_buffer.data_ptr<scalar_t>());

	// Fill partial_out[:, :, :Ns] and partial_lse[:, :, :Ns]
	{
		dim3 split_grid(bs, nh, Ns); // 线程块的总数量(gridDim) bs,nh,Ns: Ns是Nk的切分数
		dim3 split_block(Lt, 1, 1);// 每个线程块内多少个线程并行(blockDim) x*y*z = Lt(d)*1*1
		
		// // 计算T_dy需要的字节数 这样可以在运行时 让L1划分更多的空间给Shared Memory
		// size_t shared_mem_bytes = M * C * sizeof(cuscalar_t); // M=64时 是32KB

		// // 2. 设置内核属性，优先使用共享内存
        // //    这会告诉运行时，为这个内核选择一个能满足其共享内存需求的配置。
		// gpuErrchk(cudaFuncSetCacheConfig(
		// 	// 第一个参数：要配置的内核函数
		// 	(const void*)flash_decoding_paged_V_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M, C>,
		// 	// 第二个参数：要设置的缓存配置
		// 	cudaFuncCachePreferShared
		// ));

		// // 自动控制
		// // // --- 推荐的调整方式 ---
		// // // 告诉运行时，我们希望为这个内核优先分配最大可能的共享内存
		// // // A100上，这通常会分配 100KB 或 164KB 给共享内存
		// gpuErrchk(cudaFuncSetAttribute(
		// 	(const void*)flash_decoding_paged_V_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M, C>,
		// 	cudaFuncAttributePreferredSharedMemoryCarveout,
		// 	cudaSharedmemCarveoutMaxShared // 请求最大共享内存比例
		// ));
	
        // // (可选但推荐) 明确设置共享内存大小上限
        // // 这可以确保即使内核本身需要的共享内存不多，运行时也会分配一个较大的共享内存 carveout
        // // 从而可能为其他内核或未来的修改留出空间 <这个大小只是给动态内存使用的 所以不能占满164KB的上限>
        // gpuErrchk(cudaFuncSetAttribute(
        //     (const void*)flash_decoding_paged_V_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M, C>,
        //     cudaFuncAttributeMaxDynamicSharedMemorySize,
        //     shared_mem_bytes 
        // ));// 单个Block划分多少个动态Shared Memory


		flash_decoding_paged_V_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M, C><<<split_grid, split_block>>>(
			reinterpret_cast<const cuscalar_t*>(ad_lut.data_ptr<scalar_t>()),
			key_codes.data_ptr<code_t>(),
			value_codes.data_ptr<code_t>(),
			reinterpret_cast<const cuscalar_t*>(value_cents.data_ptr<scalar_t>()),
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k,
			nk,
			Ls
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	ad_lut = torch::Tensor(); // release memory

	// Fill partial_out[:, :, -1] and partial_lse[:, :, -1]
	{
		dim3 residual_grid(bs, nh);
		dim3 residual_block(d, 1, 1);
		// constexpr int DEN = sizeof(float) / sizeof(cuscalar_t); // scalar density, for example, float -> 1, half -> 2
		flash_decoding_residual_kernel<cuscalar_t, Ns, Lt, d><<<residual_grid, residual_block>>>(
			reinterpret_cast<const cuscalar_t*>(query.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(key_residuals.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(value_residuals.data_ptr<scalar_t>()),
			r,
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k
		);

		gpuErrchk(cudaPeekAtLastError());
	}
		
	// Reduce partial_out and partial_lse to out
	torch::Tensor out = torch::zeros({bs, nh, 1, d}, query.options());
	{
		dim3 reduce_grid(bs, nh);
		dim3 reduce_block(d, 1, 1);
		
		flash_decoding_reduce_kernel<cuscalar_t, Ns, d><<<reduce_grid, reduce_block>>>(
			partial_out,
			partial_lse,
			reinterpret_cast<cuscalar_t*>(out.data_ptr<scalar_t>()),
			bs,
			nh
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	return out;
}

#define register_flash_decoding_allocated_paged_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
	torch::Tensor flash_decoding_allocated_paged_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv( \
		const torch::Tensor query, \
		const torch::Tensor key_codes, \	
		const torch::Tensor value_codes, \
		const torch::Tensor key_cents, \
		const torch::Tensor value_cents, \
		const torch::Tensor key_residuals, \
		const torch::Tensor value_residuals, \
		const int r, \
		const torch::Tensor partial_out_buffer, \
		const torch::Tensor partial_lse_buffer \
	) { \
		return flash_decoding_allocated_paged_buffer<T, code_alias, Nsv, Ltv, dv, Mv, Cv>( \
			query, \
			key_codes, \
			value_codes, \
			key_cents, \
			value_cents, \
			key_residuals, \
			value_residuals, \
			r, \
			partial_out_buffer, \
			partial_lse_buffer \
		); \
}


template
<
	typename scalar_t,
	typename code_t,
	const int Ns,
	const int Lt,
	const int d,
	const int M,
	const int C
>
torch::Tensor flash_decoding_allocated_paged_split_qkv_buffer
(
	const torch::Tensor query, // shaped (bs, nh, 1, d)
	const torch::Tensor key_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor value_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor key_cents, // shaped (M, C, d/M)
	const torch::Tensor value_cents, // shaped (M, C, d/M)
	const torch::Tensor key_residuals, // shaped (bs, nh_k, Lt, d)
	const torch::Tensor value_residuals, // shaped (bs, nh_k, Lt, d)
	const int r, // residual length, 0 < r <= Lt
	const torch::Tensor partial_out_buffer, // shaped (bs, nh, Ns+1, d)
	const torch::Tensor partial_lse_buffer // shaped (bs, nh, Ns+1)
)
{
	const int bs = query.size(0);
	const int nh = query.size(1);
	const int nh_k = key_codes.size(1);// key头数
	const int nk = key_codes.size(2);// Seq长度
	
	// 切的一份seqlen的大小
	const int Ls = (nk + Ns - 1) / Ns; // number of key_codes and value_codes inside a split. 

	// compute ad_lut, shaped (bs, nh, M, 1, C). Could also be implemented as a custom CUDA kernel.
	// but come on, this is just a simple matmul. torch handles it well.
	auto query_reshaped = query.reshape({bs, nh, 1, M, d/M}).transpose(2, 3);
	auto ad_lut = at::matmul(query_reshaped, key_cents.transpose(1, 2));
	query_reshaped = torch::Tensor(); // release memory

	// convert torch type to cuda type
	using cuscalar_t = typename cs::at2cu<scalar_t>::type;

	// get pointer to partial_out, partial_lse
	cuscalar_t *partial_out = reinterpret_cast<cuscalar_t*>(partial_out_buffer.data_ptr<scalar_t>());
	cuscalar_t *partial_lse = reinterpret_cast<cuscalar_t*>(partial_lse_buffer.data_ptr<scalar_t>());

	// Fill partial_out[:, :, :Ns] and partial_lse[:, :, :Ns]
	{
		dim3 split_grid(bs, nh, Ns); // 线程块的总数量(gridDim) bs,nh,Ns: Ns是Nk的切分数 新增M个子空间并行
		dim3 split_block(Lt, 1, 1);// 每个线程块内多少个线程并行(blockDim) x*y*z = Lt(d)*1*1

		// 1. 创建 s_out 张量，和V计算
		//    形状为 (bs, nh, nk)
		const int actural_len = Ls * Ns;
		auto s_out_tensor = torch::empty({bs, nh, actural_len}, query.options());
		cuscalar_t *s_out = reinterpret_cast<cuscalar_t*>(s_out_tensor.data_ptr<scalar_t>());

		// 2. 创建 sum_exp_out 张量，用于存储每个 tile 的指数和
		//    形状为 (bs, nh, Ns + 1)
		auto sum_exp_out_tensor = torch::empty({bs, nh, Ns + 1}, query.options());
		cuscalar_t *sum_exp_out = reinterpret_cast<cuscalar_t*>(sum_exp_out_tensor.data_ptr<scalar_t>());

		// S要回传 s_out[bs,nh,nk] 用来和V做计算
		// sum_exp_out[bs,nh,Ns+1] 
		// lse 和 output 写回到默认位置
		flash_decoding_paged_V_split_kernel_phase_qk<cuscalar_t, code_t, Ns, Lt, d, M, C><<<split_grid, split_block>>>(
			reinterpret_cast<const cuscalar_t*>(ad_lut.data_ptr<scalar_t>()),
			key_codes.data_ptr<code_t>(),
			partial_out,
			partial_lse,
			s_out,
			sum_exp_out,
			bs,
			nh,
			nh_k,
			nk,
			Ls,
			actural_len
		);
		
		gpuErrchk(cudaPeekAtLastError());

		// 3. 启动第二个内核 out = (output + S@V)/sum_exp
		dim3 split_grid_2(bs, nh, Ns * M); 
		dim3 split_block_2(Lt, 1, 1);

		flash_decoding_paged_V_split_kernel_phase_sv<cuscalar_t, code_t, Ns, Lt, d, M, C><<<split_grid_2, split_block_2>>>(
			value_codes.data_ptr<code_t>(),
			reinterpret_cast<const cuscalar_t*>(value_cents.data_ptr<scalar_t>()),
			partial_out,
			partial_lse,
			s_out,
			sum_exp_out,
			bs,
			nh,
			nh_k,
			nk,
			Ls,
			actural_len
		);
		gpuErrchk(cudaPeekAtLastError()); // 确保 phase2 完成

		// === 阶段 3: Reduce (合并与归一化) ===
		dim3 grid_reduce(bs, nh, Ns);
		// 使用 d 个线程来并行处理 d 维向量
		dim3 block_reduce(d, 1, 1); 
		reduce_kernel<cuscalar_t, Ns, Lt, d, M><<<grid_reduce, block_reduce>>>(
			partial_out,
			sum_exp_out,
			bs,
			nh,
			nh_k,
			nk,
			Ls
		);
		gpuErrchk(cudaPeekAtLastError());


	}
	ad_lut = torch::Tensor(); // release memory

	// Fill partial_out[:, :, -1] and partial_lse[:, :, -1]
	{
		dim3 residual_grid(bs, nh);
		dim3 residual_block(d, 1, 1);
		// constexpr int DEN = sizeof(float) / sizeof(cuscalar_t); // scalar density, for example, float -> 1, half -> 2
		flash_decoding_residual_kernel<cuscalar_t, Ns, Lt, d><<<residual_grid, residual_block>>>(
			reinterpret_cast<const cuscalar_t*>(query.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(key_residuals.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(value_residuals.data_ptr<scalar_t>()),
			r,
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k
		);

		gpuErrchk(cudaPeekAtLastError());
	}
		
	// Reduce partial_out and partial_lse to out
	torch::Tensor out = torch::zeros({bs, nh, 1, d}, query.options());
	{
		dim3 reduce_grid(bs, nh);
		dim3 reduce_block(d, 1, 1);
		
		flash_decoding_reduce_kernel<cuscalar_t, Ns, d><<<reduce_grid, reduce_block>>>(
			partial_out,
			partial_lse,
			reinterpret_cast<cuscalar_t*>(out.data_ptr<scalar_t>()),
			bs,
			nh
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	return out;
}

#define register_flash_decoding_allocated_paged_split_qkv_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
	torch::Tensor flash_decoding_allocated_paged_split_qkv_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv( \
		const torch::Tensor query, \
		const torch::Tensor key_codes, \	
		const torch::Tensor value_codes, \
		const torch::Tensor key_cents, \
		const torch::Tensor value_cents, \
		const torch::Tensor key_residuals, \
		const torch::Tensor value_residuals, \
		const int r, \
		const torch::Tensor partial_out_buffer, \
		const torch::Tensor partial_lse_buffer \
	) { \
		return flash_decoding_allocated_paged_split_qkv_buffer<T, code_alias, Nsv, Ltv, dv, Mv, Cv>( \
			query, \
			key_codes, \
			value_codes, \
			key_cents, \
			value_cents, \
			key_residuals, \
			value_residuals, \
			r, \
			partial_out_buffer, \
			partial_lse_buffer \
		); \
}


template
<
	typename scalar_t,
	typename code_t,
	const int Ns,
	const int Lt,
	const int d,
	const int M,
	const int C
>
torch::Tensor flash_decoding_allocated_paged_lastblock_sync_buffer
(
	const torch::Tensor query, // shaped (bs, nh, 1, d)
	const torch::Tensor key_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor value_codes, // shaped (bs, nh_k, nk, M)
	const torch::Tensor key_cents, // shaped (M, C, d/M)
	const torch::Tensor value_cents, // shaped (M, C, d/M)
	const torch::Tensor key_residuals, // shaped (bs, nh_k, Lt, d)
	const torch::Tensor value_residuals, // shaped (bs, nh_k, Lt, d)
	const int r, // residual length, 0 < r <= Lt
	const torch::Tensor partial_out_buffer, // shaped (bs, nh, Ns+1, d)
	const torch::Tensor partial_lse_buffer // shaped (bs, nh, Ns+1)
)
{
	const int bs = query.size(0);
	const int nh = query.size(1);
	const int nh_k = key_codes.size(1);// key头数
	const int nk = key_codes.size(2);// Seq长度
	
	// 切的一份seqlen的大小
	const int Ls = (nk + Ns - 1) / Ns; // number of key_codes and value_codes inside a split. 

	// compute ad_lut, shaped (bs, nh, M, 1, C). Could also be implemented as a custom CUDA kernel.
	// but come on, this is just a simple matmul. torch handles it well.
	auto query_reshaped = query.reshape({bs, nh, 1, M, d/M}).transpose(2, 3);
	auto ad_lut = at::matmul(query_reshaped, key_cents.transpose(1, 2));
	query_reshaped = torch::Tensor(); // release memory

	// convert torch type to cuda type
	using cuscalar_t = typename cs::at2cu<scalar_t>::type;

	// get pointer to partial_out, partial_lse
	cuscalar_t *partial_out = reinterpret_cast<cuscalar_t*>(partial_out_buffer.data_ptr<scalar_t>());
	cuscalar_t *partial_lse = reinterpret_cast<cuscalar_t*>(partial_lse_buffer.data_ptr<scalar_t>());

	// Fill partial_out[:, :, :Ns] and partial_lse[:, :, :Ns]
	{
		dim3 split_grid(bs, nh, Ns* M); // 线程块的总数量(gridDim) bs,nh,Ns: Ns是Nk的切分数
		dim3 split_block(Lt, 1, 1);// 每个线程块内多少个线程并行(blockDim) x*y*z = Lt(d)*1*1
		
		int counter = 0;// 范围 bs * nh * (Ns*M)
		
		// 负责计算qk的线程块得全部写完到S中才能开始下一步[等待+S的读写] 需要等[bs, nh, Ns* M]个块的同步
		// 同步S的数据到HBM中 
		auto s_out_tensor = torch::empty({bs, nh, nk, M}, query.options());
		cuscalar_t *s_out = reinterpret_cast<cuscalar_t*>(s_out_tensor.data_ptr<scalar_t>());

		
		// flash_decoding_paged_lastblock_sync_V_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M, C><<<split_grid, split_block>>>(
		// 	reinterpret_cast<const cuscalar_t*>(ad_lut.data_ptr<scalar_t>()),
		// 	key_codes.data_ptr<code_t>(),
		// 	value_codes.data_ptr<code_t>(),
		// 	reinterpret_cast<const cuscalar_t*>(value_cents.data_ptr<scalar_t>()),
		// 	partial_out,
		// 	partial_lse,
		// 	bs,
		// 	nh,
		// 	nh_k,
		// 	nk,
		// 	Ls
		// );
		
		// gpuErrchk(cudaPeekAtLastError());
	}
	ad_lut = torch::Tensor(); // release memory

	// Fill partial_out[:, :, -1] and partial_lse[:, :, -1]
	{
		dim3 residual_grid(bs, nh);
		dim3 residual_block(d, 1, 1);
		// constexpr int DEN = sizeof(float) / sizeof(cuscalar_t); // scalar density, for example, float -> 1, half -> 2
		flash_decoding_residual_kernel<cuscalar_t, Ns, Lt, d><<<residual_grid, residual_block>>>(
			reinterpret_cast<const cuscalar_t*>(query.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(key_residuals.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(value_residuals.data_ptr<scalar_t>()),
			r,
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k
		);

		gpuErrchk(cudaPeekAtLastError());
	}
		
	// Reduce partial_out and partial_lse to out
	torch::Tensor out = torch::zeros({bs, nh, 1, d}, query.options());
	{
		dim3 reduce_grid(bs, nh);
		dim3 reduce_block(d, 1, 1);
		
		flash_decoding_reduce_kernel<cuscalar_t, Ns, d><<<reduce_grid, reduce_block>>>(
			partial_out,
			partial_lse,
			reinterpret_cast<cuscalar_t*>(out.data_ptr<scalar_t>()),
			bs,
			nh
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	return out;
}

#define register_flash_decoding_allocated_paged_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
	torch::Tensor flash_decoding_allocated_paged_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv( \
		const torch::Tensor query, \
		const torch::Tensor key_codes, \	
		const torch::Tensor value_codes, \
		const torch::Tensor key_cents, \
		const torch::Tensor value_cents, \
		const torch::Tensor key_residuals, \
		const torch::Tensor value_residuals, \
		const int r, \
		const torch::Tensor partial_out_buffer, \
		const torch::Tensor partial_lse_buffer \
	) { \
		return flash_decoding_allocated_paged_buffer<T, code_alias, Nsv, Ltv, dv, Mv, Cv>( \
			query, \
			key_codes, \
			value_codes, \
			key_cents, \
			value_cents, \
			key_residuals, \
			value_residuals, \
			r, \
			partial_out_buffer, \
			partial_lse_buffer \
		); \
}
register_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 64, 256);
register_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 16, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 16, 128);
register_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 16, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 16, 256);
register_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 32, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 32, 128);
register_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 32, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 32, 256);
register_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 64, 128);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 64, 128);
register_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 64, 256);
register_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 64, 256);
