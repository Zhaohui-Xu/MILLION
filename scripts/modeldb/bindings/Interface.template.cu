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
