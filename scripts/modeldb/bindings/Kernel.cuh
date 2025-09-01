#pragma once

#include "core/Scalar.cuh"
#include "core/Vector.cuh"
#include "core/Reduction.cuh"
#include "core/DeviceOps.cuh"

namespace cs  = core::Scalar;
namespace csc = core::Scalar::CuScalar;

template
<
	typename cuscalar_t, 
	typename code_t,
	int Ns, // number of splits along the nk dimension
	int Lt, // number of key_codes and value_codes inside a tile.
	int d,
	int M,
	int C
>
__global__ void flash_decoding_split_kernel
(
	const cuscalar_t * __restrict__ ad_lut, // shaped (bs, nh, M, C)
	const code_t * __restrict__ key_codes, // shaped (bs, nh_k, nk, M)
	const code_t * __restrict__ value_codes, // shaped (bs, nh_k, nk, M)
	const cuscalar_t * __restrict__ value_cents, // shaped (M, C, d_m)
	cuscalar_t * __restrict__ out, // shaped (bs, nh, Ns+1, d)
	cuscalar_t * __restrict__ lse, // shaped (bs, nh, Ns+1)
	const int bs,
	const int nh,
	const int nh_k,
	const int nk,
	const int Ls // number of key_codes and value_codes inside a split.
)
{
	/**
	 * kernel launch info:
	 * grid sized (bs, nh, Ns)
	 * block sized (Lt, 1, 1)
	*/

	using v16_t = typename core::Vector::VectorType<16>::type; // 16 bytes

	const int b = blockIdx.x;
	const int h = blockIdx.y;
	const int sid = blockIdx.z;
	const int tid = threadIdx.x;
	const cuscalar_t scale = cs::to_cuscalar<cuscalar_t>(1.0f / sqrt(d));
	constexpr int n_warps = d / 32;

    // GQA example: nh=32, nh_k=8, query with h=(0,1,2,3),(4,5,6,7) will attend to key_codes with hk=0,1
	const int hk = h / (nh / nh_k);

    __shared__ cuscalar_t S[Lt]; // regarded as [1, Lt]
	__shared__ cuscalar_t prev_rowmax;
	__shared__ cuscalar_t rowmax;
	__shared__ cuscalar_t prev_sum_exp;
	__shared__ cuscalar_t sum_exp; // denoted as l in the paper https://crfm.stanford.edu/2023/07/17/flash2.html
	__shared__ cuscalar_t online_scale;

	__shared__ cuscalar_t output[d]; // regarded as [1, d] or [1, M, d_m]
	__shared__ __align__(16) code_t local_codes[Lt*M]; 

	if (tid == blockDim.x - 1) {
		prev_rowmax  = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
		prev_sum_exp = cs::to_cuscalar<cuscalar_t>(0.0f);
		rowmax       = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
		sum_exp      = cs::to_cuscalar<cuscalar_t>(0.0f);
	}

    core::DeviceOps::block_fill_zero<cuscalar_t>(output, d, blockDim.x, tid);
    __syncthreads();

    const int lut_offset = b * (nh*M*C) + h * (M*C);
	// in a split, loop over all tiles
	const int split_j_start = sid * Ls;
	const int split_j_end = min((sid + 1) * Ls, nk);
	code_t key_code = 0;
	int local_lut_offset = 0;
    v16_t key_code_batch;

    for (int tile_j_start=split_j_start; tile_j_start<split_j_end; tile_j_start+=Lt) {
		const int tile_j_end = min(tile_j_start + Lt, split_j_end);
		const int tile_j_len = tile_j_end - tile_j_start; // <= Lt

		core::DeviceOps::block_copy<code_t>(local_codes, key_codes + b * (nh_k*nk*M) + hk * (nk*M) + tile_j_start * M, tile_j_len * M, blockDim.x, tid);
		__syncthreads();
		// S = scale * q @ KT
		if (tid < tile_j_len) { 
			local_lut_offset = 0;
			cuscalar_t sim = cs::to_cuscalar<cuscalar_t>(0.0f);	

			for (int m = 0; m < M / sizeof(v16_t); ++m) {
				key_code_batch = reinterpret_cast<v16_t&>(local_codes[tid * M + m * sizeof(v16_t)]);
				local_lut_offset = C * m * sizeof(v16_t);

				code_t *key_code_batch_ptr = reinterpret_cast<code_t*>(&key_code_batch);
				for (int i = 0; i < sizeof(v16_t); ++i) {
					key_code = key_code_batch_ptr[i];
					sim = csc::add<cuscalar_t>()(sim, ad_lut[lut_offset + local_lut_offset + i * C + key_code]);
				}
			}
			S[tid] = csc::mul<cuscalar_t>()(sim, scale);
		} else {
            S[tid] = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
        }
		__syncthreads();

		if (tid == 0) {
			// reduce S to rowmax
			rowmax = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::MaxReduction>(S, Lt);
			rowmax = csc::max<cuscalar_t>()(rowmax, prev_rowmax);
			online_scale = csc::exp<cuscalar_t>()(csc::sub<cuscalar_t>()(prev_rowmax, rowmax));
			prev_rowmax = rowmax;
		}

		__syncthreads();
		
		// subs S with rowmax, exp S
		S[tid] = csc::exp<cuscalar_t>()(csc::sub<cuscalar_t>()(S[tid], rowmax));


		// output = output * online_scale + S @ V, we decode V on-the-fly
        core::DeviceOps::block_op<cuscalar_t, core::Scalar::CuScalar::mul>(output, output, online_scale, d, blockDim.x, tid);
		core::DeviceOps::block_copy<code_t>(local_codes, value_codes + b * (nh_k*nk*M) + hk * (nk*M) + tile_j_start * M, tile_j_len * M, blockDim.x, tid);
		__syncthreads();

		// We have threads tid = 0, 1, 2, ..., Lt-1 and add S (tile_j_len) @ V (tile_j_len, M, d/M) to output (M, d/M)
		// If we parallelize over tile_j_len (which is a natural choice by blockDim.x), we have to use atomicAdd, which
		// will lead to write contention even if we use circular shift over M, as typically tile_j_len > M.
		// And, atomic operation itself is not efficient even without contention. Lock aquirement and release can be costly.
		// So, we parallelize over d and use one single thread to sum up the output on each dimension.
		// This implementation makes Lt=d the most performant choice.

		for (int i=tid; i<d; i+=blockDim.x) {
			const int m = i / (d/M); // example for d/m=2: tid = 0, 1, 2, ..., d-1 to m = 0, 0, 1, 1, ..., M-1, M-1
			const int k = i % (d/M); // example for d/m=2: tid = 0, 1, 2, ..., d-1 to k = 0, 1, 0, 1, ..., 0, 1
			cuscalar_t sum = cs::to_cuscalar<cuscalar_t>(0.0f);
			for (int j=tile_j_start; j<tile_j_end; ++j) {
				const int value_code = static_cast<int>(local_codes[(j-tile_j_start)*M + m]);
				sum = csc::add<cuscalar_t>()(sum, csc::mul<cuscalar_t>()(S[j-tile_j_start], value_cents[m * C * (d/M) + value_code * (d/M) + k]));
			}
			output[i] = csc::add<cuscalar_t>()(output[i], sum);
		}
		__syncthreads();

		// reduce S to sum_exp
		if (tid == 0) {
			sum_exp = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::SumReduction>(S, Lt);
		}
		__syncthreads();

        sum_exp = csc::add<cuscalar_t>()(csc::mul<cuscalar_t>()(online_scale, prev_sum_exp), sum_exp);
        prev_sum_exp = sum_exp;
	}

	const int out_offset = b * (nh*(Ns+1)*d) + h * ((Ns+1)*d) + sid * d;
	core::DeviceOps::block_op<cuscalar_t, core::Scalar::CuScalar::div>(out + out_offset, output, sum_exp, d, blockDim.x, tid);
    if (tid == 0) {
		lse[b * (nh*(Ns+1)) + h * (Ns+1) + sid] = csc::add<cuscalar_t>()(csc::log<cuscalar_t>()(sum_exp), rowmax);
	}
}


template
<
	typename cuscalar_t,
	int Ns,
	int Lt,
	int d
>
__global__ void flash_decoding_residual_kernel
(
	const cuscalar_t * __restrict__ query, // shaped (bs, nh, 1, d)
	const cuscalar_t * __restrict__ key, // shaped (bs, nh_k, Lt, d)
	const cuscalar_t * __restrict__ value, // shaped (bs, nh_k, Lt, d)
	const int r, // residual length, 0 < r <= Lt
	cuscalar_t * __restrict__ out, // shaped (bs, nh, Ns+1, d)
	cuscalar_t * __restrict__ lse, // shaped (bs, nh, Ns+1)
	const int bs,
	const int nh,
	const int nh_k
)
{
	/** 
	 * kernel launch info:
	 * grid sized (bs, nh)
	 * block sized (d, 1, 1)
	*/

	static_assert(Lt == d); // in practice Lt=d is the most performant choice
	static_assert(Lt == 64 || Lt == 128);

	const int b = blockIdx.x;
	const int h = blockIdx.y;
	const int tid = threadIdx.x;
	const int hk = h / (nh / nh_k); // GQA

	const cuscalar_t scale = cs::to_cuscalar<cuscalar_t>(1.0f / sqrt(d));
	constexpr int n_warps = d / 32;
	
	constexpr int DEN = sizeof(float) / sizeof(cuscalar_t); // scalar density, for example, float -> 1, half -> 2
	const int d_pack = d / DEN; // packed dimension

	// cuscalar_t S;
	__shared__ __align__(4) cuscalar_t S[Lt];
	__shared__ __align__(4) cuscalar_t Q[d];
	__shared__ __align__(4) cuscalar_t KV[Lt][d+DEN]; // redundant pad to avoid bank conflict on column-wise access
	__shared__ cuscalar_t rowmax; // reduce buffer
	__shared__ cuscalar_t rowsum; // reduce buffer

	// copy query to Q (last d_pack threads)
	const int query_offset = b * (nh*1*d) + h * (1*d);
	const float *query_ptr_f = reinterpret_cast<const float*>(query + query_offset);
	float *Q_ptr_f = reinterpret_cast<float*>(Q);
	if (tid >= d - d_pack && tid < d) { // last d_pack threads
		Q_ptr_f[tid - (d - d_pack)] = query_ptr_f[tid - (d - d_pack)];
	}

	// copy key to KV
	// given the matrix is squared, we can ensure coalesced memory access by vectorized load
	if (tid < d_pack) {
		for (int l=0; l<r; ++l) {
			const int key_offset = b * (nh_k*Lt*d) + hk * (Lt*d) + l * d;
			const float *key_ptr_f = reinterpret_cast<const float*>(key + key_offset);
			float *KV_ptr_f = reinterpret_cast<float*>(KV[l]);
			KV_ptr_f[tid] = key_ptr_f[tid];
		}
	}
	__syncthreads();


	// compute S = scale * Q @ KT
	// use one thread to accumulate the result for each element in S
	if (tid < r) {
		cuscalar_t sim = cs::to_cuscalar<cuscalar_t>(0.0f);
		for (int dpi = 0; dpi < d_pack; ++dpi) {
			// load 32-bit at a time
			const float q_pack = reinterpret_cast<const float*>(Q)[dpi];
			const float k_pack = reinterpret_cast<const float*>(KV[tid])[dpi];
			
			// reinterpret as cuscalar_t
			const cuscalar_t *q_cu = reinterpret_cast<const cuscalar_t*>(&q_pack);
			const cuscalar_t *k_cu = reinterpret_cast<const cuscalar_t*>(&k_pack);
			
			// unroll the process of high density scalar
			// Wish there was a super-scalar instruction set for CUDA!
			#pragma unroll
			for (int i = 0; i < DEN; ++i) {
				sim = csc::add<cuscalar_t>()(sim, csc::mul<cuscalar_t>()(q_cu[i], k_cu[i]));
			}
		}
		S[tid] = csc::mul<cuscalar_t>()(sim, scale);
	} else {
		S[tid] = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
	}
	__syncthreads();

	// copy value to KV
	// given the matrix is squared, we can ensure coalesced memory access by vectorized load
	if (tid < d_pack) {
		for (int l=0; l<r; ++l) {
			const int value_offset = b * (nh_k*Lt*d) + hk * (Lt*d) + l * d;
			const float *value_ptr_f = reinterpret_cast<const float*>(value + value_offset);
			float *KV_ptr_f = reinterpret_cast<float*>(KV[l]);
			KV_ptr_f[tid] = value_ptr_f[tid];
		}
	}

	// softmax and lse of S
	{
		// reduce max of S to rowmax
		if (tid == 0) {
			rowmax = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::MaxReduction>(S, Lt);
		}
		__syncthreads();
		
		// subtract rowmax from S, exp S
		S[tid] = csc::exp<cuscalar_t>()(csc::sub<cuscalar_t>()(S[tid], rowmax));
		__syncthreads();

		// reduce S to rowsum
		if (tid == 0) {
			rowsum = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::SumReduction>(S, Lt);
		}

		// compute lse, write to HBM
		if (tid == 0) {
			auto local_lse = csc::add<cuscalar_t>()(csc::log<cuscalar_t>()(rowsum), rowmax);
			lse[b * (nh*(Ns+1)) + h * (Ns+1) + Ns] = local_lse;
		}
		__syncthreads(); // unnecessary

		// div rowsum from S, get softmax
		S[tid] = csc::div<cuscalar_t>()(S[tid], rowsum);
		__syncthreads();
	}

	// compute output
	float *O_f = reinterpret_cast<float*>(Q); // sram reuse, just for readability
	if (tid < d_pack) {
		float o_pack;
		cuscalar_t *o_cu = reinterpret_cast<cuscalar_t*>(&o_pack);

		// init o_cu to zero
		#pragma unroll
		for (int i = 0; i < DEN; ++i) {
			o_cu[i] = cs::to_cuscalar<cuscalar_t>(0.0f);
		}

		for (int l=0; l<r; ++l) {
			// load value from KV, 32-bit at a time
			const float v_pack = reinterpret_cast<const float*>(KV[l])[tid];

			// reinterpret as cuscalar_t
			const cuscalar_t *v_cu = reinterpret_cast<const cuscalar_t*>(&v_pack);

			// unroll the process of high density scalar
			// o_cu[i] = o_cu[i] + S[l] * v_cu[i]
			const cuscalar_t s = S[l];
			#pragma unroll
			for (int i = 0; i < DEN; ++i) {
				o_cu[i] = csc::add<cuscalar_t>()(o_cu[i], csc::mul<cuscalar_t>()(s, v_cu[i]));
			}
		}
		O_f[tid] = o_pack;
	}
	__syncthreads();

	// write output to HBM
	const int out_offset = b * (nh*(Ns+1)*d) + h * ((Ns+1)*d) + Ns * d;
	float *out_ptr_f = reinterpret_cast<float*>(out + out_offset);
	if (tid < d_pack) {
		out_ptr_f[tid] = O_f[tid];
	}
}

template
<
	typename cuscalar_t,
	int Ns,
	int d
>
__global__ void flash_decoding_reduce_kernel
(
	const cuscalar_t * __restrict__ src, // shaped (bs, nh, Ns+1, d)
	const cuscalar_t * __restrict__ lse, // shaped (bs, nh, Ns+1)
	cuscalar_t * __restrict__ dst, // shaped (bs, nh, 1, d)
	const int bs,
	const int nh
)
{
	/**
	 * kernel launch info:
	 * grid sized (bs, nh)
	 * block sized (d, 1, 1)
	 * shared memory: 
	 * - L: max lse in a block, sizeof(cuscalar_t)
	 * - sum_buffer: sum of exps in a block, sizeof(cuscalar_t) * (Ns+1)
	 * - scaling_denom: sizeof(cuscalar_t)
	*/

	__shared__ cuscalar_t L; // max lse in a block
	__shared__ cuscalar_t sum_buffer[Ns+1]; // sum of exps in a block
	__shared__ cuscalar_t scaling_denom;

	const int b = blockIdx.x;
	const int h = blockIdx.y;
	const int tid = threadIdx.x;
	constexpr int n_warps = d / 32;
	
	const int src_offset = b * (nh*(Ns+1)*d) + h * ((Ns+1)*d);
	const int lse_offset = b * (nh*(Ns+1)) + h * (Ns+1);
	const int dst_offset = b * (nh*1*d) + h * (1*d) + 0 * d;

	// L = max(lse[b][h])
	if (tid == 0) {
		L = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::MaxReduction>(lse + lse_offset, Ns+1);
	}
	__syncthreads();

	for (int i=tid; i<Ns+1; i+=blockDim.x) {
		sum_buffer[i] = csc::exp<cuscalar_t>()(csc::sub<cuscalar_t>()(lse[lse_offset + i], L));
	}
	__syncthreads();

	// scaling_denom = sum(sum_buffer)
	if (tid == 0) {
		scaling_denom = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::SumReduction>(sum_buffer, Ns+1);
	}
	__syncthreads();

	// we assume dst is already filled with zeros
	for (int i=0; i<Ns+1; ++i) {
		dst[dst_offset + tid] = csc::add<cuscalar_t>()(dst[dst_offset + tid], csc::mul<cuscalar_t>()(src[src_offset + i*d + tid], csc::div<cuscalar_t>()(sum_buffer[i], scaling_denom)));
	}
}

template
<
	typename cuscalar_t, 
	typename code_t,
	int Ns, // number of splits along the nk dimension
	int Lt, // number of key_codes and value_codes inside a tile.
	int d,
	int M,
	int C
>
__global__ void flash_decoding_paged_v_kernel
(
	const cuscalar_t * __restrict__ ad_lut, // shaped (bs, nh, M, C)
	const code_t * __restrict__ key_codes, // shaped (bs, nh_k, nk, M)
	const int64_t * __restrict__ value_page_ids, // shaped (bs, nh_k, n_pages)
	const code_t * __restrict__ value_page_pool, // shaped (max_pages, M, page_size) - transposed
	const cuscalar_t * __restrict__ value_cents, // shaped (M, C, d_m)
	cuscalar_t * __restrict__ out, // shaped (bs, nh, Ns+1, d)
	cuscalar_t * __restrict__ lse, // shaped (bs, nh, Ns+1)
	const int bs,
	const int nh,
	const int nh_k,
	const int nk,
	const int Ls, // number of key_codes and value_codes inside a split.
	const int n_pages, // number of pages per batch-head
	const int page_size // tokens per page (typically 64)
)
{
	/**
	 * kernel launch info:
	 * grid sized (bs, nh, Ns)
	 * block sized (Lt, 1, 1)
	 */

	using v16_t = typename core::Vector::VectorType<16>::type; // 16 bytes

	const int b = blockIdx.x;
	const int h = blockIdx.y;
	const int sid = blockIdx.z;
	const int tid = threadIdx.x;
	const cuscalar_t scale = cs::to_cuscalar<cuscalar_t>(1.0f / sqrt(d));

    // GQA example: nh=32, nh_k=8, query with h=(0,1,2,3),(4,5,6,7) will attend to key_codes with hk=0,1
	const int hk = h / (nh / nh_k);

    __shared__ cuscalar_t S[Lt]; // regarded as [1, Lt]
	__shared__ cuscalar_t prev_rowmax;
	__shared__ cuscalar_t rowmax;
	__shared__ cuscalar_t prev_sum_exp;
	__shared__ cuscalar_t sum_exp;
	__shared__ cuscalar_t online_scale;

	__shared__ cuscalar_t output[d]; // regarded as [1, d] or [1, M, d_m]
	__shared__ __align__(16) code_t local_key_codes[Lt*M]; 
	__shared__ __align__(16) code_t local_value_codes[Lt*M]; // for transposed V storage

	if (tid == blockDim.x - 1) {
		prev_rowmax  = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
		prev_sum_exp = cs::to_cuscalar<cuscalar_t>(0.0f);
		rowmax       = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
		sum_exp      = cs::to_cuscalar<cuscalar_t>(0.0f);
	}

    core::DeviceOps::block_fill_zero<cuscalar_t>(output, d, blockDim.x, tid);
    __syncthreads();

    const int lut_offset = b * (nh*M*C) + h * (M*C);
	// in a split, loop over all tiles
	const int split_j_start = sid * Ls;
	const int split_j_end = min((sid + 1) * Ls, nk);
	code_t key_code = 0;
	int local_lut_offset = 0;
    v16_t key_code_batch;

    for (int tile_j_start=split_j_start; tile_j_start<split_j_end; tile_j_start+=Lt) {
		const int tile_j_end = min(tile_j_start + Lt, split_j_end);
		const int tile_j_len = tile_j_end - tile_j_start; // <= Lt

		// Copy key codes for attention computation
		core::DeviceOps::block_copy<code_t>(local_key_codes, key_codes + b * (nh_k*nk*M) + hk * (nk*M) + tile_j_start * M, tile_j_len * M, blockDim.x, tid);
		
		// Copy transposed value codes from pages
		// Each tile may span multiple pages, need to gather from page pool
		if (tid < tile_j_len) {
			const int global_token_idx = tile_j_start + tid;
			const int page_idx = global_token_idx / page_size;
			const int page_offset = global_token_idx % page_size;
			
			// 添加边界检查，防止页面ID越界
			if (page_idx < n_pages && page_idx >= 0) {
				// 确保page_id在有效范围内
				const int64_t page_id = value_page_ids[b * (nh_k * n_pages) + hk * n_pages + page_idx];
				
				// 验证page_id的有效性
				if (page_id >= 0 && page_id < n_pages) {
					// Copy from transposed storage: (M, page_size) -> local memory
					for (int m = 0; m < M; ++m) {
						const int pool_offset = page_id * (M * page_size) + m * page_size + page_offset;
						// 添加额外的边界检查
						if (pool_offset < n_pages * M * page_size) {
							local_value_codes[tid * M + m] = value_page_pool[pool_offset];
						} else {
							local_value_codes[tid * M + m] = 0;
						}
					}
				} else {
					// page_id无效，填充零
					for (int m = 0; m < M; ++m) {
						local_value_codes[tid * M + m] = 0;
					}
				}
			} else {
				// Fill with zeros if beyond available pages
				for (int m = 0; m < M; ++m) {
					local_value_codes[tid * M + m] = 0;
				}
			}
		}
		__syncthreads();

		// S = scale * q @ KT (same as original implementation)
		if (tid < tile_j_len) { 
			local_lut_offset = 0;
			cuscalar_t sim = cs::to_cuscalar<cuscalar_t>(0.0f);	

			for (int m = 0; m < M / sizeof(v16_t); ++m) {
				key_code_batch = reinterpret_cast<v16_t&>(local_key_codes[tid * M + m * sizeof(v16_t)]);
				local_lut_offset = C * m * sizeof(v16_t);

				code_t *key_code_batch_ptr = reinterpret_cast<code_t*>(&key_code_batch);
				for (int i = 0; i < sizeof(v16_t); ++i) {
					key_code = key_code_batch_ptr[i];
					sim = csc::add<cuscalar_t>()(sim, ad_lut[lut_offset + local_lut_offset + i * C + key_code]);
				}
			}
			S[tid] = csc::mul<cuscalar_t>()(sim, scale);
		} else {
            S[tid] = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
        }
		__syncthreads();

		// Online softmax computation (same as original)
		if (tid == 0) {
			rowmax = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::MaxReduction>(S, Lt);
			rowmax = csc::max<cuscalar_t>()(rowmax, prev_rowmax);
			online_scale = csc::exp<cuscalar_t>()(csc::sub<cuscalar_t>()(prev_rowmax, rowmax));
			prev_rowmax = rowmax;
		}
		__syncthreads();
		
		// subs S with rowmax, exp S
		S[tid] = csc::exp<cuscalar_t>()(csc::sub<cuscalar_t>()(S[tid], rowmax));

		// output = output * online_scale + S @ V
		// V is already decoded and stored in transposed format
        core::DeviceOps::block_op<cuscalar_t, core::Scalar::CuScalar::mul>(output, output, online_scale, d, blockDim.x, tid);
		__syncthreads();

		// Optimized S @ V computation using transposed V storage
		for (int i=tid; i<d; i+=blockDim.x) {
			const int m = i / (d/M); // subspace index
			const int k = i % (d/M); // dimension within subspace
			cuscalar_t sum = cs::to_cuscalar<cuscalar_t>(0.0f);
			for (int j=0; j<tile_j_len; ++j) {
				const int value_code = static_cast<int>(local_value_codes[j*M + m]);
				sum = csc::add<cuscalar_t>()(sum, csc::mul<cuscalar_t>()(S[j], value_cents[m * C * (d/M) + value_code * (d/M) + k]));
			}
			output[i] = csc::add<cuscalar_t>()(output[i], sum);
		}
		__syncthreads();

		// reduce S to sum_exp
		if (tid == 0) {
			sum_exp = core::DeviceOps::single_thread_reduce<cuscalar_t, core::Reduction::SumReduction>(S, Lt);
		}
		__syncthreads();

        sum_exp = csc::add<cuscalar_t>()(csc::mul<cuscalar_t>()(online_scale, prev_sum_exp), sum_exp);
        prev_sum_exp = sum_exp;
	}

	const int out_offset = b * (nh*(Ns+1)*d) + h * ((Ns+1)*d) + sid * d;
	core::DeviceOps::block_op<cuscalar_t, core::Scalar::CuScalar::div>(out + out_offset, output, sum_exp, d, blockDim.x, tid);
    if (tid == 0) {
		lse[b * (nh*(Ns+1)) + h * (Ns+1) + sid] = csc::add<cuscalar_t>()(csc::log<cuscalar_t>()(sum_exp), rowmax);
	}
}
