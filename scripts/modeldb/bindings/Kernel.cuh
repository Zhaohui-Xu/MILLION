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

		// 载入 value 的 codes 到共享内存（与 K 同步 tile）
		// output = output * online_scale + S @ V, we decode V on-the-fly
        core::DeviceOps::block_op<cuscalar_t, core::Scalar::CuScalar::mul>(output, output, online_scale, d, blockDim.x, tid); // output = output * online_scale
		// 载入 value 的 codes 到共享内存（与 K 同步 tile）<逐Token>
		core::DeviceOps::block_copy<code_t>(local_codes, value_codes + b * (nh_k*nk*M) + hk * (nk*M) + tile_j_start * M, tile_j_len * M, blockDim.x, tid);
		__syncthreads();

		// We have threads tid = 0, 1, 2, ..., Lt-1 and add S (tile_j_len) @ V (tile_j_len, M, d/M) to output (M, d/M)
		// If we parallelize over tile_j_len (which is a natural choice by blockDim.x), we have to use atomicAdd, which
		// will lead to write contention even if we use circular shift over M, as typically tile_j_len > M.
		// And, atomic operation itself is not efficient even without contention. Lock aquirement and release can be costly.
		// So, we parallelize over d and use one single thread to sum up the output on each dimension.
		// This implementation makes Lt=d the most performant choice.
		
		// 按维度 i 还原子空间与子维度，并用 code 查质心向量分量
		for (int i=tid; i<d; i+=blockDim.x) {
			const int m = i / (d/M); // example for d/m=2: tid = 0, 1, 2, ..., d-1 to m = 0, 0, 1, 1, ..., M-1, M-1 ;子空间索引
			const int k = i % (d/M); // example for d/m=2: tid = 0, 1, 2, ..., d-1 to k = 0, 1, 0, 1, ..., 0, 1 ;子空间内维度索引
			cuscalar_t sum = cs::to_cuscalar<cuscalar_t>(0.0f);
			for (int j=tile_j_start; j<tile_j_end; ++j) {
				const int value_code = static_cast<int>(local_codes[(j-tile_j_start)*M + m]);
				// value_cents [M, C, d/M]
				sum = csc::add<cuscalar_t>()(sum, csc::mul<cuscalar_t>()(
					S[j-tile_j_start], value_cents[m * C * (d/M) + value_code * (d/M) + k])); // 按列访问
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
	typename code_t,
	int Ns, // number of splits along the nk dimension
	int Lt, // number of key_codes and value_codes inside a tile.
	int d,
	int M,
	int C
>
__global__ void flash_decoding_paged_V_split_kernel
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
	// constexpr int n_warps = d / 32;

    // 定义必要的常量
    constexpr int d_per_m = d / M;
    constexpr int PAGE_SIZE = 64;  // 建议使用较小的page size，而不是d
    constexpr int num_pages = (C + PAGE_SIZE - 1) / PAGE_SIZE;

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

	__shared__ cuscalar_t T[M*C];// 用于V的计算
	// __shared__ __align__(16) cuscalar_t T[M*C];// 用于V的计算

	
	if (tid == blockDim.x - 1) {
		prev_rowmax  = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
		prev_sum_exp = cs::to_cuscalar<cuscalar_t>(0.0f);
		rowmax       = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
		sum_exp      = cs::to_cuscalar<cuscalar_t>(0.0f);
	}

    core::DeviceOps::block_fill_zero<cuscalar_t>(output, d, blockDim.x, tid);
	core::DeviceOps::block_fill_zero<cuscalar_t>(T, M*C, blockDim.x, tid);
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

		// 载入 value 的 codes 到共享内存（与 K 同步 tile）
        // output = output * online_scale + S @ V, we decode V on-the-fly
        core::DeviceOps::block_op<cuscalar_t, core::Scalar::CuScalar::mul>(output, output, online_scale, d, blockDim.x, tid);
		// value_codes shaped (bs, nh_k, nk, M) , 以M为基本搬运块 => tile_j_len表示当前处理的序列长度 通常为lt(tile大小)
		// core::DeviceOps::block_copy<code_t>(local_codes, value_codes + b * (nh_k*nk*M) + hk * (nk*M) + tile_j_start * M, tile_j_len * M, blockDim.x, tid);
		// 如果物理上转置，value_codes_transpose shape is (bs, nh_k, M, nk) => 搬运不用改 数据量是一样的 后续当value_codes_transpose[M,n1]用
		core::DeviceOps::block_copy<code_t>(local_codes, value_codes + b * (nh_k*nk*M) + hk * (nk*M) + tile_j_start * M, tile_j_len * M, blockDim.x, tid);
		__syncthreads();

		// We have threads tid = 0, 1, 2, ..., Lt-1 and add S (tile_j_len) @ V (tile_j_len, M, d/M) to output (M, d/M)
		// If we parallelize over tile_j_len (which is a natural choice by blockDim.x), we have to use atomicAdd, which
		// will lead to write contention even if we use circular shift over M, as typically tile_j_len > M.
		// And, atomic operation itself is not efficient even without contention. Lock aquirement and release can be costly.
		// So, we parallelize over d and use one single thread to sum up the output on each dimension.
		// This implementation makes Lt=d the most performant choice.
		
		
		// // // 按维度 i 还原子空间与子维度，并用 code 查质心向量分量
		// for (int i=tid; i<d; i+=blockDim.x) {
		// 	cuscalar_t sum = cs::to_cuscalar<cuscalar_t>(0.0f);
		// 	for (int j=tile_j_start; j<tile_j_end; ++j) {
		// 		const int value_code = static_cast<int>(local_codes[(j-tile_j_start)*M + m]);// 列访问 Delta_V[i][m]
		// 		sum = csc::add<cuscalar_t>()(sum, csc::mul<cuscalar_t>()(S[j-tile_j_start], value_cents[m * C * (d/M) + value_code * (d/M) + k]));
		// 	}
		// 	output[i] = csc::add<cuscalar_t>()(output[i], sum);
		// }
		// __syncthreads();
		// value_codes, // shaped (bs, nh_k, nk, M)
		// value_cents, // shaped (M, C, d_m)
		
		constexpr int WARP_SIZE = M;
		// 直接将 warp_id 映射到 m，确保一个 warp 处理一个 M
		const int m = tid / WARP_SIZE;// 理解为全局的并行度？
		const int lane_id = tid % WARP_SIZE;// 同Warp内的线程 会影响Shared Memory的Bank冲突
	
		// 因为每个 warp 只处理一个M，所以无需循环
		// Warp 内的线程在 j 维度上并行
		for (int j = tile_j_start + lane_id; j < tile_j_end; j += WARP_SIZE) {
			// 当数据为u8时,Shared Memory中32个Bank,一行排布32个Word(4byte) => 128个byte为一行 128个u8为一行
			// 如果 M=64，那么一个物理Warp（固定32个线程，同时执行动作）只构成了半个逻辑Warp。并发访问的是32个线程，而不是 M 个线程。
			// 一个物理Warp(32线程)并行访问 local_codes 的同一列 m，但行号 j 不同。<冲突是考虑的物理Warp>
			// 相邻两个线程的地址差异为M byte,之间相差多少个Word = M / 4 => 跨(M/4)%32个Bank
			// Bank_Stride == 0 => 冲突路数为 32/(一个Bank能服务的线程数)
			// Bank_Stride != 0 => 冲突路数为 gcd(32,Bank_Stride)
			// 当M=64时，跨16个Bank,冲突路数为16
			// 当M=32时，跨8个Bank,冲突路数为8
			const int value_code = static_cast<int>(local_codes[(j - tile_j_start) * M + m]);// Warp按行读取
			// 直接写入，无需原子操作，因为每个 warp 独立处理其 m
			T[m * C + value_code] = csc::add<cuscalar_t>()(T[m * C + value_code], S[j - tile_j_start]);
		}// 这个for循环主要的性能瓶颈在对Shared Memory的访问上
		__syncthreads();
		
		// 每个 warp 负责计算其对应的 m 的部分输出
		for (int i = m * d_per_m + lane_id; i < (m + 1) * d_per_m; i += WARP_SIZE) {
			const int k = i % d_per_m;              // m内的位置
			cuscalar_t sum = cs::to_cuscalar<cuscalar_t>(0.0f);
			// #pragma unroll
			for (int c = 0; c < C; ++c) {
				// 使用第m个warp计算的中间结果
				sum = csc::add<cuscalar_t>()(sum, csc::mul<cuscalar_t>()(T[m * C + c], value_cents[m * C * d_per_m + c * d_per_m + k]));
			}
			output[i] = csc::add<cuscalar_t>()(output[i], sum);
		}

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
