#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <torch/extension.h>

using u8 = uint8_t;
using u16 = uint16_t;
using f16 = at::Half;
using f32 = float;

#define declare_flash_decoding_allocated_paged_split_qkv_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
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
	)
#define declare_flash_decoding_allocated_paged_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
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
	)

#define declare_flash_decoding_allocated_buffer(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
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
	)
declare_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 64, 64, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 2, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 2, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 2, 128, 128, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 64, 64, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 4, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 4, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 4, 128, 128, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 64, 64, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 8, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 8, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 8, 128, 128, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 64, 64, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 16, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 16, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 16, 128, 128, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 64, 64, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 64, 64, 64, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 16, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 16, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 16, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 16, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 32, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 32, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 32, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 32, 256);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 64, 128);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 64, 128);
declare_flash_decoding_allocated_buffer(f16, u8, 32, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_buffer(f16, u8, 32, 128, 128, 64, 256);
declare_flash_decoding_allocated_paged_split_qkv_buffer(f16, u8, 32, 128, 128, 64, 256);
PYBIND11_MODULE(bindings, m) {
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M16C128", &flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M16C256", &flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M32C128", &flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M32C256", &flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M64C128", &flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M64C256", &flash_decoding_allocated_buffer_f16u8_Ns2Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt64d64M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M16C128", &flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M16C256", &flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M32C128", &flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M32C256", &flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M64C128", &flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M64C256", &flash_decoding_allocated_buffer_f16u8_Ns2Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns2Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns2Lt128d128M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M16C128", &flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M16C256", &flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M32C128", &flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M32C256", &flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M64C128", &flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M64C256", &flash_decoding_allocated_buffer_f16u8_Ns4Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt64d64M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M16C128", &flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M16C256", &flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M32C128", &flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M32C256", &flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M64C128", &flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M64C256", &flash_decoding_allocated_buffer_f16u8_Ns4Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns4Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns4Lt128d128M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M16C128", &flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M16C256", &flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M32C128", &flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M32C256", &flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M64C128", &flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M64C256", &flash_decoding_allocated_buffer_f16u8_Ns8Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt64d64M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M16C128", &flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M16C256", &flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M32C128", &flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M32C256", &flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M64C128", &flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M64C256", &flash_decoding_allocated_buffer_f16u8_Ns8Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns8Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns8Lt128d128M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M16C128", &flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M16C256", &flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M32C128", &flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M32C256", &flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M64C128", &flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M64C256", &flash_decoding_allocated_buffer_f16u8_Ns16Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt64d64M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M16C128", &flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M16C256", &flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M32C128", &flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M32C256", &flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M64C128", &flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M64C256", &flash_decoding_allocated_buffer_f16u8_Ns16Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns16Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns16Lt128d128M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M16C128", &flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M16C256", &flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M32C128", &flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M32C256", &flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M64C128", &flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M64C256", &flash_decoding_allocated_buffer_f16u8_Ns32Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt64d64M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt64d64M64C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M16C128", &flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M16C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M16C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M16C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M16C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M16C256", &flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M16C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M16C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M16C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M16C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M32C128", &flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M32C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M32C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M32C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M32C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M32C256", &flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M32C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M32C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M32C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M32C256);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M64C128", &flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M64C128", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M64C128);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M64C128", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M64C128);
    m.def("flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M64C256", &flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M64C256", &flash_decoding_allocated_paged_buffer_f16u8_Ns32Lt128d128M64C256);
    m.def("flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M64C256", &flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns32Lt128d128M64C256);
}
