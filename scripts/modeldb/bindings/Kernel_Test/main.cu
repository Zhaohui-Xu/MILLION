#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <chrono>
#include <torch/torch.h>
#include <nvtx3/nvtx3.hpp>

// 包含你的Kernel定义和所有必要的模板实例化
// 我们假设 Interface.cu 包含了所有 register_... 的调用
#include "../Interface.cu"

// 定义一个函数指针类型，用于存储对特定Kernel的引用
using KernelFunc = std::function<torch::Tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, int, const torch::Tensor&, const torch::Tensor&
)>;

// 用于存储已注册的Kernel函数
std::map<std::string, KernelFunc> kernel_registry;

// 宏，用于简化在main函数中注册Kernel的过程
#define REGISTER_KERNEL(T, code_alias, Nsv, Ltv, dv, Mv, Cv) \
    { \
        std::string name_std = "flash_decoding_allocated_buffer_" #T #code_alias "_Ns" #Nsv "Lt" #Ltv "d" #dv "M" #Mv "C" #Cv; \
        kernel_registry[name_std] = flash_decoding_allocated_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv; \
        std::string name_paged = "flash_decoding_allocated_paged_buffer_" #T #code_alias "_Ns" #Nsv "Lt" #Ltv "d" #dv "M" #Mv "C" #Cv; \
        kernel_registry[name_paged] = flash_decoding_allocated_paged_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv; \
        std::string name_paged_split_qkv = "flash_decoding_allocated_paged_split_qkv_buffer_" #T #code_alias "_Ns" #Nsv "Lt" #Ltv "d" #dv "M" #Mv "C" #Cv; \
        kernel_registry[name_paged_split_qkv] = flash_decoding_allocated_paged_split_qkv_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv; \
        std::string name_paged_lastblock_sync = "flash_decoding_allocated_paged_lastblock_sync_buffer_" #T #code_alias "_Ns" #Nsv "Lt" #Ltv "d" #dv "M" #Mv "C" #Cv; \
        kernel_registry[name_paged_lastblock_sync] = flash_decoding_allocated_paged_lastblock_sync_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M##Mv##C##Cv; \
    }

// C++版本的 sa_decode_4d，用于计算参考结果
torch::Tensor sa_decode_4d(const torch::Tensor& codes, const torch::Tensor& centroids) {
    // codes: (bs, nh, T, M)
    // centroids: (M, C, d_m)
    int bs = codes.size(0);
    int nh = codes.size(1);
    int T = codes.size(2);
    int M = codes.size(3);
    int C = centroids.size(1);
    int d_m = centroids.size(2);

    auto decoded = torch::empty({bs, nh, T, M, d_m}, centroids.options());
    auto codes_long = codes.to(torch::kLong);

    for (int m = 0; m < M; ++m) {
        auto codes_m = codes_long.slice(3, m, m + 1).squeeze(-1); // (bs, nh, T)
        auto cents_m = centroids.slice(0, m, m + 1).squeeze(0);  // (C, d_m)
        decoded.slice(3, m, m + 1).copy_(cents_m.index_select(0, codes_m.flatten()).view({bs, nh, T, 1, d_m}));
    }
    return decoded.view({bs, nh, T, M * d_m});
}


void run_test(int Ns, int d, int M, int C, int T, int r, int iters) {
    // --- 1. 获取内核 ---
    std::string kernel_name_base = "flash_decoding_allocated_buffer_f16u8_Ns" + std::to_string(Ns) + "Lt" + std::to_string(d) + "d" + std::to_string(d) + "M" + std::to_string(M) + "C" + std::to_string(C);
    std::string kernel_name_test = "flash_decoding_allocated_paged_buffer_f16u8_Ns" + std::to_string(Ns) + "Lt" + std::to_string(d) + "d" + std::to_string(d) + "M" + std::to_string(M) + "C" + std::to_string(C);
    std::string kernel_name_split_qkv = "flash_decoding_allocated_paged_split_qkv_buffer_f16u8_Ns" + std::to_string(Ns) + "Lt" + std::to_string(d) + "d" + std::to_string(d) + "M" + std::to_string(M) + "C" + std::to_string(C);
    std::string name_paged_lastblock_sync = "flash_decoding_allocated_paged_lastblock_sync_buffer_f16u8_Ns" + std::to_string(Ns) + "Lt" + std::to_string(d) + "d" + std::to_string(d) + "M" + std::to_string(M) + "C" + std::to_string(C);

    if (kernel_registry.find(kernel_name_base) == kernel_registry.end()) {
        std::cerr << "Error: Baseline kernel " << kernel_name_base << " not found in registry." << std::endl;
        return;
    }
    if (kernel_registry.find(kernel_name_test) == kernel_registry.end()) {
        std::cerr << "Error: Test kernel " << kernel_name_test << " not found in registry." << std::endl;
        return;
    }
    if (kernel_registry.find(kernel_name_split_qkv) == kernel_registry.end()) {
        std::cerr << "Error: Test kernel " << kernel_name_split_qkv << " not found in registry." << std::endl;
        return;
    }
    if (kernel_registry.find(name_paged_lastblock_sync) == kernel_registry.end()) {
        std::cerr << "Error: Test kernel " << name_paged_lastblock_sync << " not found in registry." << std::endl;
        return;
    }

    KernelFunc kernel_baseline = kernel_registry[kernel_name_base];
    KernelFunc kernel_test = kernel_registry[kernel_name_test];
    KernelFunc kernel_split_qkv = kernel_registry[kernel_name_split_qkv];
    KernelFunc kernel_lastblock_sync = kernel_registry[name_paged_lastblock_sync];
    std::cout << "Successfully found kernels." << std::endl;

    // --- 2. 初始化数据 ---
    const int bs = 1, nh = 32, Lt = d;
    auto options_float = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto options_uint = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

    torch::Tensor query = torch::randn({bs, nh, 1, d}, options_float);
    torch::Tensor key_codes = torch::randint(0, C, {bs, nh, T, M}, options_uint);
    torch::Tensor value_codes = torch::randint(0, C, {bs, nh, T, M}, options_uint);
    torch::Tensor key_cents = torch::randn({M, C, d / M}, options_float);
    torch::Tensor value_cents = torch::randn({M, C, d / M}, options_float);
    torch::Tensor key_residuals = torch::randn({bs, nh, Lt, d}, options_float);
    torch::Tensor value_residuals = torch::randn({bs, nh, Lt, d}, options_float);

    torch::Tensor partial_out = torch::empty({bs, nh, Ns + 1, d}, options_float);
    torch::Tensor partial_lse = torch::empty({bs, nh, Ns + 1}, options_float);
    std::cout << "Tensors initialized." << std::endl;

    // --- 3. 计算参考结果 ---
    nvtx3::scoped_range ref_range{"Reference Calculation"};
    auto K_hat = sa_decode_4d(key_codes, key_cents);
    auto V_hat = sa_decode_4d(value_codes, value_cents);
    auto K_full = torch::cat({K_hat, key_residuals.slice(2, 0, r)}, 2);
    auto V_full = torch::cat({V_hat, value_residuals.slice(2, 0, r)}, 2);
    auto out_ref = at::scaled_dot_product_attention(query, K_full, V_full, {}, {}, true);
    
    std::cout << "Reference output calculated." << std::endl;
    torch::Tensor value_codes_transposed = value_codes.transpose(2, 3).contiguous();

    nvtx3::range_handle h = nvtx3::start_range_in<nvtx3::domain::global>("Kernel_Test", nvtx3::rgb{127,255,0});
    // {
    //     auto out_base = kernel_baseline(query, key_codes, value_codes, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
    //     cudaDeviceSynchronize();
    // }
    // {
    //     auto out_test = kernel_test(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
    //     cudaDeviceSynchronize();
    // }
    // {
    //     auto out_test_qkv = kernel_split_qkv(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
    //     cudaDeviceSynchronize();
    // }
    
    auto out_test_lastblock_sync = kernel_lastblock_sync(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
    
    
    nvtx3::end_range_in<nvtx3::domain::global>(h); // Ends the range




    // --- 4. 测试 Baseline Kernel ---
    {
        nvtx3::scoped_range baseline_range{"Baseline_Kernel"};
        std::cout << "\n--- Evaluating Baseline Kernel ---" << std::endl;
        auto out_base = kernel_baseline(query, key_codes, value_codes, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        
        auto mae = (out_base - out_ref).abs().mean().item<float>();
        auto mxe = (out_base - out_ref).abs().max().item<float>();
        printf("shape: (%ld, %ld, %ld, %ld), MAE: %.4e, MaxAbsErr: %.4e\n", out_base.size(0), out_base.size(1), out_base.size(2), out_base.size(3), mae, mxe);

        for(int i=0; i<10; ++i) kernel_baseline(query, key_codes, value_codes, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();
        
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iters; ++i) kernel_baseline(query, key_codes, value_codes, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("kernel_Baseline: avg time per call: %.3f ms (Ns=%d, d=%d, M=%d, C=%d, T=%d, r=%d)\n", elapsed.count() / iters, Ns, d, M, C, T, r);
    }

    // --- 5. 测试 Test (Paged) Kernel ---
    {
        nvtx3::scoped_range test_range{"Test_Kernel"};
        std::cout << "\n--- Evaluating Test Kernel ---" << std::endl;
        auto out_test = kernel_test(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);

        auto mae = (out_test - out_ref).abs().mean().item<float>();
        auto mxe = (out_test - out_ref).abs().max().item<float>();
        printf("shape: (%ld, %ld, %ld, %ld), MAE: %.4e, MaxAbsErr: %.4e\n", out_test.size(0), out_test.size(1), out_test.size(2), out_test.size(3), mae, mxe);

        for(int i=0; i<10; ++i) kernel_test(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iters; ++i) kernel_test(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("kernel_test: avg time per call: %.3f ms (Ns=%d, d=%d, M=%d, C=%d, T=%d, r=%d)\n", elapsed.count() / iters, Ns, d, M, C, T, r);
    }
    
    // --- 5. 测试 Test (Paged_Split_qkv) Kernel ---
    {
        nvtx3::scoped_range test_range{"kernel_split_qkv"};
        std::cout << "\n--- Evaluating kernel_split_qkv Kernel ---" << std::endl;
        auto out_test_qkv = kernel_split_qkv(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);

        auto mae = (out_test_qkv - out_ref).abs().mean().item<float>();
        auto mxe = (out_test_qkv - out_ref).abs().max().item<float>();
        printf("shape: (%ld, %ld, %ld, %ld), MAE: %.4e, MaxAbsErr: %.4e\n", out_test_qkv.size(0), out_test_qkv.size(1), out_test_qkv.size(2), out_test_qkv.size(3), mae, mxe);

        for(int i=0; i<10; ++i) kernel_split_qkv(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iters; ++i) kernel_split_qkv(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("kernel_test: avg time per call: %.3f ms (Ns=%d, d=%d, M=%d, C=%d, T=%d, r=%d)\n", elapsed.count() / iters, Ns, d, M, C, T, r);
    }

    // --- 5. 测试 Test (Paged_Split_qkv) Kernel ---
    {
        nvtx3::scoped_range test_range{"kernel_lastblock_sync"};
        std::cout << "\n--- Evaluating kernel_lastblock_sync Kernel ---" << std::endl;
        auto out_test_lastblock_sync = kernel_lastblock_sync(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);

        auto mae = (out_test_lastblock_sync - out_ref).abs().mean().item<float>();
        auto mxe = (out_test_lastblock_sync - out_ref).abs().max().item<float>();
        printf("shape: (%ld, %ld, %ld, %ld), MAE: %.4e, MaxAbsErr: %.4e\n", out_test_lastblock_sync.size(0), out_test_lastblock_sync.size(1), out_test_lastblock_sync.size(2), out_test_lastblock_sync.size(3), mae, mxe);

        for(int i=0; i<10; ++i) kernel_lastblock_sync(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iters; ++i) kernel_lastblock_sync(query, key_codes, value_codes_transposed, key_cents, value_cents, key_residuals, value_residuals, r, partial_out, partial_lse);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("kernel_test: avg time per call: %.3f ms (Ns=%d, d=%d, M=%d, C=%d, T=%d, r=%d)\n", elapsed.count() / iters, Ns, d, M, C, T, r);
    }
}

int main(int argc, char* argv[]) {
    // --- 注册所有编译时实例化的Kernel ---
    // 这个过程应该与 setup.py 中的 product 循环保持一致
 // d = 64
    REGISTER_KERNEL(f16, u8, 2, 64, 64, 16, 128);
    REGISTER_KERNEL(f16, u8, 2, 64, 64, 16, 256);
    REGISTER_KERNEL(f16, u8, 2, 64, 64, 32, 128);
    REGISTER_KERNEL(f16, u8, 2, 64, 64, 32, 256);
    REGISTER_KERNEL(f16, u8, 2, 64, 64, 64, 128);
    REGISTER_KERNEL(f16, u8, 2, 64, 64, 64, 256);
    REGISTER_KERNEL(f16, u8, 4, 64, 64, 16, 128);
    REGISTER_KERNEL(f16, u8, 4, 64, 64, 16, 256);
    REGISTER_KERNEL(f16, u8, 4, 64, 64, 32, 128);
    REGISTER_KERNEL(f16, u8, 4, 64, 64, 32, 256);
    REGISTER_KERNEL(f16, u8, 4, 64, 64, 64, 128);
    REGISTER_KERNEL(f16, u8, 4, 64, 64, 64, 256);
    REGISTER_KERNEL(f16, u8, 8, 64, 64, 16, 128);
    REGISTER_KERNEL(f16, u8, 8, 64, 64, 16, 256);
    REGISTER_KERNEL(f16, u8, 8, 64, 64, 32, 128);
    REGISTER_KERNEL(f16, u8, 8, 64, 64, 32, 256);
    REGISTER_KERNEL(f16, u8, 8, 64, 64, 64, 128);
    REGISTER_KERNEL(f16, u8, 8, 64, 64, 64, 256);
    REGISTER_KERNEL(f16, u8, 16, 64, 64, 16, 128);
    REGISTER_KERNEL(f16, u8, 16, 64, 64, 16, 256);
    REGISTER_KERNEL(f16, u8, 16, 64, 64, 32, 128);
    REGISTER_KERNEL(f16, u8, 16, 64, 64, 32, 256);
    REGISTER_KERNEL(f16, u8, 16, 64, 64, 64, 128);
    REGISTER_KERNEL(f16, u8, 16, 64, 64, 64, 256);
    REGISTER_KERNEL(f16, u8, 32, 64, 64, 16, 128);
    REGISTER_KERNEL(f16, u8, 32, 64, 64, 16, 256);
    REGISTER_KERNEL(f16, u8, 32, 64, 64, 32, 128);
    REGISTER_KERNEL(f16, u8, 32, 64, 64, 32, 256);
    REGISTER_KERNEL(f16, u8, 32, 64, 64, 64, 128);
    REGISTER_KERNEL(f16, u8, 32, 64, 64, 64, 256);

    // d = 128
    REGISTER_KERNEL(f16, u8, 2, 128, 128, 16, 128);
    REGISTER_KERNEL(f16, u8, 2, 128, 128, 16, 256);
    REGISTER_KERNEL(f16, u8, 2, 128, 128, 32, 128);
    REGISTER_KERNEL(f16, u8, 2, 128, 128, 32, 256);
    REGISTER_KERNEL(f16, u8, 2, 128, 128, 64, 128);
    REGISTER_KERNEL(f16, u8, 2, 128, 128, 64, 256);
    REGISTER_KERNEL(f16, u8, 4, 128, 128, 16, 128);
    REGISTER_KERNEL(f16, u8, 4, 128, 128, 16, 256);
    REGISTER_KERNEL(f16, u8, 4, 128, 128, 32, 128);
    REGISTER_KERNEL(f16, u8, 4, 128, 128, 32, 256);
    REGISTER_KERNEL(f16, u8, 4, 128, 128, 64, 128);
    REGISTER_KERNEL(f16, u8, 4, 128, 128, 64, 256);
    REGISTER_KERNEL(f16, u8, 8, 128, 128, 16, 128);
    REGISTER_KERNEL(f16, u8, 8, 128, 128, 16, 256);
    REGISTER_KERNEL(f16, u8, 8, 128, 128, 32, 128);
    REGISTER_KERNEL(f16, u8, 8, 128, 128, 32, 256);
    REGISTER_KERNEL(f16, u8, 8, 128, 128, 64, 128);
    REGISTER_KERNEL(f16, u8, 8, 128, 128, 64, 256);
    REGISTER_KERNEL(f16, u8, 16, 128, 128, 16, 128);
    REGISTER_KERNEL(f16, u8, 16, 128, 128, 16, 256);
    REGISTER_KERNEL(f16, u8, 16, 128, 128, 32, 128);
    REGISTER_KERNEL(f16, u8, 16, 128, 128, 32, 256);
    REGISTER_KERNEL(f16, u8, 16, 128, 128, 64, 128);
    REGISTER_KERNEL(f16, u8, 16, 128, 128, 64, 256);
    REGISTER_KERNEL(f16, u8, 32, 128, 128, 16, 128);
    REGISTER_KERNEL(f16, u8, 32, 128, 128, 16, 256);
    REGISTER_KERNEL(f16, u8, 32, 128, 128, 32, 128);
    REGISTER_KERNEL(f16, u8, 32, 128, 128, 32, 256);
    REGISTER_KERNEL(f16, u8, 32, 128, 128, 64, 128);
    REGISTER_KERNEL(f16, u8, 32, 128, 128, 64, 256);

    // --- 解析命令行参数 ---
    std::map<std::string, int> args;
    args["--Ns"] = 16;
    args["--d"] = 128;
    args["--M"] = 64;
    args["--C"] = 256;
    args["--T"] = 1000;
    args["--r"] = 17;
    args["--iters"] = 100;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            std::string key = argv[i];
            if (args.count(key)) {
                args[key] = std::stoi(argv[i + 1]);
            }
        }
    }

    std::cout << "Running with parameters:" << std::endl;
    for(auto const& [key, val] : args) {
        std::cout << "  " << key << ": " << val << std::endl;
    }

    try {
        run_test(args["--Ns"], args["--d"], args["--M"], args["--C"], args["--T"], args["--r"], args["--iters"]);
    } catch (const c10::Error& e) {
        std::cerr << "Caught a c10::Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Caught a std::exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}