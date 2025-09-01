from setuptools import setup
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension # 构建的Kernel调用 会依赖于Torch
from itertools import product


################ META PROGRAMMING ################

# float_list = ['f32', 'f16']
float_list = ['f16']
uint_list = ['u8']
Ns_list = [2, 4, 8, 16, 32]
d_list = [64, 128]
M_list = [32, 64]
C_list = [256]

# generate cuda file
# cuda_template = "flash_decoding_pq.template.cu"
# cuda_outfile  = "flash_decoding_pq.cu"
cuda_template = "Interface.template.cu" # 模板文件<手动撰写>
cuda_outfile  = "Interface.cu" # 输出文件 <代码生成>
with open(cuda_template, "r") as fin, open(cuda_outfile, "w") as fout:
    for line in fin:
        fout.write(line)

    for f, u, Ns, d, M, C in product(float_list, uint_list, Ns_list, d_list, M_list, C_list):
        Lt = d # Best practice: Lt = d
        fout.write(f"register_flash_decoding_allocated_buffer({f}, {u}, {Ns}, {Lt}, {d}, {M}, {C});\n")
        fout.write(f"register_flash_decoding_paged_v({f}, {u}, {Ns}, {Lt}, {d}, {M}, {C});\n")

# generate bindings.cpp
cpp_template = "bindings.template.cpp"# 模板文件<手动撰写>
cpp_outfile  = "bindings.cpp"# 输出文件 <代码生成>
with open(cpp_template, "r") as fin, open(cpp_outfile, "w") as fout:
    for line in fin:
        fout.write(line)

    for f, u, Ns, d, M, C in product(float_list, uint_list, Ns_list, d_list, M_list, C_list):
        Lt = d # Best practice: Lt = d
        fout.write(f"declare_flash_decoding_allocated_buffer({f}, {u}, {Ns}, {Lt}, {d}, {M}, {C});\n")
        fout.write(f"declare_flash_decoding_paged_v({f}, {u}, {Ns}, {Lt}, {d}, {M}, {C});\n")

    fout.write(r"PYBIND11_MODULE(bindings, m) {" + "\n")
    for f, u, Ns, d, M, C in product(float_list, uint_list, Ns_list, d_list, M_list, C_list):
        Lt = d # Best practice: Lt = d
        fout.write(f"    m.def(\"flash_decoding_allocated_buffer_{f}{u}_Ns{Ns}Lt{Lt}d{d}M{M}C{C}\", &flash_decoding_allocated_buffer_{f}{u}_Ns{Ns}Lt{Lt}d{d}M{M}C{C});\n")
        fout.write(f"    m.def(\"flash_decoding_paged_v_{f}{u}_Ns{Ns}Lt{Lt}d{d}M{M}C{C}\", &flash_decoding_paged_v_{f}{u}_Ns{Ns}Lt{Lt}d{d}M{M}C{C});\n")
    fout.write(r"}" + "\n")

##################################################

setup(
    name='bindings',
    ext_modules=[
        CUDAExtension(
            name='bindings',
            sources=[cuda_outfile, cpp_outfile],
            include_dirs=['./', './core'],
            extra_compile_args={
                'cxx': ['-g'],
                # 'nvcc': ['-O0', '-lineinfo', '--use_fast_math', '--threads=8'],
                'nvcc': [
                    "-Xcompiler", 
                    "-fPIC", 
                    "-shared", 
                    "-arch=sm_80", 
                    # "-G", "-g", '-O0',
                    "-O3",
                    # '-I"/root/NVTX/c/include"',
                    '--threads=8',
                    '--use_fast_math',
                    '-ftz=true',
                ]
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
        'pybind11'
    ]
)