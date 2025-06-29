module CompressedSensingIPM

using LinearAlgebra, SparseArrays
using CUDA, MadNLP, MadNLPGPU
using AMDGPU
using FFTW
using Krylov
using NLPModels

export FFTNLPModel, FFTKKTSystem, FFTParameters

include("fft_utils.jl")
include("fft_model.jl")

include("mapping_cpu.jl")
# include("mapping_cpu_kernels.jl")

include("gpu_utils.jl")
include("mapping_cuda.jl")
include("mapping_rocm.jl")
# include("mapping_gpu_kernels.jl")

include("kkt.jl")

end
