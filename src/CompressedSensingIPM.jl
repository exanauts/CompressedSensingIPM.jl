module CompressedSensingIPM

using LinearAlgebra, SparseArrays
using CUDA, MadNLP, MadNLPGPU
using FFTW
using Krylov
using NLPModels

export FFTNLPModel, FFTKKTSystem, FFTParameters

include("fft_utils.jl")
include("fft_model.jl")

include("mapping_cpu.jl")
# include("mapping_cpu_kernels.jl")
include("mapping_gpu.jl")

include("kkt.jl")
# include("kkt_pdco.jl")

end
