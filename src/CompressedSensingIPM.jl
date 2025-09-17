module CompressedSensingIPM

using LinearAlgebra, SparseArrays
using CUDA, MadNLP, MadNLPGPU
using AMDGPU
using FFTW
using Krylov
using NLPModels
using LinearOperators

export FFTNLPModel, FFTKKTSystem
export GondzioNLPModel, GondzioKKTSystem
export FFTParameters, FFTOperator
export CompressedSensingADMM

include("fft_utils.jl")
include("fft_model.jl")
include("gondzio_model.jl")

include("mapping_cpu.jl")
# include("mapping_cpu_kernels.jl")

include("mapping_cuda.jl")
include("mapping_rocm.jl")
# include("mapping_gpu_kernels.jl")

include("fft_kkt.jl")
include("gondzio_kkt.jl")
include("admm.jl")

function ipm_solve!(solver::MadNLP.MadNLPSolver)
    MadNLP.print_init(solver)
    MadNLP.initialize!(solver)
    MadNLP.regular!(solver)
    return MadNLP.MadNLPExecutionStats(solver)
end

end
