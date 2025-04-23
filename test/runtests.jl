using CompressedSensingIPM
import CompressedSensingIPM: M_perpt_M_perp_vec, M_perp_tz, M_perp_beta, DFT_to_beta

using LinearAlgebra, Random, Test
using MadNLP, MadNLPGPU, CUDA
using FFTW

Random.seed!(1)

dim1 = true
dim2 = true
dim3 = true

function ipm_solve!(solver::MadNLP.MadNLPSolver)
    MadNLP.print_init(solver)
    MadNLP.initialize!(solver)
    MadNLP.regular!(solver)
    return MadNLP.MadNLPExecutionStats(solver)
end

# include("punching_centering.jl")
include("punching_centering_v2.jl")

include("fft_wei.jl")

dim1 && include("fft_example_1D.jl")
dim2 && include("fft_example_2D.jl")
dim3 && include("fft_example_3D.jl")

include("unit_tests.jl")
