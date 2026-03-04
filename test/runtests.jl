using CompressedSensingIPM
import CompressedSensingIPM: M_perpt_M_perp_vec, M_perpt_z, M_perp_beta, DFT_to_beta

using LinearAlgebra, Random, Test
using MadNLP, MadNLPGPU, CUDA, AMDGPU
using FFTW

Random.seed!(1)

dim1 = true
dim2 = true
dim3 = true

include("punching_centering.jl")
include("fft_wei.jl")

dim1 && include("ipm_example_1D.jl")
dim2 && include("ipm_example_2D.jl")
dim3 && include("ipm_example_3D.jl")
include("unit_tests_ipm.jl")

dim1 && include("admm_example_1D.jl")
dim2 && include("admm_example_2D.jl")
dim3 && include("admm_example_3D.jl")
include("unit_tests_admm.jl")
