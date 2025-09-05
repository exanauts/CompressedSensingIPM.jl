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

include("test_admm.jl")

dim1 && include("fft_example_1D.jl")
dim2 && include("fft_example_2D.jl")
dim3 && include("fft_example_3D.jl")

include("unit_tests.jl")
