using Random, Distributions
using LinearAlgebra, SparseArrays
using LaplaceInterpolation, NPZ
using MadNLPGPU, CUDA
using Test

include("fft_model.jl")
include("solver.jl")

function punch_3D_cart(center, radius, x, y, z; linear = false)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : 
                                                (radius, radius, radius)
    inds = filter(i -> (((x[i[1]]-center[1])/radius_x)^2 
                        + ((y[i[2]]-center[2])/radius_y)^2 
                        + ((z[i[3]] - center[3])/radius_z)^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    (length(inds) == 0) && error("Empty punch.")
    if linear == false
      return inds
    else
      return LinearIndices(zeros(length(x), length(y), length(z)))[inds]
    end 
end

function fft_vishwas(z3d; variant::Bool=false, gpu::Bool=false, rdft::Bool=false)
    if ! variant
        dx = 0.02
        dy = 0.02
        dz = 0.02
        x = -0.2:dx:4.01
        y = -0.2:dy:6.01
        z = -0.2:dz:6.01
        x = x[1:210]
        y = y[1:310]
        z = z[1:310]

        radius = 0.2001
        punched_pmn = copy(z3d)
        punched_pmn = punched_pmn[1:210, 1:310, 1:310]

        index_missing_2D = CartesianIndex{3}[]
        for i=0:4.
            for j=0:6.
                for k = 0:6.
                    center =[i,j,k]
                    absolute_indices1 = punch_3D_cart(center, radius, x, y, z)
                    punched_pmn[absolute_indices1] .= 0
                    append!(index_missing_2D, absolute_indices1)
                end
            end
        end
    else
        dx = 0.02
        dy = 0.02
        dz = 0.02
        x = -0.2:dx:6.01
        y = -0.2:dy:8.01
        z = -0.2:dz:8.01
        x = x[1:310]
        y = y[1:410]
        z = z[1:410]

        radius = 0.2001
        punched_pmn = copy(z3d)
        punched_pmn = punched_pmn[1:310, 1:410, 1:410]

        index_missing_2D = CartesianIndex{3}[]
        for i=0:6.
            for j=0:8.
                for k = 0:8.
                    center =[i,j,k]
                    absolute_indices1 = punch_3D_cart(center, radius, x, y, z)
                    punched_pmn[absolute_indices1] .= 0
                    append!(index_missing_2D, absolute_indices1)
                end
            end
        end
    end

    DFTsize = size(punched_pmn)  # problem dim
    DFTdim = length(DFTsize)  # problem size
    M_perptz = M_perp_tz_wei(DFTdim, DFTsize, punched_pmn)
    if gpu
        M_perptz = CuArray(M_perptz)
    end
    Nt = prod(DFTsize)

    lambda = 1

    alpha_LS = 0.1
    gamma_LS = 0.8
    eps_NT = 1e-6
    eps_barrier = 1e-6
    mu_barrier = 10

    parameters = FFTParameters(DFTdim, DFTsize, M_perptz, lambda, index_missing_2D, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

    t_init = 1
    beta_init = ones(Nt) ./ 2
    c_init = ones(Nt)

    S = gpu ? CuVector{Float64} : Vector{Float64}
    nlp = FFTNLPModel{Float64, S}(parameters; rdft)

    # Solve with MadNLP/CG
    t1 = time()
    solver = MadNLP.MadNLPSolver(
        nlp;
        max_iter=10000,
        kkt_system=FFTKKTSystem,
        nlp_scaling=false,
        print_level=MadNLP.INFO,
        dual_initialized=true,
        richardson_max_iter=0,
        tol=1e-8,
        richardson_tol=Inf,
    )
    results = ipm_solve!(solver)
    t2 = time()
    return nlp, solver, results, t2-t1
end

gpu = true
rdft = false
variant = false
z3d = npzread("../z3d_movo.npy")
nlp, solver, results, timer = fft_vishwas(z3d; gpu, rdft)
N = length(results.solution) ÷ 2
beta_MadNLP = results.solution[1:N]
println("Timer: $(timer)")

using DelimitedFiles
open("sol_vishwas.txt", "w") do io
    writedlm(io, Vector(beta_MadNLP))
end
