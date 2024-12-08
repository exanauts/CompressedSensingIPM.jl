using Random, Distributions
using MadNLPGPU, CUDA
using Test
Random.seed!(1)

include("fft_model.jl")
include("solver.jl")

## 2D
function fft_example_2D(Nt::Int, Ns::Int; gpu::Bool=false, rdft::Bool=false, check::Bool=false)
    t = collect(0:(Nt-1))
    s = collect(0:(Ns-1))
    x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))'

    y = check ? x : x + randn(Nt,Ns)  # noisy signal

    w = fft(x) ./ sqrt(Nt*Ns)  # true DFT
    DFTsize = size(x)  # problem dim
    DFTdim = length(DFTsize)  # problem size

    # randomly generate missing indices
    if check
        index_missing_Cartesian = Int[]
        z_zero = y
    else
        missing_prob = 0.15
        centers = centering(DFTdim, DFTsize, missing_prob)
        radius = 1
        index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)
    end

    # unify parameters for barrier method
    M_perptz = M_perp_tz_wei(DFTdim, DFTsize, z_zero)
    if gpu
        M_perptz = CuArray(M_perptz)
    end

    lambda = check ? 0 : 5
    alpha_LS = 0.1
    gamma_LS = 0.8
    eps_NT = 1e-6
    eps_barrier = 1e-6
    mu_barrier = 10

    parameters = FFTParameters(DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

    t_init = 1
    beta_init = zeros(prod(DFTsize))
    c_init = ones(prod(DFTsize))

    S = gpu ? CuVector{Float64} : Vector{Float64}
    nlp = FFTNLPModel{Float64, S}(parameters; rdft)

    # Solve with MadNLP/LBFGS
    # solver = MadNLP.MadNLPSolver(nlp; hessian_approximation=MadNLP.CompactLBFGS)
    # results = MadNLP.solve!(solver)
    # beta_MadNLP = results.solution[1:Nt*Ns]

    # Solve with MadNLP/CG
    solver = MadNLP.MadNLPSolver(
        nlp;
        max_iter=2000,
        kkt_system=FFTKKTSystem,
        nlp_scaling=false,
        print_level=MadNLP.INFO,
        dual_initialized=true,
        richardson_max_iter=0,
        tol=1e-8,
        richardson_tol=Inf,
    )
    results = ipm_solve!(solver)

    if check
        beta_MadNLP = results.solution[1:Nt*Ns]
        beta_true = DFT_to_beta(DFTdim, DFTsize, gpu ? CuArray(w) : w)
        @test norm(beta_true - beta_MadNLP) ≤ 1e-6
    end

    return nlp, solver, results
end

Nt = 16
Ns = 16
gpu = false
rdft = true
check = false
nlp, solver, results = fft_example_2D(Nt, Ns; gpu, rdft, check)
beta_MadNLP = results.solution[1:Nt*Ns]
elapsed_time = results.counters.total_time
println("Timer: $(elapsed_time)")
