using Random, Distributions
using MadNLPGPU, CUDA
Random.seed!(1)

include("fft_model.jl")

## 3D
function fft_example_3D(N1::Int, N2::Int, N3::Int; gpu::Bool=false, rdft::Bool=false)
    idx1 = collect(0:(N1-1))
    idx2 = collect(0:(N2-1))
    idx3 = collect(0:(N3-1))
    x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3]
    y = x + rand(N1, N2, N3) # noisy signal

    w = fft(x) ./ sqrt(N1*N2*N3)  # true DFT
    DFTsize = size(x)  # problem dim
    DFTdim = length(DFTsize)  # problem size
    beta_true = DFT_to_beta(DFTdim, DFTsize, w)
    sum(abs.(beta_true))


    # randomly generate missing indices
    missing_prob = 0.15
    centers = centering(DFTdim, DFTsize, missing_prob)
    radius = 1

    index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

    # unify parameters for barrier method
    M_perptz = M_perp_tz_wei(DFTdim, DFTsize, z_zero)
    if gpu
        M_perptz = CuArray(M_perptz)
    end

    lambda = 5

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
    # beta_MadNLP = results.solution[1:N1*N2*N3]

    # Solve with MadNLP/CG
    solver = MadNLP.MadNLPSolver(
        nlp;
        max_iter=2000,
        kkt_system=FFTKKTSystem,
        print_level=MadNLP.INFO,
        dual_initialized=true,
        richardson_max_iter=0,
        tol=1e-6,
        richardson_tol=Inf,
    )
    results = MadNLP.solve!(solver)
    return nlp, solver, results
end

N1 = 8
N2 = 8
N3 = 8
gpu = false
rdft = false
nlp, solver, results = fft_example_3D(N1, N2, N3; gpu, rdft)
beta_MadNLP = results.solution[1:N1*N2*N3]
