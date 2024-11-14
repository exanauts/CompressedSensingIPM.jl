using Random, Distributions
Random.seed!(1)

include("fft_model.jl")

# 1D
# Nt = 100
Nt = 100000
t = collect(0:(Nt-1))

x1 = 2 * cos.(2*pi*t*6/Nt)  .+ 3 * sin.(2*pi*t*6/Nt)
x2 = 4 * cos.(2*pi*t*10/Nt) .+ 5 * sin.(2*pi*t*10/Nt)
x3 = 6 * cos.(2*pi*t*40/Nt) .+ 7 * sin.(2*pi*t*40/Nt)
x = x1 .+ x2 .+ x3  # signal

y = x + randn(Nt)  # noisy signal

w = fft(x) ./ sqrt(Nt)  # true DFT
DFTsize = size(x)  # problem dim
DFTdim = length(DFTsize)  # problem size

missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

M_perptz = M_perp_tz_wei(DFTdim, DFTsize, z_zero)  # M_perptz

lambda = 1

alpha_LS = 0.1
gamma_LS = 0.8
eps_NT = 1e-6
eps_barrier = 1e-6
mu_barrier = 10

parameters = FFTParameters(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1
beta_init = ones(Nt) ./ 2
c_init = ones(Nt)

nlp = FFTNLPModel{Float64, Vector{Float64}}(parameters)

# Solve with MadNLP/LBFGS
# solver = MadNLP.MadNLPSolver(nlp; hessian_approximation=MadNLP.CompactLBFGS)
# results = MadNLP.solve!(solver)
# beta_MadNLP = results.solution[1:Nt]

# Solve with MadNLP/CG
solver = MadNLP.MadNLPSolver(
    nlp;
    max_iter=2000,
    kkt_system=FFTKKTSystem,
    print_level=MadNLP.INFO,
    dual_initialized=true,
    richardson_max_iter=0,
    tol=1e-8,
    richardson_tol=Inf,
)
results = MadNLP.solve!(solver)
beta_MadNLP = results.solution[1:Nt]
