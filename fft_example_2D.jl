using Random, Distributions
using MadNLP

Random.seed!(1)

include("fft_model.jl")

## 2D
Nt = 20
Ns = 24
t = collect(0:(Nt-1))
s = collect(0:(Ns-1))
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))'

y = x + randn(Nt,Ns)  # noisy signal

w = fft(x) ./ sqrt(Nt*Ns)  # true DFT
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

nlp = FFTNLPModel(parameters)

# Solve with MadNLP/LBFGS
# solver = MadNLP.MadNLPSolver(nlp; hessian_approximation=MadNLP.CompactLBFGS)
# results = MadNLP.solve!(solver)
# beta_MadNLP = results.solution[1:Nt]

# Solve with MadNLP/CG
solver = MadNLP.MadNLPSolver(
    nlp;
    max_iter=20,
    kkt_system=FFTKKTSystem,
    print_level=MadNLP.INFO,
    dual_initialized=true,
    richardson_max_iter=10,
    tol=1e-8,
    richardson_tol=1e-8,
)
results = MadNLP.solve!(solver)
beta_MadNLP = results.solution[1:Nt]
