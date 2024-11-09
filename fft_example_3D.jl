using Random, Distributions
using MadNLP

Random.seed!(1)

include("fft_model.jl")

## 3D
N1 = 4
N2 = 4
N3 = 4

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
lambda = 5

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
