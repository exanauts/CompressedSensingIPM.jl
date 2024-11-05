using Random, Distributions
using MadNLP

Random.seed!(1)

include("fft_model.jl")

# 1D
Nt = 100
# Nt = 10^5
t = collect(0:(Nt-1))

x1 = 2 * cos.(2*pi*t*6/Nt)  .+ 3 * sin.(2*pi*t*6/Nt)
x2 = 4 * cos.(2*pi*t*10/Nt) .+ 5 * sin.(2*pi*t*10/Nt)
x3 = 6 * cos.(2*pi*t*40/Nt) .+ 7 * sin.(2*pi*t*40/Nt)
x = x1 .+ x2 .+ x3  # signal

y = x + randn(Nt)  # noisy signal
graphics = false

w = fft(x) ./ sqrt(Nt)  # true DFT
DFTsize = size(x)  # problem dim
DFTdim = length(DFTsize)  # problem size

missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

M_perptz = M_perp_tz_wei(DFTdim, DFTsize, z_zero) # M_perptz

lambda = 10

alpha_LS = 0.1
gamma_LS = 0.8
eps_NT = 1e-6
eps_barrier = 1e-6
mu_barrier = 10

parameters = FFTParameters(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1
beta_init = ones(Nt) ./ 2
c_init = ones(Nt)

nlp = FFTNLPModel(parameters)
nvar = get_nvar(nlp)
ncon = get_ncon(nlp)
d = [beta_init; c_init]
obj(nlp, d)
cons(nlp, d)
grad(nlp, d)
N = prod(DFTsize)
y = ones(Float64, ncon)
hv = zeros(Float64, nvar)
v = rand(Float64, nvar)
hprod!(nlp, d, y, v, hv)
Jv = zeros(Float64, ncon)
jprod!(nlp, d, v, Jv)
Jtv = zeros(Float64, nvar)
# w = rand(Float64, ncon)

nnzj = NLPModels.get_nnzj(nlp)
rows, cols = NLPModels.jac_structure(nlp)
vals = zeros(nnzj)
jac_coord!(nlp, d, vals)

# Solve with MadNLP/LBFGS
solver = MadNLP.MadNLPSolver(nlp; hessian_approximation=MadNLP.CompactLBFGS)
results = MadNLP.solve!(solver)
beta_MadNLP = results.solution[1:Nt]


# beta_MadNLP, c_MadNLP, subgrad_MadNLP, time_MadNLP = barrier_mtd(beta_init, c_init, t_init, paramset)
# println("Number of calls to CG: $(nkrylov_ipm).")

# #### comparison with orginal data
# w_est = beta_to_DFT(DFTdim, DFTsize, beta_MadNLP)
# norm(w .- w_est)
# #############

# if graphics
#     plot(subgrad_MadNLP, time_MadNLP, seriestype=:scatter, title = "IP: 1d (500) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
#     plot(log.(subgrad_MadNLP), time_MadNLP, seriestype=:scatter, title = "IP: 1d (500) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
#     plot(log.(subgrad_MadNLP), title = "IP: 1d (500) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
# end
