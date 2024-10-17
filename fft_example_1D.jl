include("fft_model.jl")

# 1D
# Nt = 500;
Nt = 10^5;
t = collect(0:(Nt-1))

x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt)
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt)
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt)
x = x1.+x2.+x3; #signal
Random.seed!(1)
y = x + randn(Nt) #noisy signal
graphics = false

w = round.(fft(x)./sqrt(Nt), digits = 4) #true DFT
DFTsize = size(x) # problem dim
DFTdim = length(DFTsize) # problem size

missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

M_perptz = M_perp_tz_old(DFTdim, DFTsize, z_zero) # M_perptz

lambda = 10;

alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 1e-6;
eps_barrier = 1e-6;
mu_barrier = 10;

paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1;
beta_init = ones(Nt)./2;
c_init = ones(Nt)

beta_MadNLP, c_MadNLP, subgrad_MadNLP, time_MadNLP = barrier_mtd(beta_init, c_init, t_init, paramset)
println("Number of calls to CG: $(nkrylov_ipm).")

#### comparison with orginal data
w_est = beta_to_DFT(DFTdim, DFTsize, beta_MadNLP)
norm(w .- w_est)
#############

if graphics
    plot(subgrad_MadNLP, time_MadNLP, seriestype=:scatter, title = "IP: 1d (500) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
    plot(log.(subgrad_MadNLP), time_MadNLP, seriestype=:scatter, title = "IP: 1d (500) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
    plot(log.(subgrad_MadNLP), title = "IP: 1d (500) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
end
