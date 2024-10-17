include("fft_model.jl")

## 2D
# Nt = 20;
# Ns = 24;
Nt = 4;
Ns = 4;
t = collect(0:(Nt-1))
s = collect(0:(Ns-1))
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)  #noisy signal
graphics = false

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4)#true DFT
DFTsize = size(x) # problem dim
DFTdim = length(DFTsize) # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w)
sum(abs.(beta_true))

# randomly generate missing indices
missing_prob = 0.15
centers = centering(DFTdim, DFTsize, missing_prob)
radius = 1

index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)


# unify parameters for barrier method
M_perptz = M_perp_tz_old(DFTdim, DFTsize, z_zero)
lambda = 5

paramset = paramunified(DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)

t_init = 1;
beta_init = zeros(prod(DFTsize))
c_init = ones(prod(DFTsize))

beta_MadNLP, c_MadNLP, subgrad_MadNLP, time_MadNLP = barrier_mtd(beta_init, c_init, t_init, paramset)

#### comparison with orginal data
w_est = beta_to_DFT(DFTdim, DFTsize, beta_MadNLP)
norm(w .- w_est)

#############

if graphics
    plot(subgrad_MadNLP, time_MadNLP, seriestype=:scatter, title = "IP: 2d (20*24) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
    plot(log.(subgrad_MadNLP), time_MadNLP, seriestype=:scatter, title = "IP: 2d (20*24) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
    plot(log.(subgrad_MadNLP), title = "IP: 2d (20*24) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
end
