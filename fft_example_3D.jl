include("fft_model.jl")

## 3D
# N1 = 6;
# N2 = 8;
# N3 = 10;

N1 = 100;
N2 = 100;
N3 = 100;

idx1 = collect(0:(N1-1))
idx2 = collect(0:(N2-1))
idx3 = collect(0:(N3-1))
x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3];
Random.seed!(2)
y = x + rand(N1, N2, N3) # noisy signal
graphics = false

w = round.(fft(x)./sqrt(N1*N2*N3), digits = 4)#true DFT
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

rho = 1
paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing_Cartesian)

if version_alexis
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM_alexis(paramf, rho)
    t = @elapsed cgADMM_alexis(paramf, rho)
    println("The third example requires $t seconds.")
else
    beta_ADMM, subgrad_ADMM, time_ADMM = cgADMM(paramf, rho)
    t = @elapsed cgADMM(paramf, rho)
    println("The third example requires $t seconds.")
end

#### comparison with orginal data
w_est = beta_to_DFT(DFTdim, DFTsize, beta_MadNLP)
norm(w .- w_est)
#############

if graphics
    plot(subgrad_MadNLP, time_MadNLP, seriestype=:scatter, title = "IP: 3d (6*8*10) time vs subgrad", xlab = "subgrad", ylab = "time", legend = false)
    plot(log.(subgrad_MadNLP), time_MadNLP, seriestype=:scatter, title = "IP: 3d (6*8*10) time vs log(subgrad)", xlab = "log(subgrad)", ylab = "time", legend = false)
    plot(log.(subgrad_MadNLP), title = "IP: 3d (6*8*10) log(subgrad)", xlabel = "iter", ylabel = "log(subgrad)", legend = false)
end
