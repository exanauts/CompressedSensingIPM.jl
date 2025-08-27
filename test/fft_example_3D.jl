## 3D
function fft_example_3D(N1::Int, N2::Int, N3::Int; gpu::Bool=false, gpu_arch::String="cuda", rdft::Bool=false, check::Bool=false)
    idx1 = collect(0:(N1-1))
    idx2 = collect(0:(N2-1))
    idx3 = collect(0:(N3-1))

    print("Generate x: ")
    x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3]
    println("✓")

    print("Generate y: ")
    y = check ? x : x + rand(N1, N2, N3)  # noisy signal
    println("✓")

    w = fft(x) ./ sqrt(N1*N2*N3)  # true DFT
    DFTsize = size(x)  # problem dim
    DFTdim = length(DFTsize)  # problem size

    # randomly generate missing indices
    print("Generate missing indices: ")
    if check
        index_missing = Int[]
        z_zero = y
    else
        missing_prob = 0.15
        centers = centering(DFTdim, DFTsize, missing_prob)
        radius = 1
        index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)
    end
    println("✓")

    # unify parameters for barrier method
    M_perptz = M_perp_tz_wei(DFTdim, DFTsize, z_zero)
    AT = Array
    S = Vector{Float64}
    if gpu
        if gpu_arch == "cuda"
            M_perptz = CuArray(M_perptz)
            AT = CuArray
            S = CuVector{Float64}
        elseif gpu_arch == "rocm"
            M_perptz = ROCArray(M_perptz)
            AT = ROCArray
            S = ROCVector{Float64}
        else
            error("Unsupported GPU architecture \"$gpu_arch\".")
        end
    end

    lambda = check ? 0 : 5
    parameters = FFTParameters(DFTdim, DFTsize, M_perptz, lambda, index_missing)
    nlp = FFTNLPModel{Float64, S}(parameters; rdft)

    # Solve with MadNLP/CG
    t1 = time()
    solver = MadNLP.MadNLPSolver(
        nlp;
        max_iter=2000,
        kkt_system=FFTKKTSystem,
        print_level=MadNLP.INFO,
        nlp_scaling=false,
        dual_initialized=true,
        richardson_max_iter=0,
        tol=1e-8,
        richardson_tol=Inf,
    )
    results = CompressedSensingIPM.ipm_solve!(solver)
    t2 = time()

    if check
        beta_MadNLP = results.solution[1:N1*N2*N3]
        beta_true = DFT_to_beta(DFTdim, DFTsize, w |> AT)
        @test norm(beta_true - beta_MadNLP) ≤ 1e-6
    end

    return nlp, solver, results, t2-t1
end

# N1 = 8
# N2 = 8
# N3 = 8
# gpu = false
# gpu_arch = "cuda"
# rdft = true
# check = false
# nlp, solver, results, timer = fft_example_3D(N1, N2, N3; gpu, gpu_arch, rdft, check)
# beta_MadNLP = results.solution[1:N1*N2*N3]
# println("Timer: $(timer)")
