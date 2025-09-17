## 2D
function ipm_example_2D(Nt::Int, Ns::Int; kkt=FFTKKTSystem, gpu::Bool=false, gpu_arch::String="cuda", rdft::Bool=false, check::Bool=false)
    t = collect(0:(Nt-1))
    s = collect(0:(Ns-1))

    print("Generate x: ")
    x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))'
    println("✓")

    print("Generate y: ")
    y = check ? x : x + randn(Nt,Ns)  # noisy signal
    println("✓")

    w = fft(x) ./ sqrt(Nt*Ns)  # true DFT
    DFTsize = size(x)  # problem dim
    DFTdim = length(DFTsize)  # problem size

    # randomly generate missing indices
    print("Generate missing indices: ")
    if check
        index_missing = Int[]
        z0 = y
    else
        missing_prob = 0.15
        centers = centering(DFTdim, DFTsize, missing_prob)
        radius = 1
        index_missing, z0 = punching(DFTdim, DFTsize, centers, radius, y)
        # println("length(index_missing) = ", length(index_missing))
    end
    println("✓")

    if gpu
        if gpu_arch == "cuda"
            AT = CuArray
            VT = CuVector{Float64}
        elseif gpu_arch == "rocm"
            AT = ROCArray
            VT = ROCVector{Float64}
        else
            error("Unsupported GPU architecture \"$gpu_arch\".")
        end
    else
        AT = Array
        VT = Vector{Float64}
    end

    lambda = check ? 0 : 5
    parameters = FFTParameters(DFTdim, DFTsize, z0 |> AT, lambda, index_missing)
    if kkt == FFTKKTSystem
        nlp = FFTNLPModel{VT}(parameters; rdft)
    else
        nlp = GondzioNLPModel{VT}(parameters; rdft)
    end

    # Solve with MadNLP/CG
    t1 = time()
    solver = MadNLP.MadNLPSolver(
        nlp;
        max_iter=2000,
        kkt_system=kkt,
        nlp_scaling=false,
        print_level=MadNLP.INFO,
        dual_initialized=true,
        richardson_max_iter=0,
        tol=1e-8,
        richardson_tol=Inf,
    )
    results = CompressedSensingIPM.ipm_solve!(solver)
    t2 = time()

    if check
        beta_MadNLP = results.solution[1:Nt*Ns]
        beta_true = DFT_to_beta(DFTdim, DFTsize, w |> AT)
        @test norm(beta_true - beta_MadNLP) ≤ 1e-6
    end

    return nlp, solver, results, t2-t1
end

# Nt = 16
# Ns = 16
# gpu = false
# gpu_arch = "cuda"
# rdft = true
# check = false
# nlp, solver, results, timer = ipm_example_2D(Nt, Ns; gpu, gpu_arch, rdft, check)
# beta_MadNLP = results.solution[1:Nt*Ns]
# println("Timer: $(timer)")
