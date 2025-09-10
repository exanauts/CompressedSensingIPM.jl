## 2D
function admm_example_2D(Nt::Int, Ns::Int; gpu::Bool=false, gpu_arch::String="cuda", rdft::Bool=false, check::Bool=false)
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

    lambda = check ? 0 : 1
    parameters = FFTParameters(DFTdim, DFTsize, z0 |> AT, lambda, index_missing)
    fft_operator = FFTOperator{VT}(prod(DFTsize), DFTdim, DFTsize, index_missing, rdft)

    # Solve with ADMM
    t1 = time()
    solution = CompressedSensingADMM(parameters, fft_operator; rho=1, maxt=1000, tol=1e-8)
    t2 = time()

    if check
        beta_ADMM = solution[1:Nt*Ns]
        beta_true = DFT_to_beta(DFTdim, DFTsize, w |> AT)
        @test norm(beta_true - beta_ADMM) ≤ 1e-6
    end

    return solution, t2-t1
end
