## 1D
function admm_example_1D(Nt::Int; gpu::Bool=false, gpu_arch::String="cuda", rdft::Bool=false, check::Bool=false)
    t = collect(0:(Nt-1))

    print("Generate x: ")
    x1 = 2 * cos.(2*pi*t*6/Nt)  .+ 3 * sin.(2*pi*t*6/Nt)
    x2 = 4 * cos.(2*pi*t*10/Nt) .+ 5 * sin.(2*pi*t*10/Nt)
    x3 = 6 * cos.(2*pi*t*40/Nt) .+ 7 * sin.(2*pi*t*40/Nt)
    x = x1 .+ x2 .+ x3  # signal
    println("✓")

    print("Generate y: ")
    y = check ? x : x + randn(Nt)  # noisy signal
    println("✓")

    w = fft(x) ./ sqrt(Nt)  # true DFT
    DFTsize = size(x)  # problem dim
    DFTdim = length(DFTsize)  # problem size

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
        beta_MadNLP = solution[1:Nt]
        beta_true = DFT_to_beta(DFTdim, DFTsize, w |> AT)
        @test norm(beta_true - beta_MadNLP) ≤ 1e-6
    end

    return solution, t2-t1
end

# 1D
if dim1
  for N in (100, 200, 500)
    for rdft in (false, true)
      @testset "1D -- ADMM -- CPU -- rdft=$rdft -- $N" begin
        solution, timer = admm_example_1D(N; gpu=false, rdft, check=true)
      end
    end
  end
end
