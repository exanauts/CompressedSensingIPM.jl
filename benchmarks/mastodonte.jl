using Random, Distributions
using LinearAlgebra, SparseArrays
using HDF5
using DelimitedFiles
using FFTW
using MadNLP, MadNLPGPU
using CUDA, AMDGPU
using Test
using CompressedSensingIPM

function mastodonte(z0, mask; lambda::Float64=1.0, gpu::Bool=false, gpu_arch::String="cuda", rdft::Bool=false)
  nnz_missing = length(mask) - nnz(vec(mask) |> sparse)

  # index_missing = Vector{CartesianIndex{3}}(undef, nnz_missing)
  index_missing = Vector{Int}(undef, nnz_missing)

  DFTsize = size(mask)  # problem dim
  DFTdim = length(DFTsize)  # problem size

  if gpu
    if gpu_arch == "cuda"
      AT = CuArray{Float64}
      VT = CuVector{Float64}
      VI = CuVector{Int}
    elseif gpu_arch == "rocm"
      AT = ROCArray{Float64}
      VT = ROCVector{Float64}
      VI = ROCVector{Int}
    else
      error("Unsupported GPU architecture \"$gpu_arch\".")
    end
  else
    AT = Array{Float64}
    VT = Vector{Float64}
    VI = Vector{Int}
  end

  pos = 0
  for k in 1:DFTsize[3]
    for j in 1:DFTsize[2]
      for i in 1:DFTsize[1]
        if mask[i,j,k] == 0
          pos += 1
          # index_missing[pos] = CartesianIndex{3}(i, j, k)
          index_missing[pos] = i + (j-1) * DFTsize[1] + (k-1) * DFTsize[1] * DFTsize[2]
        end
      end
    end
  end
  index_missing = VI(index_missing)

  parameters = FFTParameters(DFTdim, DFTsize, z0 |> AT, lambda, index_missing)
  nlp = FFTNLPModel{VT}(parameters; rdft, preconditioner=true)

  # Solve with MadNLP/CG
  t1 = time()
  solver = MadNLP.MadNLPSolver(
    nlp;
    max_iter=23,
    kkt_system=FFTKKTSystem,
    nlp_scaling=false,
    print_level=MadNLP.INFO,
    dual_initialized=true,
    richardson_max_iter=0,
    tol=1e-8,
    richardson_tol=Inf,
  )

  results = CompressedSensingIPM.ipm_solve!(solver)
  t2 = time()
  return nlp, solver, results, t2-t1
end

# Load the problem (old storage)
path_z0_h5 = "7_7_sec_ord_rings_800_800_200.h5"
z0_h5 = h5open(path_z0_h5, "r")
z0 = read(z0_h5["data"])

# path_mask_h5 = "mask_template_800_800_200.h5"
path_mask_h5 = "mask_template_f_n_s_800_800_200.h5"
mask_h5 = h5open(path_mask_h5, "r")
mask = read(mask_h5["data"])

# Load the problem (new storage)
# path_h5 = "movo2_40_120K_slice_pm4.0xpm4.0xpm4.0_br0.20_sg123.nxs.h5"
# file_h5 = file_h5 = h5open(path_h5, "r")
# mask = read(file_h5["entry"]["data"]["weights"])
# z0 = read(file_h5["entry"]["data"]["signal"])

# Options for the solver
gpu = true
gpu_arch = "cuda"
# gpu_arch = "rocm"
rdft = true
lambda = 1e-6

# Solve the problem and recover the solution
nlp, solver, results, timer = mastodonte(z0, mask; lambda, gpu, gpu_arch, rdft)
N = length(results.solution) ÷ 2
beta_MadNLP = results.solution[1:N]
println("Timer: $(timer)")

# solver.kkt.krylov_iterations
# solver.kkt.krylov_timer
# nlp.fft_timer[]
# nlp.mapping_timer[]

# Dump the solution in a file
dump_solution = true
if dump_solution
  DFTsize = size(mask)
  v = CompressedSensingIPM.beta_to_DFT(3, DFTsize, beta_MadNLP; rdft=rdft)
  x_recovered = ifft(v) .* sqrt(prod(DFTsize))
  x_recovered = Array(x_recovered) |> real
  h5write("solution.h5", "solution", x_recovered)
end
