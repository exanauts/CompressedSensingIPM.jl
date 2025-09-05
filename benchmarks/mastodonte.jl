using Random, Distributions
using LinearAlgebra, SparseArrays
using HDF5
using DelimitedFiles
using FFTW
using MadNLP, MadNLPGPU
using CUDA, AMDGPU
using Test
using CompressedSensingIPM

include("../test/fft_wei.jl")
include("../test/punching_centering.jl")

function punch_3D_cart(center, radius, x, y, z; linear = false)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : 
                                                (radius, radius, radius)
    inds = filter(i -> (((x[i[1]]-center[1])/radius_x)^2 
                        + ((y[i[2]]-center[2])/radius_y)^2 
                        + ((z[i[3]] - center[3])/radius_z)^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    (length(inds) == 0) && error("Empty punch.")
    if linear == false
      return inds
    else
      return LinearIndices(zeros(length(x), length(y), length(z)))[inds]
    end 
end

function mastodonte(A; gpu::Bool=false, gpu_arch::String="cuda", rdft::Bool=false)
  punched_pmn = copy(A)
  index_missing_3D = CartesianIndex{3}[]

  DFTsize = size(punched_pmn)  # problem dim
  DFTdim = length(DFTsize)  # problem size
  if gpu
    if gpu_arch == "cuda"
      AT = CuArray{Float64}
      VT = CuVector{Float64}
    elseif gpu_arch == "rocm"
      AT = ROCArray{Float64}
      VT = ROCVector{Float64}
    else
      error("Unsupported GPU architecture \"$gpu_arch\".")
    end
  else
    AT = Array{Float64}
    VT = Vector{Float64}
  end

  lambda = 1
  parameters = FFTParameters(DFTdim, DFTsize, punched_pmn |> AT, lambda, index_missing_3D)
  nlp = FFTNLPModel{VT}(parameters; rdft, preconditioner=true)

  # Solve with MadNLP/CG
  t1 = time()
  solver = MadNLP.MadNLPSolver(
    nlp;
    max_iter=10000,
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

gpu = true
gpu_arch = "cuda"  # "rocm"
rdft = true
path_h5 = "mask_template_800_800_200.h5"
h5 = h5open(path_h5, "r")
obj = h5["data"]
A = read(obj)
nlp, solver, results, timer = mastodonte(A; gpu, gpu_arch, rdft)
N = length(results.solution) ÷ 2
beta_MadNLP = results.solution[1:N]
println("Timer: $(timer)")

# solver.kkt.krylov_iterations
# solver.kkt.krylov_timer
# nlp.fft_timer[]
# nlp.mapping_timer[]

dump_solution = false
if dump_solution
    open("sol_vishwas.txt", "w") do io
        writedlm(io, Vector(beta_MadNLP))
    end
end
