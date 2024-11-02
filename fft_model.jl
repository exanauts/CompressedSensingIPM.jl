using NLPModels, FFTW

include("fft_utils.jl")
include("punching_centering.jl")

mutable struct FFTParameters
    paramB  # ::Tuple{Float64, Int64}
    eps_NT  # ::Float64
    paramLS # ::Tuple{Float64, Float64}
    paramf  # ::Tuple{Int64, Tuple{Int64}, Vector{Float64}, Int64, Vector{Int64}}
end

function FFTParameters(DFTdim, DFTsize, M_perptz, lambda, index_missing, alpha_LS, gamma_LS, eps_NT, mu_barrier, eps_barrier)
    paramf = (DFTdim, DFTsize, M_perptz, lambda, index_missing)
    paramLS = (alpha_LS, gamma_LS)
    paramB = (eps_barrier, mu_barrier)
    FFTParameters(paramB, eps_NT, paramLS, paramf)
end

mutable struct FFTNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    parameters::FFTParameters
    N::Int
    counters::Counters
end

function FFTNLPModel(parameters::FFTParameters)
    DFTdim = parameters.paramf[1]   # problem size (1, 2, 3)
    DFTsize = parameters.paramf[2]  # problem dimension
    N = prod(DFTsize)
    nvar = 2 * N
    ncon = 2 * N
    x0 = zeros(Float64, nvar)
    y0 = zeros(Float64, ncon)
    lvar = -Inf * ones(Float64, nvar)
    for i = N+1:2N
      lvar[i] = zero(Float64)  # cᵢ ≥ 0
    end
    uvar =  Inf * ones(Float64, nvar)
    lcon = -Inf * ones(Float64, ncon)
    ucon = zeros(Float64, ncon)
    meta = NLPModelMeta(
        nvar,
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        ncon = ncon,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        nnzj = nvar * ncon,
        nnzh = nvar * nvar,
        minimize = true,
        islp = false,
        name = "CompressedSensing-$(DFTdim)D",
    )
    return FFTNLPModel(meta, parameters, N, Counters())
end

function NLPModels.cons!(nlp::FFTNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons)
  N = nlp.N
  for i = 1:N
    c[i]     = -x[i] - x[i+N]  # -βᵢ - cᵢ
    c[N+i]   =  x[i] - x[i+N]  #  βᵢ - cᵢ
  end
  return c
end

function NLPModels.jprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod)
  N = nlp.N
  for i = 1:N
    Jv[i]   = -v[i] - v[i+N]
    Jv[N+i] =  v[i] - v[i+N]
  end
  return Jv
end

function NLPModels.jtprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod)
  N = nlp.N
  for i = 1:N
    Jtv[i]   = -v[i] + v[N+i]
    Jtv[i+N] = -v[i] - v[N+i]
  end
  return Jtv
end

function NLPModels.obj(nlp::FFTNLPModel, x::AbstractVector)
    increment!(nlp, :neval_obj)
    DFTdim = nlp.parameters.paramf[1]
    DFTsize = nlp.parameters.paramf[2]
    M_perptz = nlp.parameters.paramf[3]
    lambda = nlp.parameters.paramf[4]
    index_missing = nlp.parameters.paramf[5]
    # Mt = nlp.parameters.paramf[6]

    fft_val = M_perp_beta_wei(DFTdim, DFTsize, x, index_missing)
    N = nlp.N
    beta = view(x, 1:N)
    c = view(x, N+1:2*N)
    fval = 0.5 * dot(fft_val, fft_val) - dot(beta, M_perptz) + lambda * sum(c)
    return fval
end

function NLPModels.grad!(nlp::FFTNLPModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  DFTdim = nlp.parameters.paramf[1]
  DFTsize = nlp.parameters.paramf[2]
  M_perptz = nlp.parameters.paramf[3]
  lambda = nlp.parameters.paramf[4]
  index_missing = nlp.parameters.paramf[5]
  # Mt = nlp.parameters.paramf[6]

  n = prod(DFTsize)
  g_b = view(g, 1:n)
  g_c = view(g, n+1:2*n)
  beta = view(x, 1:n)
  res = M_perpt_M_perp_vec_wei(DFTdim, DFTsize, beta, index_missing)
  g_b .= res .- M_perptz
  g_c .= lambda .* ones(Float64, n)
  return g
end

function NLPModels.hprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nlp, :neval_hprod)
  DFTdim = nlp.parameters.paramf[1]
  DFTsize = nlp.parameters.paramf[2]
  M_perptz = nlp.parameters.paramf[3]
  lambda = nlp.parameters.paramf[4]
  index_missing = nlp.parameters.paramf[5]
  # Mt = nlp.parameters.paramf[6]

  n = prod(DFTsize)
  hv_b = view(hv, 1:n)
  hv_c = view(hv, n+1:2*n)
  hv_b .= M_perpt_M_perp_vec_wei(DFTdim, DFTsize, v[1:n], index_missing) .- M_perptz
  hv_c .= 0.0
  return hv
end
