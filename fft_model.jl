using NLPModels

include("fft_utils.jl")
include("punching_centering.jl")

mutable struct FFTParameters
    paramB
    eps_NT
    paramLS
    paramf
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
    c::Vector{Float64}
    counters::Counters
end

meta = NLPModelMeta(
nvar,
x0 = x0,
lvar = lvar,
uvar = uvar,
ncon = ncon,
y0 = zeros(ncon),
lcon = lcon,
ucon = ucon,
nnzj = nnzj,
nnzh = nnzh,
lin = collect(1:nlin),
lin_nnzj = lincon.nnzj,
nln_nnzj = quadcon.nnzj + nlcon.nnzj,
minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE,
islp = (obj.type == "LINEAR") && (nnln == 0) && (quadcon.nquad == 0),
name = name,
)

function NLPModels.cons!(nlp::FFTNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons)
  # ...
  return c
end

function NLPModels.jprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod)
  # ...
  return Jv
end

function NLPModels.jtprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod)
  # ...
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
    fval = 0.5 * dot(fft_val, fft_val) - dot(beta, M_perptz) + lambda * sum(nlp.c)
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

  n = prod(DFTdim)
  g_b = view(g, 1:n)
  g_c = view(g, n+1:2*n)
  g_b .= M_perpt_M_perp_vec_wei(DFTdim, DFTsize, x, index_missing) .- Mperptz
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

  n = prod(DFTdim)
  hv_b = view(hv, 1:n)
  hv_c = view(hv, n+1:2*n)
  hv_b .= M_perpt_M_perp_vec_wei(DFTdim, DFTsize, v, index_missing) .- Mperptz
  hv_c .= 0.0
  return hv
end
