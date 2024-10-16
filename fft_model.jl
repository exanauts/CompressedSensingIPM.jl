using NLPModels

mutable struct FFTNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
	meta::NLPModelMeta{Float64, Vector{Float64}}
	counters::Counters
end

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
  ...
  return c
end

function NLPModels.jprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod)
  ...
  return Jv
end

function NLPModels.jtprod!(
  nlp::FFTNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod)
  ...
  return Jtv
end

function NLPModels.obj(nlp::FFTNLPModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  ...
  return res
end

function NLPModels.grad!(nlp::FFTNLPModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  ...
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
  ...
  return hv
end
