using NLPModels

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
    DFTdim = nlp.parameters.paramf[1]
    DFTsize = nlp.parameters.paramf[2]
    M_perptz = nlp.parameters.paramf[3]
    lambda = nlp.parameters.paramf[4]
    index_missing = nlp.parameters.paramf[5]
	# Mt = nlp.parameters.paramf[6]

    # c should be in the model?
    fval = 0.5 * sum((M_perp_beta_old(DFTdim, DFTsize, beta, index_missing)).^2) - beta' * M_perptz + lambda * sum(c)
    return fval
  return res
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
  gradb = view(g, 1:n)
  gradc = view(g, n+1:2*n)
  gradb .= M_perpt_M_perp_vec_old(DFTdim, DFTsize, beta, index_missing).-Mperptz)
  gradc .= lambda .* ones(n)
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
