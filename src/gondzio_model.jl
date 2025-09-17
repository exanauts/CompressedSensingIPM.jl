mutable struct GondzioNLPModel{T,VT,FFT,P} <: AbstractNLPModel{T,VT}
    meta::NLPModelMeta{T,VT}
    counters::Counters
    parameters::P
    nβ::Int
    op_fft::FFT
    M_perpt_z0::VT
    krylov_solver::Symbol
    preconditioner::Bool
end

function GondzioNLPModel{VT}(parameters::FFTParameters;
                             krylov_solver::Symbol=:cg,
                             rdft::Bool=false,
                             preconditioner::Bool=true) where {VT <: AbstractVector}
    T = eltype(VT)
    DFTdim = parameters.DFTdim   # problem size (1, 2, 3)
    DFTsize = parameters.DFTsize  # problem dimension
    index_missing = parameters.index_missing
    nβ = prod(DFTsize)

    nvar = ...
    ncon = ...
    x0 = ...
    y0 = ...
    lvar = ...
    uvar = ...
    lcon = ...
    ucon = ...
    meta = NLPModelMeta(
        nvar,
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        ncon = ncon,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        nnzj = 0,
        nnzh = 0,
        minimize = true,
        islp = false,
        name = "CompressedSensing-Gondzio-$(DFTdim)D",
    )

    # FFT operator
    op_fft = FFTOperator{VT}(nβ, DFTdim, DFTsize, index_missing, rdft)
    tmp = M_perpt_z(op_fft, parameters.z0)
    M_perpt_z0 = copy(tmp)
    return GondzioNLPModel(meta, Counters(), parameters, nβ, op_fft, M_perpt_z0, krylov_solver, preconditioner)
end

function NLPModels.obj(nlp::GondzioNLPModel, x::AbstractVector)
    increment!(nlp, :neval_obj)
    DFTdim = nlp.parameters.DFTdim
    DFTsize = nlp.parameters.DFTsize
    lambda = nlp.parameters.lambda
    index_missing = nlp.parameters.index_missing

    fval = ...
    return fval
end

function NLPModels.grad!(nlp::GondzioNLPModel, x::AbstractVector, g::AbstractVector)
    increment!(nlp, :neval_grad)
    DFTdim = nlp.parameters.DFTdim
    DFTsize = nlp.parameters.DFTsize
    lambda = nlp.parameters.lambda
    index_missing = nlp.parameters.index_missing

    ...
    return g
end

function NLPModels.cons!(nlp::GondzioNLPModel, x::AbstractVector, c::AbstractVector)
    increment!(nlp, :neval_cons)
    ...
    return c
end
