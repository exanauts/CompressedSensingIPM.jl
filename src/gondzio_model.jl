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
    m = nβ

    nvar = 2 * nβ + m
    ncon = m
    x0 = VT(undef, nvar) ; fill!(x0, zero(T))
    y0 = VT(undef, ncon) ; fill!(x0, zero(T))
    lvar = VT(undef, nvar)
    view(lvar, 1:2*nβ) .= 0
    view(lvar, 2*nβ+1:nvar) .= -Inf
    uvar = VT(undef, nvar)
    uvar .= Inf
    lcon = VT(undef, ncon)
    lcon .= 0
    ucon = VT(undef, ncon)
    ucon .= 0
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
        variable_bounds_analysis = false,
        constraint_bounds_analysis = false,
        grad_available = true,
        jac_available = false,
        hess_available = false,
        jprod_available = false,
        jtprod_available = false,
        hprod_available = false,
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
    nvar = nlp.meta.nvar

    nβ = nlp.nβ
    theta_x = view(x, 1:2*nβ)
    theta_r = view(x, 2*nβ+1:nvar)
    fval = 0.5 * dot(theta_r, theta_r) + lambda * sum(theta_x)
    return fval
end

function NLPModels.grad!(nlp::GondzioNLPModel, x::AbstractVector, g::AbstractVector)
    increment!(nlp, :neval_grad)
    DFTdim = nlp.parameters.DFTdim
    DFTsize = nlp.parameters.DFTsize
    lambda = nlp.parameters.lambda
    nvar = nlp.meta.nvar

    nβ = nlp.nβ
    g_b = view(g, 1:2*nβ)
    g_b .= lambda
    g_c = view(g, 2*nβ+1:nvar)
    g_c .= view(x, 2*nβ+1:nvar)
    return g
end

function NLPModels.cons!(nlp::GondzioNLPModel, x::AbstractVector, c::AbstractVector)
    increment!(nlp, :neval_cons)
    DFTdim = nlp.parameters.DFTdim
    DFTsize = nlp.parameters.DFTsize
    lambda = nlp.parameters.lambda
    nβ = nlp.nβ
    nvar = nlp.meta.nvar

    theta_r = view(x, 2*nβ+1:nvar)
    b = nlp.parameters.z0
    diff_x = view(x, 1:nβ) - view(x, nβ+1:2*nβ)
    Ux = M_perp_beta(nlp.op_fft, diff_x)
    c .= b .- Ux .- theta_r
    return c
end
