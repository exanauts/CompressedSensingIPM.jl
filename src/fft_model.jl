mutable struct FFTParameters{AT,N,IM}
    DFTdim::Int64
    DFTsize::NTuple{N,Int64}
    z0::AT
    lambda::Float64
    index_missing::IM

    function FFTParameters(DFTdim, DFTsize, z0, lambda, index_missing)
        AT = typeof(z0)
        IM = typeof(index_missing)
        N = DFTdim
        new{AT,N,IM}(DFTdim, DFTsize, z0, lambda, index_missing)
    end
end

mutable struct FFTNLPModel{T,VT,FFT,P} <: AbstractNLPModel{T,VT}
    meta::NLPModelMeta{T,VT}
    counters::Counters
    parameters::P
    nβ::Int
    op_fft::FFT
    M_perpt_z0::VT
    krylov_solver::Symbol
    preconditioner::Bool
end

function FFTNLPModel{VT}(parameters::FFTParameters;
                         krylov_solver::Symbol=:cg,
                         rdft::Bool=false,
                         preconditioner::Bool=true) where {VT <: AbstractVector}
    T = eltype(VT)
    DFTdim = parameters.DFTdim   # problem size (1, 2, 3)
    DFTsize = parameters.DFTsize  # problem dimension
    index_missing = parameters.index_missing
    nβ = prod(DFTsize)
    nvar = 2 * nβ
    ncon = 2 * nβ
    x0 = VT(undef, nvar)
    y0 = VT(undef, ncon)
    lvar = VT(undef, nvar)
    uvar = VT(undef, nvar)
    lcon = VT(undef, ncon)
    ucon = VT(undef, ncon)
    fill!(x0, zero(T))
    fill!(y0, zero(T))
    fill!(lvar, -Inf)
    fill!(uvar, Inf)
    fill!(lcon, -Inf)
    fill!(ucon, zero(T))
    meta = NLPModelMeta(
        nvar,
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        ncon = ncon,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        nnzj = 0, # 2 * ncon,
        nnzh = 0, # div(nβ * (nβ + 1), 2),
        minimize = true,
        islp = false,
        name = "CompressedSensing-$(DFTdim)D",
    )

    # FFT operator
    op_fft = FFTOperator{VT}(nβ, DFTdim, DFTsize, index_missing, rdft)
    tmp = M_perpt_z(op_fft, parameters.z0)
    M_perpt_z0 = copy(tmp)
    return FFTNLPModel(meta, Counters(), parameters, nβ, op_fft, M_perpt_z0, krylov_solver, preconditioner)
end

function NLPModels.obj(nlp::FFTNLPModel, x::AbstractVector)
    increment!(nlp, :neval_obj)
    lambda = nlp.parameters.lambda
    fft_val = M_perp_beta(nlp.op_fft, x)
    nβ = nlp.nβ
    beta = view(x, 1:nβ)
    c = view(x, nβ+1:2*nβ)
    fval = 0.5 * dot(fft_val, fft_val) - dot(beta, nlp.M_perpt_z0) + lambda * sum(c)
    return fval
end

function NLPModels.grad!(nlp::FFTNLPModel, x::AbstractVector, g::AbstractVector)
    increment!(nlp, :neval_grad)
    lambda = nlp.parameters.lambda

    nβ = nlp.nβ
    g_b = view(g, 1:nβ)
    g_c = view(g, nβ+1:2*nβ)
    beta = view(x, 1:nβ)
    res = M_perpt_M_perp_vec(nlp.op_fft, beta)
    g_b .= res .- nlp.M_perpt_z0
    fill!(g_c, lambda)
    return g
end

function NLPModels.cons!(nlp::FFTNLPModel, x::AbstractVector, c::AbstractVector)
    increment!(nlp, :neval_cons)
    nβ = nlp.nβ
    xβ = view(x, 1:nβ)
    xc = view(x, nβ+1:2*nβ)
    cβ = view(c, 1:nβ)
    cc = view(c, nβ+1:2*nβ)
    cβ .= .- xβ .- xc  # -βᵢ - cᵢ for 1 ≤ i ≤ nβ
    cc .=    xβ .- xc  #  βᵢ - cᵢ for nβ+1 ≤ i ≤ 2nβ
    return c
end
