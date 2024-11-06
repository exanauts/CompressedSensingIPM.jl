using NLPModels, FFTW

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
        nnzj = ncon * 2,
        nnzh = div(N * (N + 1), 2),
        minimize = true,
        islp = false,
        name = "CompressedSensing-$(DFTdim)D",
    )
    return FFTNLPModel(meta, parameters, N, Counters())
end

include("kkt.jl")
include("fft_utils.jl")
include("punching_centering.jl")

function NLPModels.cons!(nlp::FFTNLPModel, x::AbstractVector, c::AbstractVector)
    increment!(nlp, :neval_cons)
    N = nlp.N
    for i = 1:N
        c[i]     = -x[i] - x[i+N]  # -βᵢ - cᵢ
        c[N+i]   =  x[i] - x[i+N]  #  βᵢ - cᵢ
    end
    return c
end

function NLPModels.jac_structure!(nlp::FFTNLPModel, rows::Vector{Int}, cols::Vector{Int})
    N = nlp.N
    k = 0
    for i = 1:N
        # -βᵢ - cᵢ
        rows[k+1] = i
        cols[k+1] = i
        rows[k+2] = i
        cols[k+2] = i + N
        #  βᵢ - cᵢ
        rows[k+3] = i + N
        cols[k+3] = i
        rows[k+4] = i + N
        cols[k+4] = i + N
        k += 4
    end
    return (rows, cols)
end

function NLPModels.jac_coord!(nlp::FFTNLPModel, x::AbstractVector{T}, vals::AbstractVector{T}) where T
    increment!(nlp, :neval_jac)
    N = nlp.N
    k = 0
    for i = 1:N
        # -βᵢ - cᵢ
        vals[k+1] = -one(T)
        vals[k+2] = -one(T)
        #  βᵢ - cᵢ
        vals[k+3] =  one(T)
        vals[k+4] = -one(T)
        k += 4
    end
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
    hv_b .= M_perpt_M_perp_vec_wei(DFTdim, DFTsize, v[1:n], index_missing)
    hv_c .= 0.0
    return hv
end

function NLPModels.hess_structure!(nlp::FFTNLPModel, rows, cols)
    nβ = nlp.N
    cnt = 1
    for i in 1:nβ
        for j in 1:i
            rows[cnt] = i
            cols[cnt] = j
            cnt += 1
        end
    end
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::FFTNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight::Float64 = 1.0,
)
    increment!(nlp, :neval_hess)
    DFTdim = nlp.parameters.paramf[1]
    DFTsize = nlp.parameters.paramf[2]
    M_perptz = nlp.parameters.paramf[3]
    lambda = nlp.parameters.paramf[4]
    index_missing = nlp.parameters.paramf[5]

    nβ = nlp.N
    H = zeros(nβ, nβ)
    v = zeros(nβ)
    for i in 1:nβ
        fill!(v, 0.0)
        v[i] = 1.0
        H[:, i] .= M_perpt_M_perp_vec_wei(DFTdim, DFTsize, v, index_missing)
    end

    cnt = 1
    for i in 1:nβ
        for j in 1:i
            hess[cnt] = H[i, j]
            cnt += 1
        end
    end

    return hess
end
