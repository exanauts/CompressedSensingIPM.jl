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

mutable struct FFTNLPModel{T,VT,FFT,R,C} <: AbstractNLPModel{T,VT}
    meta::NLPModelMeta{T,VT}
    parameters::FFTParameters
    N::Int
    counters::Counters
    op::FFT
    buffer_real::R
    buffer_complex1::C
    buffer_complex2::C
    rdft::Bool
    fft_timer::Base.RefValue{Float64}
    mapping_timer::Base.RefValue{Float64}
    krylov_solver::Symbol
    preconditioner::Bool
end

function FFTNLPModel{T,VT}(parameters::FFTParameters; krylov_solver::Symbol=:cg, rdft::Bool=false, preconditioner::Bool=true) where {T,VT}
    DFTdim = parameters.paramf[1]   # problem size (1, 2, 3)
    DFTsize = parameters.paramf[2]  # problem dimension
    N = prod(DFTsize)
    nvar = 2 * N
    ncon = 2 * N
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
        nnzh = 0, # div(N * (N + 1), 2),
        minimize = true,
        islp = false,
        name = "CompressedSensing-$(DFTdim)D",
    )

    # FFT operator
    A_vec = VT(undef, N)
    A = reshape(A_vec, DFTsize)
    buffer_real = A
    if rdft == true
        op = plan_rfft(A)
        M1 = (DFTsize[1] ÷ 2)
        if DFTdim == 1
            buffer_complex1 = Complex{T}.(A[1:M1+1])
        elseif DFTdim == 2
            buffer_complex1 = Complex{T}.(A[1:M1+1,:])
        else
            buffer_complex1 = Complex{T}.(A[1:M1+1,:,:])
        end
        buffer_complex2 = buffer_complex1
    else
        op = plan_fft(A)
        buffer_complex1 = Complex{T}.(A)
        buffer_complex2 = copy(buffer_complex1)
    end
    fft_timer = Ref{Float64}(0.0)
    mapping_timer = Ref{Float64}(0.0)
    return FFTNLPModel(meta, parameters, N, Counters(), op, buffer_real, buffer_complex1,
                       buffer_complex2, rdft, fft_timer, mapping_timer, krylov_solver, preconditioner)
end

function NLPModels.cons!(nlp::FFTNLPModel, x::AbstractVector, c::AbstractVector)
    increment!(nlp, :neval_cons)
    N = nlp.N
    xβ = view(x, 1:N)
    xc = view(x, N+1:2*N)
    cβ = view(c, 1:N)
    cc = view(c, N+1:2*N)
    cβ .= .- xβ .- xc  # -βᵢ - cᵢ for 1 ≤ i ≤ N
    cc .=    xβ .- xc  #  βᵢ - cᵢ for N+1 ≤ i ≤ 2N
    return c
end

function NLPModels.obj(nlp::FFTNLPModel, x::AbstractVector)
    increment!(nlp, :neval_obj)
    DFTdim = nlp.parameters.paramf[1]
    DFTsize = nlp.parameters.paramf[2]
    M_perptz = nlp.parameters.paramf[3]
    lambda = nlp.parameters.paramf[4]
    index_missing = nlp.parameters.paramf[5]
    # Mt = nlp.parameters.paramf[6]

    fft_val = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, DFTdim, DFTsize, x, index_missing, nlp.fft_timer, nlp.mapping_timer; nlp.rdft)
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
    res = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, DFTdim, DFTsize, beta, index_missing, nlp.fft_timer, nlp.mapping_timer; nlp.rdft)
    g_b .= res .- M_perptz
    fill!(g_c, lambda)
    return g
end
