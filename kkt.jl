
using LinearAlgebra
using SparseArrays
using Krylov
using MadNLP
using CUDA

#=
    Operator for matrix

    [ MᵀM + Σ₁ + Σ₂    Σ₁ - Σ₂ ]
    [ Σ₁ - Σ₂          Σ₁ + Σ₂ ]

=#

struct CondensedFFTKKT{T, VT, FFT, R, C} <: AbstractMatrix{T}
    nβ::Int
    params::FFTParameters  # for MᵀM
    buf1::VT
    Λ1::VT  # Σ₁ + Σ₂
    Λ2::VT  # Σ₁ - Σ₂
    op::FFT # FFT operator
    buffer_real::R      # Buffer for fft and ifft
    buffer_complex1::C  # Buffer for fft and ifft
    buffer_complex2::C  # Buffer for fft and ifft
end

function CondensedFFTKKT{T, VT}(nlp::FFTNLPModel{T, VT}) where {T, VT}
    nβ = nlp.N
    buf1 = VT(undef, nβ)
    Λ1 = VT(undef, nβ)
    Λ2 = VT(undef, nβ)
    return CondensedFFTKKT{T, VT, typeof(nlp.op), typeof(nlp.buffer_real), typeof(nlp.buffer_complex1)}(nβ, nlp.parameters, buf1, Λ1, Λ2, nlp.op, nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2)
end

Base.size(K::CondensedFFTKKT) = (2*K.nβ, 2*K.nβ)
Base.eltype(K::CondensedFFTKKT{T, VT}) where {T, VT} = T

function LinearAlgebra.mul!(y::AbstractVector, K::CondensedFFTKKT, x::AbstractVector, alpha::Number, beta::Number)
    nβ = K.nβ
    @assert length(y) == length(x) == 2 * nβ
    # Load parameters
    DFTdim = K.params.paramf[1]
    DFTsize = K.params.paramf[2]
    M_perptz = K.params.paramf[3]
    lambda = K.params.paramf[4]
    index_missing = K.params.paramf[5]

    Mβ = K.buf1
    yβ  = view(y, 1:nβ)
    yz  = view(y, nβ+1:2*nβ)
    xβ  = view(x, 1:nβ)
    xz  = view(x, nβ+1:2*nβ)

    # Evaluate Mᵀ M xβ
    Mβ .= M_perpt_M_perp_vec(K.buffer_real, K.buffer_complex1, K.buffer_complex2, K.op, DFTdim, DFTsize, xβ, index_missing)

    yβ .= beta .* yβ .+ alpha .* (Mβ .+ K.Λ1 .* xβ .+ K.Λ2 .* xz)
    yz .= beta .* yz .+ alpha .* (K.Λ2 .* xβ .+ K.Λ1 .* xz)
    return y
end

#=
    Operator for preconditioner

    [ I + Σ₁ + Σ₂      Σ₁ - Σ₂ ]⁻¹  = [ P11   P12 ]
    [ Σ₁ - Σ₂          Σ₁ + Σ₂ ]      [ P12'  P22 ]

=#

struct FFTPreconditioner{T, VT}
    nβ::Int
    P11::VT
    P12::VT
    P22::VT
end

function FFTPreconditioner{T, VT}(nβ) where {T, VT}
    return FFTPreconditioner{T, VT}(
        nβ,
        VT(undef, nβ),
        VT(undef, nβ),
        VT(undef, nβ),
    )
end

Base.size(P::FFTPreconditioner) = (2*P.nβ, 2*P.nβ)
Base.eltype(P::FFTPreconditioner{T, VT}) where {T, VT} = T

function LinearAlgebra.mul!(y::AbstractVector, P::FFTPreconditioner, x::AbstractVector, alpha::Number, beta::Number)
    nβ = P.nβ
    @assert length(y) == length(x) == 2 * nβ
    yβ  = view(y, 1:nβ)
    yz  = view(y, nβ+1:2*nβ)
    xβ  = view(x, 1:nβ)
    xz  = view(x, nβ+1:2*nβ)

    yβ .= beta .* yβ .+ alpha .* (P.P11 .* xβ .+ P.P12 .* xz)
    yz .= beta .* yz .+ alpha .* (P.P12 .* xβ .+ P.P22 .* xz)

    return y
end

#=
    FFTKKTSystem
=#

struct FFTKKTSystem{T, VI, VT, MT, LS} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    nlp::FFTNLPModel
    # Operators
    K::MT
    P::FFTPreconditioner{T, VT}
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    ind_lb::VI
    ind_ub::VI
    # Buffers
    z1::VT           # dimension nβ
    z2::VT           # dimension 2 * nβ
    linear_solver::LS
end

function MadNLP.create_kkt_system(
    ::Type{FFTKKTSystem},
    cb::MadNLP.AbstractCallback{T, VT},
    ind_cons,
    linear_solver;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
) where {T, VT}
    # Load original model
    nlp = cb.nlp
    nβ = nlp.N
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)
    n_ineq = length(ind_cons.ind_ineq)

    # Number of variables, including slacks
    n = NLPModels.get_nvar(nlp) + n_ineq
    # Number of constraints
    m = NLPModels.get_ncon(nlp)

    pr_diag = VT(undef, n)
    du_diag = VT(undef, m)
    reg     = VT(undef, n)
    l_diag  = VT(undef, nlb)
    u_diag  = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    linear_solver = Krylov.CgSolver(2*nβ, 2*nβ, VT)

    z1 = VT(undef, nβ)
    z2 = VT(undef, 2*nβ)

    K = CondensedFFTKKT{T, VT}(nlp)
    P = FFTPreconditioner{T, VT}(nβ)
    VI = Vector{Int}

    return FFTKKTSystem{T, VI, VT, typeof(K), typeof(linear_solver)}(
        nlp, K, P,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ind_cons.ind_lb, ind_cons.ind_ub,
        z1, z2,
        linear_solver,
    )
end

MadNLP.num_variables(kkt::FFTKKTSystem) = 2*kkt.nlp.N
MadNLP.get_hessian(kkt::FFTKKTSystem) = nothing
MadNLP.get_jacobian(kkt::FFTKKTSystem) = nothing

# Dirty wrapper to MadNLP's linear solver
MadNLP.is_inertia(::Krylov.CgSolver) = true
MadNLP.inertia(::Krylov.CgSolver) = (0, 0, 0)
MadNLP.introduce(::Krylov.CgSolver) = "CG"
MadNLP.improve!(::Krylov.CgSolver) = true
MadNLP.factorize!(::Krylov.CgSolver) = nothing

MadNLP.is_inertia_correct(kkt::FFTKKTSystem, p, n, z) = true

Base.eltype(kkt::FFTKKTSystem{T}) where T = T

function Base.size(kkt::FFTKKTSystem)
    n_lb = length(kkt.l_diag)
    n_ub = length(kkt.u_diag)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    N = n + 2*m + n_lb + n_ub
    return (N, N)
end

function MadNLP.initialize!(kkt::FFTKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    return
end

# Don't evaluate Jacobian
function MadNLP.eval_jac_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::FFTKKTSystem,
    x::MadNLP.PrimalVector{T},
) where T
    return
end

# Don't evaluate Hessian
function MadNLP.eval_lag_hess_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::FFTKKTSystem,
    x::MadNLP.PrimalVector{T},
    l::AbstractVector{T};
    is_resto=false,
) where T
    return
end

function MadNLP.mul!(y::VT, kkt::FFTKKTSystem, x::VT, alpha::Number, beta::Number) where VT <: MadNLP.AbstractKKTVector
    nlp = kkt.nlp
    # FFT parameters
    DFTdim = nlp.parameters.paramf[1]
    DFTsize = nlp.parameters.paramf[2]
    M_perptz = nlp.parameters.paramf[3]
    lambda = nlp.parameters.paramf[4]
    index_missing = nlp.parameters.paramf[5]

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)

    _x = MadNLP.full(x)
    _y = MadNLP.full(y)

    nβ = nlp.N
    Mβ = kkt.z1

    yβ  = view(_y, 1:nβ)
    yz  = view(_y, nβ+1:2*nβ)
    ys1 = view(_y, 2*nβ+1:3*nβ)
    ys2 = view(_y, 3*nβ+1:4*nβ)
    yy1 = view(_y, 4*nβ+1:5*nβ)
    yy2 = view(_y, 5*nβ+1:6*nβ)

    xβ  = view(_x, 1:nβ)
    xz  = view(_x, nβ+1:2*nβ)
    xs1 = view(_x, 2*nβ+1:3*nβ)
    xs2 = view(_x, 3*nβ+1:4*nβ)
    xy1 = view(_x, 4*nβ+1:5*nβ)
    xy2 = view(_x, 5*nβ+1:6*nβ)

    # Evaluate (MᵀM) * xβ
    Mβ .= M_perpt_M_perp_vec(kkt.K.buffer_real, kkt.K.buffer_complex1, kkt.K.buffer_complex2, kkt.K.op, DFTdim, DFTsize, xβ, index_missing)
    yβ .= beta .* yβ .+ alpha .* (Mβ .- xy1 .+ xy2)
    yz .= beta .* yz .- alpha .* (xy1 .+ xy2)
    ys1 .= beta .* ys1 .- alpha .* xy1
    ys2 .= beta .* ys2 .- alpha .* xy2
    yy1 .= beta .* yy1 .+ alpha .* (.-xβ .- xz .- xs1)
    yy2 .= beta .* yy2 .+ alpha .* (xβ .- xz .- xs2)

    MadNLP._kktmul!(y, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)

    return y
end

function MadNLP.jtprod!(
    y::AbstractVector,
    kkt::FFTKKTSystem{T, VI, VT, MT},
    x::AbstractVector,
) where {T, VI, VT, MT}
    nlp = kkt.nlp
    nβ = nlp.N

    xy1 = view(x, 1:nβ)
    xy2 = view(x, nβ+1:2*nβ)

    yβ = view(y, 1:nβ)
    yz = view(y, nβ+1:2*nβ)
    ys1 = view(y, 2*nβ+1:3*nβ)
    ys2 = view(y, 3*nβ+1:4*nβ)

    yβ .= .-xy1 .+ xy2
    yz .= .-xy1 .- xy2
    ys1 .= .-xy1
    ys2 .= .-xy2

    return y
end

function MadNLP.compress_jacobian!(kkt::FFTKKTSystem)
    return
end

function MadNLP.compress_hessian!(kkt::FFTKKTSystem)
    return
end

function MadNLP.build_kkt!(kkt::FFTKKTSystem)
    nlp = kkt.nlp
    nβ = nlp.N
    # Assemble preconditioner
    Σ1 = view(kkt.pr_diag, 2*nβ+1:3*nβ)
    Σ2 = view(kkt.pr_diag, 3*nβ+1:4*nβ)

    Λ1 = kkt.K.Λ1
    Λ2 = kkt.K.Λ2
    Minv = kkt.z1

    # Update values in CondensedFFTKKT
    Λ1 .= Σ1 .+ Σ2
    Λ2 .= Σ1 .- Σ2

    # Update values in FFTPreconditioner
    S = kkt.P.P22
    Minv .= 1.0 ./ (1.0 .+ Λ1)
    S .= 1.0 ./ (Λ1 .- Λ2.^2 .* Minv)
    kkt.P.P12 .= .-Λ2 .* S .* Minv
    kkt.P.P11 .= Minv .+ Minv .* Λ2 .* S .* Λ2 .* Minv
    return
end

function MadNLP.solve!(kkt::FFTKKTSystem, w::MadNLP.AbstractKKTVector)
    nlp = kkt.nlp
    nβ = nlp.N
    # Build reduced KKT vector.
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    Σ1 = view(kkt.pr_diag, 2*nβ+1:3*nβ)
    Σ2 = view(kkt.pr_diag, 3*nβ+1:4*nβ)

    # Buffers
    b = kkt.z2
    bβ = view(b, 1:nβ)
    bz = view(b, 1+nβ:2*nβ)

    # Unpack right-hand-side
    _w = MadNLP.full(w)
    w1 = view(_w, 1:nβ)         # / β
    w2 = view(_w, nβ+1:2*nβ)    # / z
    w3 = view(_w, 2*nβ+1:3*nβ)  # / s1
    w4 = view(_w, 3*nβ+1:4*nβ)  # / s2
    w5 = view(_w, 4*nβ+1:5*nβ)  # / y1
    w6 = view(_w, 5*nβ+1:6*nβ)  # / y2

    # Assemble right-hand-side
    bβ .= w1 .- w3 .+ w4 .- Σ1 .* w5 .+ Σ2 .* w6
    bz .= w2 .- w3 .- w4 .- Σ1 .* w5 .- Σ2 .* w6

    # Solve with CG
    Krylov.solve!(kkt.linear_solver, kkt.K, b, M=kkt.P, atol=1e-12, rtol=0.0, verbose=0)
    x = Krylov.solution(kkt.linear_solver)

    # Unpack solution
    w1 .= x[1:nβ]                              # / x
    w2 .= x[(nβ+1):(2*nβ)]                     # / z
    w5 .= .-(w3 .+ Σ1 .* (w1 .+ w2 .+ w5))     # / y1
    w6 .= .-(w4 .+ Σ2 .* (.-w1 .+ w2 .+ w6))   # / y2
    w3 .= (w3 .+ w5) ./ Σ1                     # / s1
    w4 .= (w4 .+ w6) ./ Σ2                     # / s2

    MadNLP.finish_aug_solve!(kkt, w)
    return true
end

function MadNLP.factorize_wrapper!(
    solver::MadNLP.MadNLPSolver{T,Vector{T},Vector{Int},KKT}
    ) where {T,KKT<:FFTKKTSystem{T,Vector{Int},Vector{T}}}
    MadNLP.build_kkt!(solver.kkt)
    # No factorization needed
    return true
end

function MadNLP.factorize_wrapper!(
    solver::MadNLP.MadNLPSolver{T,CuVector{T},CuVector{Int},KKT}
    ) where {T,KKT<:FFTKKTSystem{T,CuVector{Int},CuVector{T}}}
    MadNLP.build_kkt!(solver.kkt)
    # No factorization needed
    return true
end

#=
    Uncomment to have custom control on iterative refinement
=#

# function MadNLP.solve_refine_wrapper!(
#     d,
#     solver::MadNLP.MadNLPSolver{T,Vector{T},Vector{Int},KKT},
#     p,
#     w,
# ) where {T,KKT<:FFTKKTSystem{T,Vector{Int},Vector{T}}}
#     result = false
#     kkt = solver.kkt
#     copyto!(MadNLP.full(d), MadNLP.full(p))
#     MadNLP.solve!(kkt, d)

#     return true
# end

