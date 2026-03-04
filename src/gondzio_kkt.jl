#=
    Operator for matrix

    [MᵀM + P^{-1}W_p     - MᵀM]
    [-MᵀM      MᵀM + Q^{-1}W_q]

    ...
=#

struct CondensedGondzioKKTSystem{T,VT,NLP} <: AbstractMatrix{T}
    nlp::NLP
    buffer::VT
    InvP_Wp::VT  # P^{-1}W_p
    InvQ_Wq::VT  # Q^{-1}W_q
end

function CondensedGondzioKKTSystem{T,VT}(nlp::GondzioNLPModel{T,VT}) where {T,VT}
    buffer = VT(undef, nlp.nβ)  ; fill!(buffer, zero(T))
    InvP_Wp = VT(undef, nlp.nβ) ; fill!(InvP_Wp, zero(T))
    InvQ_Wq = VT(undef, nlp.nβ) ; fill!(InvQ_Wq, zero(T))
    NLP = typeof(nlp)
    return CondensedGondzioKKTSystem{T,VT,NLP}(nlp, buffer, InvP_Wp, InvQ_Wq)
end

Base.size(K::CondensedGondzioKKTSystem) = (2*K.nlp.nβ, 2*K.nlp.nβ)
Base.eltype(K::CondensedGondzioKKTSystem{T, VT}) where {T, VT} = T

function LinearAlgebra.mul!(y::AbstractVector, K::CondensedGondzioKKTSystem, x::AbstractVector, alpha::Number, beta::Number)
    nlp = K.nlp
    nβ = nlp.nβ
    parameters = nlp.parameters
    @assert length(y) == length(x) == 2 * nβ

    # Load parameters
    DFTdim = parameters.DFTdim
    DFTsize = parameters.DFTsize
    lambda = parameters.lambda
    index_missing = parameters.index_missing

    p_y = view(y, 1:nβ)
    q_y = view(y, nβ+1:2*nβ)
    p_x = view(x, 1:nβ)
    q_x = view(x, nβ+1:2*nβ)

    # Evaluate Mᵀ M p_x and Mᵀ M q_x and use it to construct the product
    diff_x = K.buffer
    diff_x .= p_x .- q_x
    y_diff = M_perpt_M_perp_vec(nlp.op_fft, diff_x)

    p_y .= alpha .* (K.InvP_Wp .* p_x .+ y_diff) .+ beta .* p_y
    q_y .= alpha .* (K.InvQ_Wq .* q_x .- y_diff) .+ beta .* q_y
    return y
end

#=
    Operator for preconditioner

    [ I + P^{-1}W_p       -I ]⁻¹  = [ P11   P12 ]
    [ -I        I + Q^{-1}W_q]      [ P12'  P22 ]

=#

struct GondzioPreconditioner{T, VT}
    nβ::Int
    P11::VT
    P12::VT
    P22::VT
end

function GondzioPreconditioner{T, VT}(nβ) where {T, VT}
    return GondzioPreconditioner{T, VT}(
        nβ,
        VT(undef, nβ),
        VT(undef, nβ),
        VT(undef, nβ),
    )
end

Base.size(P::GondzioPreconditioner) = (2*P.nβ, 2*P.nβ)
Base.eltype(P::GondzioPreconditioner{T, VT}) where {T, VT} = T

function LinearAlgebra.mul!(y::AbstractVector, P::GondzioPreconditioner, x::AbstractVector, alpha::Number, beta::Number)
    nβ = P.nβ
    @assert length(y) == length(x) == 2 * nβ
    y_p  = view(y, 1:nβ)
    y_q  = view(y, nβ+1:2*nβ)
    x_p  = view(x, 1:nβ)
    x_q  = view(x, nβ+1:2*nβ)

    y_p .= alpha .* (P.P11 .* x_p + P.P12 .* x_q) .+ beta .* y_p
    y_q .= alpha .* (P.P12 .* x_p + P.P22 .* x_q) .+ beta .* y_q
    return y
end

#=
    GondzioKKTSystem
=#

struct GondzioKKTSystem{T, VI, VT, MT, LS, NLP} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    nlp::NLP
    # Operators
    K::MT
    P::GondzioPreconditioner{T, VT}
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
    buffer1::VT  # # dimension nβ
    buffer2::VT  # dimension 2 * nβ
    linear_solver::LS
    krylov_iterations::Vector{Int}
    krylov_timer::Vector{Float64}
end

function MadNLP.create_kkt_system(
    ::Type{GondzioKKTSystem},
    cb::MadNLP.AbstractCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
) where {T, VT}
    # Load original model
    nlp = cb.nlp
    nβ = nlp.nβ
    nlb, nub = length(cb.ind_lb), length(cb.ind_ub)
    n_ineq = length(cb.ind_ineq)

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

    workspace = Krylov.krylov_workspace(Val(nlp.krylov_solver), 2*nβ, 2*nβ, VT)

    buffer1 = VT(undef, nβ)
    buffer2 = VT(undef, 2*nβ)

    K = CondensedGondzioKKTSystem{T, VT}(nlp)
    P = GondzioPreconditioner{T, VT}(nlp.nβ)
    VI = Vector{Int}

    return GondzioKKTSystem{T, VI, VT, typeof(K), typeof(workspace), typeof(nlp)}(
        nlp, K, P,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        cb.ind_lb, cb.ind_ub,
        buffer1, buffer2,
        workspace, Int[], Float64[],
    )
end

MadNLP.num_variables(kkt::GondzioKKTSystem) = 2*kkt.nlp.nβ
MadNLP.get_hessian(kkt::GondzioKKTSystem) = nothing
MadNLP.get_jacobian(kkt::GondzioKKTSystem) = nothing
MadNLP.is_inertia_correct(kkt::GondzioKKTSystem, p, n, z) = true

Base.eltype(kkt::GondzioKKTSystem{T}) where T = T

function Base.size(kkt::GondzioKKTSystem)
    n_lb = length(kkt.l_diag)
    n_ub = length(kkt.u_diag)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    N = n + m + n_lb + n_ub
    return (N, N)
end

function MadNLP.initialize!(kkt::GondzioKKTSystem{T}) where T
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
    kkt::GondzioKKTSystem,
    x::MadNLP.PrimalVector{T},
) where T
    return
end

# Don't evaluate Hessian
function MadNLP.eval_lag_hess_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::GondzioKKTSystem,
    x::MadNLP.PrimalVector{T},
    l::AbstractVector{T};
    is_resto=false,
) where T
    return
end

function MadNLP.mul!(y::VT, kkt::GondzioKKTSystem, x::VT, alpha::Number, beta::Number) where VT <: MadNLP.AbstractKKTVector
    nlp = kkt.nlp
    nβ = nlp.nβ
    parameters = nlp.parameters
    β = kkt.buffer1

    # FFT parameters
    DFTdim = parameters.DFTdim
    DFTsize = parameters.DFTsize
    lambda = parameters.lambda
    index_missing = parameters.index_missing

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)

    _x = MadNLP.full(x)
    _y = MadNLP.full(y)

    # Unpack LHS
    y_p = view(_y, 1:nβ)
    y_q = view(_y, nβ+1:2*nβ)
    y_r = view(_y, 2*nβ+1:2*nβ+m)
    y_y = view(_y, 2*nβ+m+1:2*nβ+2*m)

    # Unpack RHS
    x_p = view(_x, 1:nβ)
    x_q = view(_x, nβ+1:2*nβ)
    x_r = view(_x, 2*nβ+1:2*nβ+m)
    x_y = view(_x, 2*nβ+m+1:2*nβ+2*m)

    β .= x_q .- x_p
    tmp = M_perpt_z(kkt.nlp.op_fft, reshape(x_y, parameters.DFTsize))
    y_p .= .-alpha .* tmp .+ beta .* y_p
    y_q .= alpha .* tmp .+ beta .* y_q
    y_r .= alpha .* (x_r .- x_y) .+ beta .* y_r
    tmp = M_perp_beta(kkt.nlp.op_fft, β)
    y_y .= alpha .* (vec(tmp) .- x_r) .+ beta .* y_y

    MadNLP._kktmul!(y, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)

    return y
end

function MadNLP.jtprod!(
    y::AbstractVector,
    kkt::GondzioKKTSystem{T, VI, VT, MT},
    x::AbstractVector,
) where {T, VI, VT, MT}
    nlp = kkt.nlp
    n = NLPModels.get_nvar(nlp)
    nβ = nlp.nβ

    tmp = M_perpt_z(kkt.nlp.op_fft, reshape(x, nlp.parameters.DFTsize))

    yp = view(y, 1:nβ)
    yq = view(y, nβ+1:2*nβ)
    yr = view(y, 2*nβ+1:n)

    yp .= .-tmp
    yq .= tmp
    yr .= .-x
    return y
end

function MadNLP.compress_jacobian!(kkt::GondzioKKTSystem)
    return
end

function MadNLP.compress_hessian!(kkt::GondzioKKTSystem)
    return
end

function MadNLP.build_kkt!(kkt::GondzioKKTSystem)
    nlp = kkt.nlp
    nβ = nlp.nβ
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)

    # Assemble preconditioner
    reg_p = view(kkt.pr_diag, 1:nβ)
    reg_q = view(kkt.pr_diag, nβ+1:2*nβ)

    InvP_Wp = kkt.K.InvP_Wp
    InvQ_Wq = kkt.K.InvQ_Wq

    InvP_Wp .= 1.0 ./ (1.0 .+ reg_p)

    # Update values in Gondzio Preconditioner
    S = kkt.P.P22
    S .= 1.0 ./ (1.0 .+ reg_q .- InvP_Wp)

    kkt.P.P12 .= InvP_Wp .* S
    kkt.P.P11 .= InvP_Wp .+ InvP_Wp .* S .* InvP_Wp

    # Update values in operator
    InvP_Wp .= reg_p
    InvQ_Wq .= reg_q
    return
end

function MadNLP.solve_kkt!(kkt::GondzioKKTSystem, w::MadNLP.AbstractKKTVector)
    nlp = kkt.nlp
    nβ = nlp.nβ

    # Build reduced KKT vector.
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    #Parameters
    parameters = nlp.parameters
    lambda = parameters.lambda
    mu = kkt.reg
    m = NLPModels.get_ncon(nlp)

    # Buffers
    buffer1 = kkt.buffer1
    rhs = kkt.buffer2

    # Unpack right-hand-side
    _w = MadNLP.full(w)
    w_x = view(_w, 1:2*nβ)               # / x
    w_p = view(_w, 1:nβ)                 # / p
    w_q = view(_w, nβ+1:2*nβ)            # / q
    w_r = view(_w, 2*nβ+1:2*nβ+m)        # / r
    w_y = view(_w, 2*nβ+m+1:2*nβ+2*m)    # / y

    # Assemble right-hand side
    # [  X⁻¹Z   0  -Uᵀ] [ Δx ]   [ r₁ ]
    # [  0      I  -I ] [ Δr ] = [ r₂ ]
    # [ -U     -I   0 ] [ Δy ]   [ r₃ ]
    #
    # If we eliminate Δr:
    #
    # Δr = Δy + r₂
    #
    # [  X⁻¹Z  -Uᵀ ] [ Δx ] = [ r₁ + X⁻¹r₄]
    # [ -U     -I  ] [ Δy ]   [ r₂ + r₃   ]
    #
    # If we eliminate Δy:
    #
    #              Δy = -UΔx - r₂ - r₃
    # (X⁻¹Z + UᵀU) Δx = r₁ - Uᵀ(r₂ + r₃)
    buffer3 = w_r + w_y  # need a dedicated buffer of size ncon!
    tmp = M_perpt_z(kkt.nlp.op_fft, reshape(buffer3, nlp.parameters.DFTsize))
    rhs1 = view(rhs, 1:nβ)
    rhs2 = view(rhs, nβ+1:2*nβ)
    rhs1 .= w_p .- tmp
    rhs2 .= w_q .+ tmp

    # Solve with the Krylov solver (CG by default)
    P = kkt.nlp.preconditioner ? kkt.P : I
    Krylov.krylov_solve!(kkt.linear_solver, kkt.K, rhs, M=P, atol=1e-12, rtol=0.0, verbose=0)
    w_x .= Krylov.solution(kkt.linear_solver)
    push!(kkt.krylov_iterations, kkt.linear_solver |> Krylov.iteration_count)
    push!(kkt.krylov_timer, kkt.linear_solver |> Krylov.elapsed_time)

    # Unpack solution
    #
    # -IΔr - UΔx = r₃ => Δr = -r₃ - UΔx
    #  IΔr - IΔy = r₂ => Δy = Δr - r₂

    copy_w_r = copy(w_r)
    buffer1 .= w_p .- w_q
    UΔx = M_perp_beta(kkt.nlp.op_fft, buffer1)
    w_r .= .-w_y .- vec(UΔx)      # Δr = -r₃ - UΔx
    w_y .= w_r .- copy_w_r        # Δy = Δr - r₂

    MadNLP.finish_aug_solve!(kkt, w)
    return true
end

function MadNLP.factorize_wrapper!(
    solver::MadNLP.MadNLPSolver{T,VT,IT,KKT}
    ) where {T, VT<:AbstractVector{T}, IT<:AbstractVector{Int}, KKT<:GondzioKKTSystem{T,IT,VT}}
    MadNLP.build_kkt!(solver.kkt)
    # No factorization needed
    return true
end
