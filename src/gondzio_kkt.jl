#=
    Operator for matrix

    [MᵀM + P^{-1}W_p    - MᵀM]
    [MᵀM      MᵀM + Q^{-1}W_q]

    ...
=#

struct CondensedGondzioKKTSystem{T,VT,NLP} <: AbstractMatrix{T}
    nlp::NLP
    buf1::VT
    InvP_Wp::VT  # P^{-1}W_p
    InvQ_Wq::VT  # Q^{-1}W_q
end

function CondensedGondzioKKTSystem{T,VT}(nlp::GondzioNLPModel{T,VT}) where {T,VT}
    buf1 = VT(undef, nlp.nβ)
    InvP_Wp = VT(undef, nlp.nβ)
    InvQ_Wq = VT(undef, nlp.nβ)
    NLP = typeof(nlp)
    return CondensedGondzioKKTSystem{T,VT,NLP}(nlp, buf1, InvP_Wp, InvQ_Wq)
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
    diff_x = K.buf1
    diff_x .= p_x .- q_x
    y_diff = M_perpt_M_perp_vec(nlp.op_fft, diff_x)



    # p_y .= beta .* yβ .+ alpha .* (Mβ .+ K.InvP_Wp .* xβ .+ K.InvQ_Wq .* xz)
    # yz .= beta .* yz .+ alpha .* (K.InvQ_Wq .* xβ .+ K.InvP_Wp .* xz)
    p_y .= K.InvP_Wp .* p_x .+ y_diff
    q_y .=  K.InvQ_Wq .* p_y .- y_diff

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

    # yβ .= beta .* yβ .+ alpha .* (P.P11 .* xβ .+ P.P12 .* xz)
    # yz .= beta .* yz .+ alpha .* (P.P12 .* xβ .+ P.P22 .* xz)

    y_p .= P.P11 .* x_p + P.P12 .* x_q
    y_q .= P.P12 .* x_p + P.P22 .* x_q

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
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
) where {T, VT}
    # Load original model
    nlp = cb.nlp
    nβ = nlp.nβ
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

    
    workspace = Krylov.krylov_workspace(Val(nlp.krylov_solver), 2*nβ, 2*nβ, VT)

    buffer1 = VT(undef, nβ)
    buffer2 = VT(undef, 2*nβ)

    K = CondensedGondzioKKTSystem{T, VT}(nlp)
    P = GondzioPreconditioner{T, VT}(nlp.nβ)
    VI = Vector{Int}

    return GondzioKKTSystem{T, VI, VT, typeof(K), typeof(workspace), typeof(nlp)}(
        nlp, K, P,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ind_cons.ind_lb, ind_cons.ind_ub,
        buffer1, buffer2,
        workspace, Int[], Float64[],
    )
end

MadNLP.num_variables(kkt::GondzioKKTSystem) = 2*kkt.nlp.nβ
MadNLP.get_hessian(kkt::GondzioKKTSystem) = nothing
MadNLP.get_jacobian(kkt::GondzioKKTSystem) = nothing

# Dirty wrapper to MadNLP's linear solver
# MadNLP.is_inertia(::Krylov.KrylovWorkspace) = true
# MadNLP.inertia(::Krylov.KrylovWorkspace) = (0, 0, 0)
# MadNLP.introduce(::Krylov.KrylovWorkspace) = "Krylov"
# MadNLP.improve!(::Krylov.KrylovWorkspace) = true
# MadNLP.factorize!(::Krylov.KrylovWorkspace) = nothing

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
    buffer1 = kkt.buffer1

    # FFT parameters
    DFTdim = parameters.DFTdim
    DFTsize = parameters.DFTsize
    lambda = parameters.lambda
    index_missing = parameters.index_missing

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)

    _x = MadNLP.full(x)
    _y = MadNLP.full(y)

    y_x = view(_y, 1:2*nβ)
    y_r = view(_y, 2*nβ+1:2*nβ+m)
    y_y = view(_y, 2*nβ+m+1:2*nβ+2*m)
    y_w = view(_y, 2*nβ+2*m+1:4*nβ+2*m)

    y_p = view(_y, 1:nβ)
    y_q = view(_y, nβ+1:2*nβ)
    y_wp = view(_y, 2*nβ+2*m+1:3*nβ+2*m)
    y_wq = view(_y, 3*nβ+2*m+1:4*nβ+2*m)

    x_x = view(_x, 1:2*nβ)
    x_r = view(_x, 2*nβ+1:2*nβ+m)
    x_y = view(_x, 2*nβ+m+1:2*nβ+2*m)
    x_w = view(_x, 2*nβ+2*m+1:4*nβ+2*m)

    x_p = view(_x, 1:nβ)
    x_q = view(_x, nβ+1:2*nβ)
    x_wp = view(_x, 2*nβ+2*m+1:3*nβ+2*m)
    x_wq = view(_x, 3*nβ+2*m+1:4*nβ+2*m)

    buffer1 .= x_q .- x_p
    tmp = M_perpt_z(kkt.nlp.op_fft, x_y)
    y_p .=   tmp .- x_wp
    y_q .= .-tmp .- x_wq
    y_r .= x_r .- x_y
    y_y .= M_perp_beta(kkt.nlp.op_fft, buffer1) .- x_r
    y_w .= diag(l_lower) .* x_x .+ diag(l_lower) .* x_w
    return y
end

function MadNLP.jtprod!(
    y::AbstractVector,
    kkt::GondzioKKTSystem{T, VI, VT, MT},
    x::AbstractVector,
) where {T, VI, VT, MT}
    nlp = kkt.nlp
    nβ = nlp.nβ
    tmp = M_perpt_z(kkt.nlp.op_fft, x)

    y1 = view(y, 1:nβ)
    y2 = view(y, nβ+1:2*nβ)

    y1 .= -tmp
    y2 .= tmp
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
    Buff1 = kkt.buffer1
    # Assemble preconditioner

    p = view(kkt.pr_diag, 1:nβ)
    q = view(kkt.pr_diag, nβ+1:2*nβ)
    Wp = view(kkt.l_lower, 1:nβ)
    Wq = view(kkt.l_lower, nβ+1:2*nβ)

    InvP_Wp  = kkt.K.InvP_Wp
    InvQ_Wq = kkt.K.InvQ_Wq

    InvP_Wp .= Wp./p
    InvQ_Wq .= Wq./q

    # Update values in Gondzio Preconditioner
    kkt.P.P22 .= 1.0 ./ ((1.0 .+ InvQ_Wq) .- 1.0 ./ (1.0 .+ InvP_Wp))
    S = kkt.P.P22

    Buff1 = 1.0./(1 .+ InvP_Wp)

    kkt.P.P12 .= Buff1 .* S
    kkt.P.P11 .= Buff1 .+ Buff1.*S.*Buff1
    
    

    return
end

function MadNLP.solve!(kkt::GondzioKKTSystem, w::MadNLP.AbstractKKTVector)
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
    Buf1 = kkt.buffer1
    b_rhs = kkt.buffer2

    # Unpack right-hand-side
    _x = MadNLP.full(w)
    x_x = view(_x, 1:2*nβ)
    x_r = view(_x, 2*nβ+1:2*nβ+m)
    x_y = view(_x, 2*nβ+m+1:2*nβ+2*m)
    x_w = view(_x, 2*nβ+2*m+1:4*nβ+2*m)

    p = view(_x, 1:nβ)
    q = view(_x, nβ+1:2*nβ)

    #assemble RHS

    b_rhs1 = view(b_rhs, 1:nβ)
    b_rhs2 = view(b_rhs, nβ+1:2*nβ)

    Buf1 = M_perpt_z(kkt.nlp.op_fft, nlp.M_perpt_z0) - M_perpt_M_perp_vec(kkt.nlp.op_fft, p .- q)

    b_rhs1 .= Buf1 .- lambda + reg./p
    b_rhs2 .= -Buf1 .- lambda + reg./q

    P = kkt.nlp.preconditioner ? kkt.P : I
    Krylov.krylov_solve!(kkt.linear_solver, kkt.K, b_rhs, M=P, atol=1e-12, rtol=0.0, verbose=0)
    x = Krylov.solution(kkt.linear_solver)
    push!(kkt.krylov_iterations, kkt.linear_solver |> Krylov.iteration_count)
    push!(kkt.krylov_timer, kkt.linear_solver |> Krylov.elapsed_time)
    

    # Unpack solution
    x_deltax = kkt.buffer2
    x_deltax .= x .+ x_x
    p1 = view(x_deltax, 1:nβ)
    p2 = view(x_deltax, nβ +1:2*nβ)
    # x_x .= view(x, 1:2*nβ)                        # / x   
    Buf1 .= p1 .- p2 
    deltay = - M_perp_beta(kkt.nlp.op_fft, Buf1) .+  nlp.M_perpt_z0 .- x_y
    deltar = deltay .+ x_y .- x_r
    deltaw = -x_w./x_x .* x .+ mu./x_x .- x_w

    x_x .= x
    x_r .= deltar
    x_y .= deltay
    x_w .= deltaw

    # w5 .= .-(w3 .+ Σ1 .* (w1 .+ w2 .+ w5))     # / y1
    # w6 .= .-(w4 .+ Σ2 .* (.-w1 .+ w2 .+ w6))   # / y2
    # w3 .= (w3 .+ w5) ./ Σ1                     # / s1
    # w4 .= (w4 .+ w6) ./ Σ2                     # / s2

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
