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

function CondensedGondzioKKTSystem{T,VT}(nlp::FFTNLPModel{T,VT}) where {T,VT}
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
MadNLP.is_inertia(::Krylov.KrylovWorkspace) = true
MadNLP.inertia(::Krylov.KrylovWorkspace) = (0, 0, 0)
MadNLP.introduce(::Krylov.KrylovWorkspace) = "Krylov"
MadNLP.improve!(::Krylov.KrylovWorkspace) = true
MadNLP.factorize!(::Krylov.KrylovWorkspace) = nothing

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

    # FFT parameters
    DFTdim = parameters.DFTdim
    DFTsize = parameters.DFTsize
    lambda = parameters.lambda
    index_missing = parameters.index_missing

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)

    _x = MadNLP.full(x)
    _y = MadNLP.full(y)

    temp = kkt.z1 #I know this size is not proper, this has to be changed

    yp  = view(_y, 1:nβ)
    yq  = view(_y, nβ+1:2*nβ)
    y_con1 = view(_y, 2*nβ+1:2*nβ+m)
    y_con2 = view(_y, 2*nβ+m+1:2*nβ+2*m)
    yy1 = view(_y, 2*nβ+2*m+1:3*nβ+2*m)
    yy2 = view(_y, 3*nβ+2*m+1:4*nβ+2*m)

    xp  = view(_y, 1:nβ)
    xq  = view(_y, nβ+1:2*nβ)
    x_con1 = view(_y, 2*nβ+1:2*nβ+m)
    x_con2 = view(_y, 2*nβ+m+1:2*nβ+2*m)
    xy1 = view(_y, 2*nβ+2*m+1:3*nβ+2*m)
    xy2 = view(_y, 3*nβ+2*m+1:4*nβ+2*m)

    temp .= M_perpt_z(kkt.nlp.op_fft, x_con2)
    yp .= -temp .- xy1
    yq .= -temp .- xy2
    y_con1 .= x_con1 .- x_con2

    temp = kkt.z1 #Same comment about the size
    temp2 = kkt.z2
    temp .= xp .- xq

    temp2 .= -M_perp(kkt.nlp.op_fft, temp)

    y_con2 .= temp2 .- x_con1
    # Have to do W \Deltax + X \Delta w


    return y
end

function MadNLP.jtprod!(
    y::AbstractVector,
    kkt::GondzioKKTSystem{T, VI, VT, MT},
    x::AbstractVector,
) where {T, VI, VT, MT}
    nlp = kkt.nlp
    nβ = nlp.nβ

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

function MadNLP.compress_jacobian!(kkt::GondzioKKTSystem)
    return
end

function MadNLP.compress_hessian!(kkt::GondzioKKTSystem)
    return
end

function MadNLP.build_kkt!(kkt::GondzioKKTSystem)
    ...
    return
end

function MadNLP.solve!(kkt::GondzioKKTSystem, w::MadNLP.AbstractKKTVector)
    ...
    return true
end

function MadNLP.factorize_wrapper!(
    solver::MadNLP.MadNLPSolver{T,VT,IT,KKT}
    ) where {T, VT<:AbstractVector{T}, IT<:AbstractVector{Int}, KKT<:GondzioKKTSystem{T,IT,VT}}
    MadNLP.build_kkt!(solver.kkt)
    # No factorization needed
    return true
end
