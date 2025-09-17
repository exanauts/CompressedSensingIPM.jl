#=
    Operator for matrix

    ...
=#

struct CondensedGondzioKKTSystem{T,VT,NLP} <: AbstractMatrix{T}
    ...
end

function CondensedGondzioKKTSystem{T,VT}(nlp::FFTNLPModel{T,VT}) where {T,VT}
    ...
end

Base.size(K::CondensedGondzioKKTSystem) = (2*K.nlp.nβ, 2*K.nlp.nβ)
Base.eltype(K::CondensedGondzioKKTSystem{T, VT}) where {T, VT} = T

function LinearAlgebra.mul!(y::AbstractVector, K::CondensedGondzioKKTSystem, x::AbstractVector, alpha::Number, beta::Number)
    ...
    return y
end

#=
    Operator for preconditioner

    ...
=#

struct GondzioPreconditioner{T, VT}
    ...
end

function GondzioPreconditioner{T, VT}(nβ) where {T, VT}
    ...
end

Base.size(P::GondzioPreconditioner) = (..., ...)
Base.eltype(P::GondzioPreconditioner{T, VT}) where {T, VT} = T

function LinearAlgebra.mul!(y::AbstractVector, P::GondzioPreconditioner, x::AbstractVector, alpha::Number, beta::Number)
    ...
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
    z1::VT  # dimension ...
    z2::VT  # dimension ...
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
    ...

    # Number of variables, including slacks
    n = ...
    # Number of constraints
    m = ...

    pr_diag = ...
    du_diag = ...
    reg     = ...
    l_diag  = ...
    u_diag  = ...
    l_lower = ...
    u_lower = ...

    workspace = Krylov.krylov_workspace(Val(nlp.krylov_solver), ..., ..., VT)

    z1 = ...
    z2 = ...

    K = CondensedGondzioKKTSystem{T, VT}(nlp)
    P = GondzioPreconditioner{T, VT}(nlp.nβ)
    VI = Vector{Int}

    return GondzioKKTSystem{T, VI, VT, typeof(K), typeof(workspace), typeof(nlp)}(
        nlp, K, P,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ind_cons.ind_lb, ind_cons.ind_ub,
        z1, z2,
        workspace, Int[], Float64[],
    )
end

MadNLP.num_variables(kkt::GondzioKKTSystem) = ...
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
    ...
    return (N, N)
end

function MadNLP.initialize!(kkt::GondzioKKTSystem{T}) where T
    ...
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

    ...
    return y
end

function MadNLP.jtprod!(
    y::AbstractVector,
    kkt::GondzioKKTSystem{T, VI, VT, MT},
    x::AbstractVector,
) where {T, VI, VT, MT}
    ...
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
