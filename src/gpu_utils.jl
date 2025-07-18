using AMDGPU.rocBLAS, AMDGPU.rocSPARSE

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: ROCVector{T}}
    return view(vec,shift:shift+n-1)
end

# Local transfer! function to move data on the device.
transfer!(x::AbstractArray, y::AbstractArray) = copyto!(x, y)

#=
    SparseMatrixCSC to ROCSparseMatrixCSC
=#

function rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(
        ROCArray(A.colptr),
        ROCArray(A.rowval),
        ROCArray(A.nzval),
        size(A),
    )
end

#=
    SparseMatrixCOO to ROCSparseMatrixCSC
=#

function MadNLP.transfer!(
    dest::rocSPARSE.ROCSparseMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

#=
    ROCSparseMatrixCSC to ROCMatrix
=#

function MadNLPGPU.transfer!(y::ROCMatrix{T}, x::ROCSparseMatrixCSC{T}) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = ROCBackend()
    MadNLPGPU._csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    synchronize(backend)
    return
end

# BLAS operations
MadNLPGPU.symul!(y::ROCVector{T}, A::ROCMatrix{T}, x::ROCVector{T}, α = 1.0, β = 0.0) where T = rocBLAS.symv!('L', T(α), A, x, T(β), y)

MadNLP._ger!(alpha::Number, x::ROCVector{T}, y::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.ger!(T(alpha), x, y, A)

#=
    GPU wrappers for DenseKKTSystem/DenseCondensedKKTSystem
=#

function MadNLP.diag!(dest::ROCVector{T}, src::ROCMatrix{T}) where {T}
    @assert length(dest) == size(src, 1)
    backend = ROCBackend()
    MadNLPGPU._copy_diag_kernel!(backend)(dest, src, ndrange = length(dest))
    synchronize(beckend)
    return
end

#=
    MadNLP.diag_add!
=#

function MadNLP.diag_add!(dest::ROCMatrix, src1::ROCVector, src2::ROCVector)
    backend = ROCBackend()
    MadNLPGPU._add_diagonal_kernel!(backend)(dest, src1, src2, ndrange = size(dest, 1))
    synchronize(backend)
    return
end

#=
    MadNLP._set_diag!
=#

function MadNLP._set_diag!(A::ROCMatrix, inds, a)
    if !isempty(inds)
        backend = ROCBackend()
        MadNLPGPU._set_diag_kernel!(backend)(A, inds, a; ndrange = length(inds))
        synchronize(backend)
    end
    return
end

#=
    MadNLP._build_dense_kkt_system!
=#

function MadNLP._build_dense_kkt_system!(
    dest::ROCMatrix,
    hess::ROCMatrix,
    jac::ROCMatrix,
    pr_diag::ROCVector,
    du_diag::ROCVector,
    diag_hess::ROCVector,
    ind_ineq,
    n,
    m,
    ns,
)
    ind_ineq_gpu = ind_ineq |> ROCArray
    ndrange = (n + m + ns, n)
    backend = ROCBackend()
    MadNLPGPU._build_dense_kkt_system_kernel!(backend)(
        dest,
        hess,
        jac,
        pr_diag,
        du_diag,
        diag_hess,
        ind_ineq_gpu,
        n,
        m,
        ns,
        ndrange = ndrange,
    )
    synchronize(backend)
    return
end

#=
    MadNLP._build_ineq_jac!
=#

function MadNLP._build_ineq_jac!(
    dest::ROCMatrix,
    jac::ROCMatrix,
    diag_buffer::ROCVector,
    ind_ineq::AbstractVector,
    n,
    m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = ind_ineq |> ROCArray
    ndrange = (m_ineq, n)
    backend = ROCBackend()
    MadNLPGPU._build_jacobian_condensed_kernel!(backend)(
        dest,
        jac,
        diag_buffer,
        ind_ineq_gpu,
        m_ineq,
        ndrange = ndrange,
    )
    synchronize(backend)
    return
end

#=
    MadNLP._build_condensed_kkt_system!
=#

function MadNLP._build_condensed_kkt_system!(
    dest::ROCMatrix,
    hess::ROCMatrix,
    jac::ROCMatrix,
    pr_diag::ROCVector,
    du_diag::ROCVector,
    ind_eq::AbstractVector,
    n,
    m_eq,
)
    ind_eq_gpu = ind_eq |> ROCArray
    ndrange = (n + m_eq, n)
    backend = ROCBackend()
    MadNLPGPU._build_condensed_kkt_system_kernel!(backend)(
        dest,
        hess,
        jac,
        pr_diag,
        du_diag,
        ind_eq_gpu,
        n,
        m_eq,
        ndrange = ndrange,
    )
    synchronize(backend)
    return
end

function MadNLP.mul_hess_blk!(
    wx::ROCVector{T},
    kkt::Union{MadNLP.DenseKKTSystem,MadNLP.DenseCondensedKKTSystem},
    t,
) where {T}
    n = size(kkt.hess, 1)
    rocBLAS.symv!('L', one(T), kkt.hess, view(t, 1:n), zero(T), view(wx, 1:n))
    fill!(@view(wx[n+1:end]), 0)
    return wx .+= t .* kkt.pr_diag
end

#=
    GPU wrappers for SparseCondensedKKTSystem.
=#

function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T,VT},
    kkt::MadNLP.SparseCondensedKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T,VT<:ROCVector{T}}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    # Decompose results
    xx = view(MadNLP.full(x), 1:n)
    xs = view(MadNLP.full(x), n+1:n+m)
    xz = view(MadNLP.full(x), n+m+1:n+2*m)

    # Decompose buffers
    wx = view(MadNLP.full(w), 1:n)
    ws = view(MadNLP.full(w), n+1:n+m)
    wz = view(MadNLP.full(w), n+m+1:n+2*m)

    MadNLP.mul!(wx, kkt.hess_com, xx, alpha, beta)
    MadNLP.mul!(wx, kkt.hess_com', xx, alpha, one(T))
    MadNLP.mul!(wx, kkt.jt_csc, xz, alpha, beta)
    if !isempty(kkt.ext.diag_map_to)
        backend = ROCBackend()
        MadNLPGPU._diag_operation_kernel!(backend)(
            wx,
            kkt.hess_com.nzVal,
            xx,
            alpha,
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
        synchronize(backend)
    end

    MadNLP.mul!(wz, kkt.jt_csc', xx, alpha, one(T))
    MadNLP.axpy!(-alpha, xz, ws)
    MadNLP.axpy!(-alpha, xs, wz)
    return MadNLP._kktmul!(
        w,
        x,
        kkt.reg,
        kkt.du_diag,
        kkt.l_lower,
        kkt.u_lower,
        kkt.l_diag,
        kkt.u_diag,
        alpha,
        beta,
    )
end

function MadNLP.mul_hess_blk!(
    wx::VT,
    kkt::Union{MadNLP.SparseKKTSystem,MadNLP.SparseCondensedKKTSystem},
    t,
) where {T,VT<:ROCVector{T}}
    n = size(kkt.hess_com, 1)
    wxx = @view(wx[1:n])
    tx = @view(t[1:n])

    MadNLP.mul!(wxx, kkt.hess_com, tx, one(T), zero(T))
    MadNLP.mul!(wxx, kkt.hess_com', tx, one(T), one(T))
    if !isempty(kkt.ext.diag_map_to)
        backend = ROCBackend()
        MadNLPGPU._diag_operation_kernel!(backend)(
            wxx,
            kkt.hess_com.nzVal,
            tx,
            one(T),
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
        synchronize(backend)
    end

    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
    return
end

function MadNLP.get_tril_to_full(csc::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    cscind = MadNLP.SparseMatrixCSC{Int,Ti}(
        Symmetric(
            MadNLP.SparseMatrixCSC{Int,Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadNLP.nnz(csc)),
            ),
            :L,
        ),
    )
    return rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(
        ROCArray(cscind.colptr),
        ROCArray(cscind.rowval),
        ROCVector{Tv}(undef, MadNLP.nnz(cscind)),
        size(csc),
    ),
    view(csc.nzVal, ROCArray(cscind.nzval))
end

function MadNLP.get_sparse_condensed_ext(
    ::Type{VT},
    hess_com,
    jptr,
    jt_map,
    hess_map,
) where {T,VT<:ROCVector{T}}
    zvals = ROCVector{Int}(1:length(hess_map))
    hess_com_ptr = map((i, j) -> (i, j), hess_map, zvals)
    if length(hess_com_ptr) > 0 # otherwise error is thrown
        sort!(hess_com_ptr)
    end

    jvals = ROCVector{Int}(1:length(jt_map))
    jt_csc_ptr = map((i, j) -> (i, j), jt_map, jvals)
    if length(jt_csc_ptr) > 0 # otherwise error is thrown
        sort!(jt_csc_ptr)
    end

    by = (i, j) -> i[1] != j[1]
    jptrptr = MadNLP.getptr(jptr, by = by)
    hess_com_ptrptr = MadNLP.getptr(hess_com_ptr, by = by)
    jt_csc_ptrptr = MadNLP.getptr(jt_csc_ptr, by = by)

    diag_map_to, diag_map_fr = MadNLPGPU.get_diagonal_mapping(hess_com.colPtr, hess_com.rowVal)

    return (
        jptrptr = jptrptr,
        hess_com_ptr = hess_com_ptr,
        hess_com_ptrptr = hess_com_ptrptr,
        jt_csc_ptr = jt_csc_ptr,
        jt_csc_ptrptr = jt_csc_ptrptr,
        diag_map_to = diag_map_to,
        diag_map_fr = diag_map_fr,
    )
end

function MadNLP._sym_length(Jt::rocSPARSE.ROCSparseMatrixCSC)
    return mapreduce(
        (x, y) -> begin
            z = x - y
            div(z^2 + z, 2)
        end,
        +,
        @view(Jt.colPtr[2:end]),
        @view(Jt.colPtr[1:end-1])
    )
end

function MadNLP._first_and_last_col(sym2::ROCVector, ptr2)
    CUDA.@allowscalar begin
        first = sym2[1][2]
        last = sym2[ptr2[end]][2]
    end
    return (first, last)
end

MadNLP.nzval(H::rocSPARSE.ROCSparseMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::ROCVector, rowval, nzval)
    return rocSPARSE.ROCSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function getij(idx, n)
    j = ceil(Int, ((2n + 1) - sqrt((2n + 1)^2 - 8 * idx)) / 2)
    i = idx - div((j - 1) * (2n - j), 2)
    return (i, j)
end

#=
    MadNLP._set_colptr!
=#

function MadNLP._set_colptr!(colptr::ROCVector, ptr2, sym2, guide)
    if length(ptr2) > 1 # otherwise error is thrown
        backend = ROCBackend()
        MadNLPGPU._set_colptr_kernel!(backend)(
            colptr,
            sym2,
            ptr2,
            guide;
            ndrange = length(ptr2) - 1,
        )
        synchronize(backend)
    end
    return
end


#=
    MadNLP.tril_to_full!
=#

function MadNLP.tril_to_full!(dense::ROCMatrix{T}) where {T}
    n = size(dense, 1)
    backend = ROCBackend()
    MadNLPGPU._tril_to_full_kernel!(backend)(dense; ndrange = div(n^2 + n, 2))
    synchronize(backend)
    return
end

#=
    MadNLP.force_lower_triangular!
=#

function MadNLP.force_lower_triangular!(I::ROCVector{T}, J) where {T}
    if !isempty(I)
        backend = ROCBackend()
        MadNLPGPU._force_lower_triangular_kernel!(backend)(I, J; ndrange = length(I))
        synchronize(backend)
    end
    return
end

#=
    MadNLP.coo_to_csc
=#

function MadNLP.coo_to_csc(
    coo::MadNLP.SparseMatrixCOO{T,I,VT,VI},
) where {T,I,VT<:ROCArray,VI<:ROCArray}
    zvals = ROCVector{Int}(1:length(coo.I))
    coord = map((i, j, k) -> ((i, j), k), coo.I, coo.J, zvals)
    if length(coord) > 0
        sort!(coord, lt = (((i, j), k), ((n, m), l)) -> (j, i) < (m, n))
    end

    mapptr = MadNLP.getptr(coord; by = ((x1, x2), (y1, y2)) -> x1 != y1)

    colptr = similar(coo.I, size(coo, 2) + 1)

    coord_csc = coord[view(mapptr, 1:end-1)]

    if length(coord_csc) > 0
        backend = ROCBackend()
        MadNLPGPU._set_coo_to_colptr_kernel!(backend)(
            colptr,
            coord_csc,
            ndrange = length(coord_csc),
        )
        synchronize(backend)
    else
        fill!(colptr, one(Int))
    end

    rowval = map(x -> x[1][1], coord_csc)
    nzval = similar(rowval, T)

    csc = rocSPARSE.ROCSparseMatrixCSC(colptr, rowval, nzval, size(coo))

    cscmap = similar(coo.I, Int)
    if length(mapptr) > 1
        backend = ROCBackend()
        MadNLPGPU._set_coo_to_csc_map_kernel!(backend)(
            cscmap,
            mapptr,
            coord,
            ndrange = length(mapptr) - 1,
        )
        synchronize(backend)
    end

    return csc, cscmap
end

#=
    MadNLP.build_condensed_aug_coord!
=#

function MadNLP.build_condensed_aug_coord!(
    kkt::MadNLP.AbstractCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:rocSPARSE.ROCSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    if length(kkt.hptr) > 0
        backend = ROCBackend()
        MadNLPGPU._transfer_hessian_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.hptr,
            kkt.hess_com.nzVal;
            ndrange = length(kkt.hptr),
        )
        synchronize(backend)
    end
    if length(kkt.dptr) > 0
        backend = ROCBackend()
        MadNLPGPU._transfer_hessian_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.dptr,
            kkt.pr_diag;
            ndrange = length(kkt.dptr),
        )
        synchronize(backend)
    end
    if length(kkt.ext.jptrptr) > 1 # otherwise error is thrown
        backend = ROCBackend()
        MadNLPGPU._transfer_jtsj_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.jptr,
            kkt.ext.jptrptr,
            kkt.jt_csc.nzVal,
            kkt.diag_buffer;
            ndrange = length(kkt.ext.jptrptr) - 1,
        )
        synchronize(backend)
    end
    return
end

#=
    MadNLP.compress_hessian! / MadNLP.compress_jacobian!
=#

function MadNLP.compress_hessian!(
    kkt::MadNLP.AbstractSparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:rocSPARSE.ROCSparseMatrixCSC{T,Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    if length(kkt.ext.hess_com_ptrptr) > 1
        backend = ROCBackend()
        MadNLPGPU._transfer_to_csc_kernel!(backend)(
            kkt.hess_com.nzVal,
            kkt.ext.hess_com_ptr,
            kkt.ext.hess_com_ptrptr,
            kkt.hess_raw.V;
            ndrange = length(kkt.ext.hess_com_ptrptr) - 1,
        )
        synchronize(backend)
    end
    return
end

function MadNLP.compress_jacobian!(
    kkt::MadNLP.SparseCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:rocSPARSE.ROCSparseMatrixCSC{T,Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        backend = ROCBackend()
        MadNLPGPU._transfer_to_csc_kernel!(backend)(
            kkt.jt_csc.nzVal,
            kkt.ext.jt_csc_ptr,
            kkt.ext.jt_csc_ptrptr,
            kkt.jt_coo.V;
            ndrange = length(kkt.ext.jt_csc_ptrptr) - 1,
        )
        synchronize(backend)
    end
    return
end

#=
    MadNLP._set_con_scale_sparse!
=#

function MadNLP._set_con_scale_sparse!(
    con_scale::VT,
    jac_I,
    jac_buffer,
) where {T,VT<:ROCVector{T}}
    ind_jac = ROCVector{Int}(1:length(jac_I))
    inds = map((i, j) -> (i, j), jac_I, ind_jac)
    if !isempty(inds)
        sort!(inds)
    end
    ptr = MadNLP.getptr(inds; by = ((x1, x2), (y1, y2)) -> x1 != y1)
    if length(ptr) > 1
        backend = ROCBackend()
        MadNLPGPU._set_con_scale_sparse_kernel!(backend)(
            con_scale,
            ptr,
            inds,
            jac_I,
            jac_buffer;
            ndrange = length(ptr) - 1,
        )
        synchronize(backend)
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_hess
=#

function MadNLP._build_condensed_aug_symbolic_hess(
    H::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti},
    sym,
    sym2,
) where {Tv,Ti}
    if size(H, 2) > 0
        backend = ROCBackend()
        MadNLPGPU._build_condensed_aug_symbolic_hess_kernel!(backend)(
            sym,
            sym2,
            H.colPtr,
            H.rowVal;
            ndrange = size(H, 2),
        )
        synchronize(backend)
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_jt
=#

function MadNLP._build_condensed_aug_symbolic_jt(
    Jt::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti},
    sym,
    sym2,
) where {Tv,Ti}
    if size(Jt, 2) > 0
        _offsets = map(
            (i, j) -> div((j - i)^2 + (j - i), 2),
            @view(Jt.colPtr[1:end-1]),
            @view(Jt.colPtr[2:end])
        )
        offsets = cumsum(_offsets)
        backend = ROCBackend()
        MadNLPGPU._build_condensed_aug_symbolic_jt_kernel!(backend)(
            sym,
            sym2,
            Jt.colPtr,
            Jt.rowVal,
            offsets;
            ndrange = size(Jt, 2),
        )
        synchronize(backend)
    end
    return
end
