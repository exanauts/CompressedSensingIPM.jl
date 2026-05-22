# Low-level FFTW guru interface — interleaved and split (planar) layouts.
#
# Interleaved plans  (standard ComplexF64 arrays):
#   InterleavedC2CPlan  — plan_dft(n, x_in, x_out, sign)
#   InterleavedR2CPlan  — plan_dft_r2c(n, x_in, x_out)
#   InterleavedC2RPlan  — plan_dft_c2r(n, x_in, x_out)
#
# Split plans  (separate Float64 real/imag arrays, == "planar" in rocFFT):
#   SplitC2CPlan  — plan_split_dft(n, ri, ii, ro, io)
#   SplitR2CPlan  — plan_split_dft_r2c(n, x, ro, io)
#   SplitC2RPlan  — plan_split_dft_c2r(n, ri, ii, x)
#
# All plans carry a finalizer that calls fftw_destroy_plan automatically.
#
# Notes on C2C split direction:
#   fftw_plan_guru_split_dft has no sign parameter (always forward).
#   Backward transform = call execute! with ri↔ii (and ro↔io) swapped.

import FFTW

# ─── private helpers ─────────────────────────────────────────────────────────

struct _fftw_iodim
    n::Cint
    is::Cint   # input stride
    os::Cint   # output stride
end

const _FFTW_ESTIMATE = Cuint(64)
const _FFTW_FORWARD  = Cint(-1)
const _FFTW_BACKWARD = Cint(+1)

_make_dims(n::Int) = [_fftw_iodim(Cint(n), Cint(1), Cint(1))]

function _fftw_destroy(ptr::Ptr{Cvoid})
    ptr != C_NULL &&
        ccall((:fftw_destroy_plan, FFTW.libfftw3), Cvoid, (Ptr{Cvoid},), ptr)
end

# ─── plan types ──────────────────────────────────────────────────────────────

for (T, fields) in (
        (:InterleavedC2CPlan, :(n::Int)),
        (:InterleavedR2CPlan, :(n::Int; m::Int = n ÷ 2 + 1)),
        (:InterleavedC2RPlan, :(n::Int; m::Int = n ÷ 2 + 1)),
        (:SplitC2CPlan,       :(n::Int)),
        (:SplitR2CPlan,       :(n::Int; m::Int = n ÷ 2 + 1)),
        (:SplitC2RPlan,       :(n::Int; m::Int = n ÷ 2 + 1)),
    )
    @eval begin
        mutable struct $T
            ptr::Ptr{Cvoid}
            n::Int
            function $T(ptr::Ptr{Cvoid}, n::Int)
                p = new(ptr, n)
                finalizer(p -> _fftw_destroy(p.ptr), p)
                return p
            end
        end
        Base.show(io::IO, p::$T) = print(io, $(string(T)), "(n=", p.n, ")")
    end
end

# ─── plan constructors ───────────────────────────────────────────────────────

"""
    plan_dft(n, x_in, x_out, sign=_FFTW_FORWARD) → InterleavedC2CPlan

C2C interleaved FFT plan. `sign` = `_FFTW_FORWARD` (-1) or `_FFTW_BACKWARD` (+1).
Backward transform is unnormalized (same as rocFFT complex_inverse).
"""
function plan_dft(n::Int,
                  x_in::Vector{ComplexF64}, x_out::Vector{ComplexF64},
                  sign::Cint = _FFTW_FORWARD)
    dims = _make_dims(n)
    ptr = GC.@preserve dims ccall(
        (:fftw_plan_guru_dft, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{ComplexF64}, Ptr{ComplexF64}, Cint, Cuint),
        1, dims, 0, C_NULL, x_in, x_out, sign, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_dft failed")
    return InterleavedC2CPlan(ptr, n)
end

"""
    plan_dft_r2c(n, x_in, x_out) → InterleavedR2CPlan

R2C interleaved: real input (length n) → complex hermitian output (length n÷2+1).
"""
function plan_dft_r2c(n::Int,
                      x_in::Vector{Float64}, x_out::Vector{ComplexF64})
    dims = _make_dims(n)
    ptr = GC.@preserve dims ccall(
        (:fftw_plan_guru_dft_r2c, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{ComplexF64}, Cuint),
        1, dims, 0, C_NULL, x_in, x_out, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_dft_r2c failed")
    return InterleavedR2CPlan(ptr, n)
end

"""
    plan_dft_c2r(n, x_in, x_out) → InterleavedC2RPlan

C2R interleaved: complex hermitian input (length n÷2+1) → real output (length n).
Output is unnormalized (same as rocFFT real_inverse).
"""
function plan_dft_c2r(n::Int,
                      x_in::Vector{ComplexF64}, x_out::Vector{Float64})
    dims = _make_dims(n)
    ptr = GC.@preserve dims ccall(
        (:fftw_plan_guru_dft_c2r, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{ComplexF64}, Ptr{Cdouble}, Cuint),
        1, dims, 0, C_NULL, x_in, x_out, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_dft_c2r failed")
    return InterleavedC2RPlan(ptr, n)
end

"""
    plan_split_dft(n, ri, ii, ro, io) → SplitC2CPlan

C2C split (planar) FFT plan (always forward).
Backward: call `execute!(plan, ii, ri, io, ro)` (swap ri↔ii and ro↔io).
"""
function plan_split_dft(n::Int,
                        ri::Vector{Float64}, ii::Vector{Float64},
                        ro::Vector{Float64}, io::Vector{Float64})
    dims = _make_dims(n)
    ptr = GC.@preserve dims ccall(
        (:fftw_plan_guru_split_dft, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cuint),
        1, dims, 0, C_NULL, ri, ii, ro, io, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_split_dft failed")
    return SplitC2CPlan(ptr, n)
end

"""
    plan_split_dft_r2c(n, x, ro, io) → SplitR2CPlan

R2C split: real input (length n) → split hermitian output `(ro, io)` (length n÷2+1 each).
"""
function plan_split_dft_r2c(n::Int,
                             x::Vector{Float64},
                             ro::Vector{Float64}, io::Vector{Float64})
    dims = _make_dims(n)
    ptr = GC.@preserve dims ccall(
        (:fftw_plan_guru_split_dft_r2c, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cuint),
        1, dims, 0, C_NULL, x, ro, io, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_split_dft_r2c failed")
    return SplitR2CPlan(ptr, n)
end

"""
    plan_split_dft_c2r(n, ri, ii, x) → SplitC2RPlan

C2R split: split hermitian input `(ri, ii)` (length n÷2+1 each) → real output (length n).
Output is unnormalized.
"""
function plan_split_dft_c2r(n::Int,
                             ri::Vector{Float64}, ii::Vector{Float64},
                             x::Vector{Float64})
    dims = _make_dims(n)
    ptr = GC.@preserve dims ccall(
        (:fftw_plan_guru_split_dft_c2r, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cuint),
        1, dims, 0, C_NULL, ri, ii, x, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_split_dft_c2r failed")
    return SplitC2RPlan(ptr, n)
end

# ─── execute! ────────────────────────────────────────────────────────────────

function execute!(p::InterleavedC2CPlan,
                  x_in::Vector{ComplexF64}, x_out::Vector{ComplexF64})
    ccall((:fftw_execute_dft, FFTW.libfftw3), Cvoid,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Ptr{ComplexF64}),
          p.ptr, x_in, x_out)
end

function execute!(p::InterleavedR2CPlan,
                  x_in::Vector{Float64}, x_out::Vector{ComplexF64})
    ccall((:fftw_execute_dft_r2c, FFTW.libfftw3), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{ComplexF64}),
          p.ptr, x_in, x_out)
end

function execute!(p::InterleavedC2RPlan,
                  x_in::Vector{ComplexF64}, x_out::Vector{Float64})
    ccall((:fftw_execute_dft_c2r, FFTW.libfftw3), Cvoid,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Ptr{Cdouble}),
          p.ptr, x_in, x_out)
end

function execute!(p::SplitC2CPlan,
                  ri::AbstractArray{Float64}, ii::AbstractArray{Float64},
                  ro::AbstractArray{Float64}, io::AbstractArray{Float64})
    ccall((:fftw_execute_split_dft, FFTW.libfftw3), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          p.ptr, ri, ii, ro, io)
end

function execute!(p::SplitR2CPlan,
                  x::AbstractArray{Float64},
                  ro::AbstractArray{Float64}, io::AbstractArray{Float64})
    ccall((:fftw_execute_split_dft_r2c, FFTW.libfftw3), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          p.ptr, x, ro, io)
end

function execute!(p::SplitC2RPlan,
                  ri::AbstractArray{Float64}, ii::AbstractArray{Float64},
                  x::AbstractArray{Float64})
    ccall((:fftw_execute_split_dft_c2r, FFTW.libfftw3), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          p.ptr, ri, ii, x)
end

# ─── multi-dim split plan constructors ───────────────────────────────────────
# Column-major strides for NxN...xN arrays.

function _iodims_c2c_nd(dims::NTuple{D,Int}) where D
    strides = ones(Int, D)
    for i in 2:D; strides[i] = strides[i-1] * dims[i-1]; end
    [_fftw_iodim(Cint(dims[i]), Cint(strides[i]), Cint(strides[i])) for i in 1:D]
end

# R2C: input NxN...xN, output (n÷2+1)xN...xN (first dim halved).
function _iodims_r2c_nd(dims::NTuple{D,Int}) where D
    m = dims[1] ÷ 2 + 1
    out_dims = (m, dims[2:end]...)
    is = ones(Int, D);  os = ones(Int, D)
    for i in 2:D
        is[i] = is[i-1] * dims[i-1]
        os[i] = os[i-1] * out_dims[i-1]
    end
    [_fftw_iodim(Cint(dims[i]), Cint(is[i]), Cint(os[i])) for i in 1:D]
end

# C2R: inverse of R2C (input hermitian (n÷2+1)xN...xN, output NxN...xN).
function _iodims_c2r_nd(dims::NTuple{D,Int}) where D
    m = dims[1] ÷ 2 + 1
    in_dims = (m, dims[2:end]...)
    is = ones(Int, D);  os = ones(Int, D)
    for i in 2:D
        is[i] = is[i-1] * in_dims[i-1]
        os[i] = os[i-1] * dims[i-1]
    end
    [_fftw_iodim(Cint(dims[i]), Cint(is[i]), Cint(os[i])) for i in 1:D]
end

"""
    plan_split_dft_nd(dims, ri, ii, ro, io) → SplitC2CPlan

C2C split (planar) FFT plan for ND arrays (always forward).
Backward: call `execute!(plan, ii, ri, io, ro)`.
"""
function plan_split_dft_nd(dims::NTuple{D,Int},
                            ri::AbstractArray{Float64,D}, ii::AbstractArray{Float64,D},
                            ro::AbstractArray{Float64,D}, io::AbstractArray{Float64,D}) where D
    fftw_dims = _iodims_c2c_nd(dims)
    ptr = GC.@preserve fftw_dims ccall(
        (:fftw_plan_guru_split_dft, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cuint),
        D, fftw_dims, 0, C_NULL, ri, ii, ro, io, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_split_dft failed for dims=$dims")
    return SplitC2CPlan(ptr, prod(dims))
end

"""
    plan_split_dft_r2c_nd(dims, x, ro, io) → SplitR2CPlan

R2C split ND: real input `dims` → split hermitian output `(ro, io)` of shape `(dims[1]÷2+1, dims[2:end]...)`.
"""
function plan_split_dft_r2c_nd(dims::NTuple{D,Int},
                                x::AbstractArray{Float64,D},
                                ro::AbstractArray{Float64}, io::AbstractArray{Float64}) where D
    fftw_dims = _iodims_r2c_nd(dims)
    ptr = GC.@preserve fftw_dims ccall(
        (:fftw_plan_guru_split_dft_r2c, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cuint),
        D, fftw_dims, 0, C_NULL, x, ro, io, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_split_dft_r2c failed for dims=$dims")
    return SplitR2CPlan(ptr, prod(dims))
end

"""
    plan_split_dft_c2r_nd(dims, ri, ii, x) → SplitC2RPlan

C2R split ND: split hermitian input `(ri, ii)` of shape `(dims[1]÷2+1, dims[2:end]...)` → real output `dims`.
Output is unnormalized.
"""
function plan_split_dft_c2r_nd(dims::NTuple{D,Int},
                                ri::AbstractArray{Float64}, ii::AbstractArray{Float64},
                                x::AbstractArray{Float64,D}) where D
    fftw_dims = _iodims_c2r_nd(dims)
    ptr = GC.@preserve fftw_dims ccall(
        (:fftw_plan_guru_split_dft_c2r, FFTW.libfftw3), Ptr{Cvoid},
        (Cint, Ptr{_fftw_iodim}, Cint, Ptr{_fftw_iodim},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cuint),
        D, fftw_dims, 0, C_NULL, ri, ii, x, _FFTW_ESTIMATE,
    )
    ptr == C_NULL && error("fftw_plan_guru_split_dft_c2r failed for dims=$dims")
    return SplitC2RPlan(ptr, prod(dims))
end
