using LinearAlgebra
using FFTW
using Random, Distributions

include("mapping_cpu.jl")
# include("mapping_cpu_v2.jl")
include("mapping_gpu.jl")

# compute M_{\perp}^{\top}z

# @param z_zero The zero-imputed signal, i.e. replacing all the missing values in the signal with 0.
# e.g. The signal is [2;3;missing;4], then z_zero = [2;3;0;4].
# @param dim The dimension of the problem (dim = 1, 2, 3)
# @param size The size of each dimension of the problem
#(we only consider the cases when the sizes are even for all the dimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))

# @details This function computes M_{\perp}^{\top}z.

# @return M_{\perp}^{\top}z A vector with length equal to the product of size
# @example
# >widetildez = [2;3;missing;4]
# >z_zero  = [2;3;0;4]
# >dim = 1;
# >size1 = 4;
# >M_perptz = M_perp_tz(z_zero, dim, size1)

function M_perp_tz(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, z_zero, fft_timer, mapping_timer; rdft::Bool=false)
    N = prod(_size)

    t1 = time_ns()
    if rdft
        temp = mul!(buffer_complex1, op, z_zero)  # op_rfft
    else
        buffer_complex2 .= z_zero  # z_zero should be store in a complex buffer for mul!
        temp = mul!(buffer_complex1, op, buffer_complex2)  # op_fft
    end
    temp ./= sqrt(N)
    t2 = time_ns()
    fft_timer[] = fft_timer[] + (t2 - t1) / 1e9

    t3 = time_ns()
    beta = vec(buffer_real)
    DFT_to_beta!(beta, dim, _size, temp; rdft)
    t4 = time_ns()
    mapping_timer[] = mapping_timer[] + (t4 - t3) / 1e9
    return beta
end

function M_perp_beta(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, beta, idx_missing, fft_timer, mapping_timer; rdft::Bool=false)
    N = prod(_size)

    t3 = time_ns()
    v = buffer_complex2
    beta_to_DFT!(v, dim, _size, beta; rdft)
    t4 = time_ns()
    mapping_timer[] = mapping_timer[] + (t4 - t3) / 1e9

    t1 = time_ns()
    if rdft
        ldiv!(buffer_real, op, v)  # op_rfft
        buffer_real .*= sqrt(N)
    else
        temp = ldiv!(buffer_complex1, op, v)  # op_fft
        buffer_real .= real.(temp) .* sqrt(N)
    end
    t2 = time_ns()
    fft_timer[] = fft_timer[] + (t2 - t1) / 1e9

    buffer_real[idx_missing] .= 0
    return buffer_real
end

function M_perpt_M_perp_vec(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, vec, idx_missing, fft_timer, mapping_timer; rdft::Bool=false)
    temp = M_perp_beta(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, vec, idx_missing, fft_timer, mapping_timer; rdft)
    temp = M_perp_tz(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, temp, fft_timer, mapping_timer; rdft)
    return temp
end

# mapping between DFT and real vector beta

# mapping DFT to beta
# @param dim The dimension of the problem (dim = 1, 2, 3)
# @param size The size of each dimension of the problem
#(we only consider the cases when the sizes are even for all the dimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))
# @param v DFT

# @details This fucnction maps DFT to beta

# @return A 1-dimensional real vector beta whose length is the product of size
# @example
# >dim = 2;
# >size1 = (6, 8)
# >x = randn(6, 8)
# >v = fft(x)/sqrt(prod(size1))
# >beta = DFT_to_beta(dim, size1, v)

function DFT_to_beta!(beta, dim::Int, size, v; rdft::Bool=false)
    if (dim == 1)
        DFT_to_beta_1d!(beta, v, size; rdft)
    elseif (dim == 2)
        DFT_to_beta_2d!(beta, v, size; rdft)
    else
        DFT_to_beta_3d!(beta, v, size; rdft)
    end
    return beta
end

function DFT_to_beta(dim::Int, size, v::Array{ComplexF64}; rdft::Bool=false)
    N = prod(size)
    beta = Vector{Float64}(undef, N)
    DFT_to_beta!(beta, dim, size, v; rdft)
    return beta
end

function DFT_to_beta(dim::Int, size, v::CuArray{ComplexF64}; rdft::Bool=false)
    N = prod(size)
    beta = CuVector{Float64}(undef, N)
    DFT_to_beta!(beta, dim, size, v; rdft)
    return beta
end

# mapping beta to DFT
# @param dim The dimension of the problem (dim = 1, 2, 3)
# @param size The size of each dimension of the problem
#(we only consider the cases when the sizes are even for all the dimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))
# @param beta A 1-dimensional real vector with length equal to the product of size

# @details This fucnction maps beta to DFT

# @return DFT DFT shares the same size as param sizes

# @example
# >dim = 2;
# >size1 = (6, 8)
# >x = randn(6, 8)
# >v = fft(x)/sqrt(prod(size1))
# >beta = DFT_to_beta(dim, size1, v)
# >w = beta_to_DFT(dim, size1, beta) (w should be equal to v)

function beta_to_DFT!(v, dim::Int, size, beta; rdft::Bool=false)
    if (dim == 1)
        v = beta_to_DFT_1d!(v, beta, size; rdft)
    elseif (dim == 2)
        v = beta_to_DFT_2d!(v, beta, size; rdft)
    else
        v = beta_to_DFT_3d!(v, beta, size; rdft)
    end
    return v
end

function beta_to_DFT(dim::Int, size, beta::StridedVector{Float64}; rdft::Bool=false)
    v = Array{ComplexF64}(undef, size)
    beta_to_DFT!(v, dim, size, beta; rdft)
    return v
end

function beta_to_DFT(dim::Int, size, beta::StridedCuVector{Float64}; rdft::Bool=false)
    v = CuArray{ComplexF64}(undef, size)
    beta_to_DFT!(v, dim, size, beta; rdft)
    return v
end
