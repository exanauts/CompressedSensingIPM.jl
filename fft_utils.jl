using LinearAlgebra
using FFTW
using Random, Distributions

include("mapping_cpu.jl")
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

function M_perp_tz(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, z_zero)
    N = prod(_size)
    # println("-- M_perp_tz --")
    # println(_size)
    # println("z_zero | ", z_zero |> size, " | ", typeof(z_zero))
    buffer_complex2 .= z_zero  # z_zero should be store in a complex buffer for mul!
    temp = mul!(buffer_complex1, op, buffer_complex2)
    temp ./= sqrt(N)
    # Out-of-place
    # beta = DFT_to_beta(dim, _size, temp)
    # In-place
    beta = vec(buffer_real)
    DFT_to_beta!(beta, dim, _size, temp)
    # println("beta | ", beta |> size, " | ", typeof(beta))
    # display(beta)
    return beta
end

function M_perp_beta(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, beta, idx_missing)
    N = prod(_size)
    # println("beta | ", beta |> size, " | ", typeof(beta))
    # display(beta)
    v = buffer_complex2
    beta_to_DFT!(v, dim, _size, beta)
    # println("-- M_perp_beta --")
    # println(_size)
    # println("v | ", v |> size, " | ", typeof(v))
    temp = ldiv!(buffer_complex1, op, v)
    buffer_real .= real.(temp) .* sqrt(N)
    buffer_real[idx_missing] .= 0
    return buffer_real
end

function M_perpt_M_perp_vec(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, vec, idx_missing)
    temp = M_perp_beta(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, vec, idx_missing)
    temp = M_perp_tz(buffer_real, buffer_complex1, buffer_complex2, op, dim, _size, temp)
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

function DFT_to_beta!(beta, dim, size, v)
    if (dim == 1)
        DFT_to_beta_1d!(beta, v, size)
    elseif (dim == 2)
        DFT_to_beta_2d!(beta, v, size)
    else
        DFT_to_beta_3d!(beta, v, size)
    end
    return beta
end

function DFT_to_beta(dim, size, v)
    if (dim == 1)
        beta = DFT_to_beta_1d(v, size)
    elseif (dim == 2)
        beta = DFT_to_beta_2d(v, size)
    else
        beta = DFT_to_beta_3d(v, size)
    end
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

function beta_to_DFT!(v, dim, size, beta)
    if (dim == 1)
        v = beta_to_DFT_1d!(v, beta, size)
    elseif (dim == 2)
        v = beta_to_DFT_2d!(v, beta, size)
    else
        v = beta_to_DFT_3d!(v, beta, size)
    end
    return v
end

function beta_to_DFT(dim, size, beta)
    if (dim == 1)
        return beta_to_DFT_1d(beta, size)
    elseif (dim == 2)
        return beta_to_DFT_2d(beta, size)
    elseif (dim == 3)
        return beta_to_DFT_3d(beta, size)
    end
    return v
end
