using LinearAlgebra
using FFTW, CUDA
using Random, Distributions

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
    if dim == 3
        v = beta_to_DFT(dim, _size, beta)
    else
        v = buffer_complex2
        beta_to_DFT!(v, dim, _size, beta)
    end
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

# dim = 1
function DFT_to_beta_1d!(beta::Vector{Float64}, v, size)
    N = size[1]
    M = N ÷ 2
    beta[1] = real(v[  1])
    beta[2] = real(v[M+1])
    for i in 2:M
        beta[i+1] = sqrt(2) * real(v[i])
        beta[i+M] = sqrt(2) * imag(v[i])
    end
    return beta
end

function DFT_to_beta_1d!(beta::CuVector{Float64}, v, size)
    N = size[1]
    M = N ÷ 2
    view(beta, 1:2) .= real.(view(v, 1:M:M+1))
    view(beta, 3:M+1) .= sqrt(2) .* real.(view(v, 2:M))
    view(beta, M+2:N) .= sqrt(2) .* imag.(view(v, 2:M))
    return beta
end

function DFT_to_beta_1d(v::Array{ComplexF64}, size)
    N = size[1]
    beta = Vector{Float64}(undef, N)
    DFT_to_beta_1d!(beta, v, size)
end

function DFT_to_beta_1d(v::CuArray{ComplexF64}, size)
    N = size[1]
    beta = CuVector{Float64}(undef, N)
    DFT_to_beta_1d!(beta, v, size)
end

# dim = 2
function DFT_to_beta_2d!(beta::Array{Float64}, v, size)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    PP = P1 * P2
    beta[1] = real(v[1   , 1   ])
    beta[2] = real(v[1   , M2+1])
    beta[3] = real(v[M1+1, 1   ])
    beta[4] = real(v[M1+1, M2+1])
    index = 4

    for l = index+1 : index+P2
        t = l - index + 1
        beta[l     ] = sqrt(2) * real(v[1   , t])
        beta[l+  P2] = sqrt(2) * imag(v[1   , t])
        beta[l+2*P2] = sqrt(2) * real(v[M1+1, t])
        beta[l+3*P2] = sqrt(2) * imag(v[M1+1, t])
    end
    index = index + 4 * P2

    for l = index+1 : index+P1
        t = l - index + 1
        beta[l     ] = sqrt(2) * real(v[t, 1   ])
        beta[l+  P1] = sqrt(2) * imag(v[t, 1   ])
        beta[l+2*P1] = sqrt(2) * real(v[t, M2+1])
        beta[l+3*P1] = sqrt(2) * imag(v[t, M2+1])
    end
    index = index + 4 * P1

    l = index
    for j = 2:M2
        for i = 2:M1
            l = l+1
            beta[l     ] = sqrt(2) * real(v[i, j   ])
            beta[l+  PP] = sqrt(2) * imag(v[i, j   ])
            beta[l+2*PP] = sqrt(2) * real(v[i, j+M2])
            beta[l+3*PP] = sqrt(2) * imag(v[i, j+M2])
        end
    end
    return beta
end

function DFT_to_beta_2d!(beta::CuArray{Float64}, v, size)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    PP = P1 * P2
    view(beta, 1:2) .= real.(view(v, 1   , 1:M2:M2+1))
    view(beta, 3:4) .= real.(view(v, M1+1, 1:M2:M2+1))
    view(beta, 5               :4+  P2          ) .= sqrt(2) .* real.(view(v, 1, 2:M2))
    view(beta, 5+  P2          :4+2*P2          ) .= sqrt(2) .* imag.(view(v, 1, 2:M2))
    view(beta, 5+2*P2          :4+3*P2          ) .= sqrt(2) .* real.(view(v, M1+1, 2:M2))
    view(beta, 5+3*P2          :4+4*P2          ) .= sqrt(2) .* imag.(view(v, M1+1, 2:M2))
    view(beta, 5+4*P2          :4+4*P2+P1       ) .= sqrt(2) .* real.(view(v, 2:M1, 1))
    view(beta, 5+4*P2+  P1     :4+4*P2+2*P1     ) .= sqrt(2) .* imag.(view(v, 2:M1, 1))
    view(beta, 5+4*P2+2*P1     :4+4*P2+3*P1     ) .= sqrt(2) .* real.(view(v, 2:M1, M2+1))
    view(beta, 5+4*P2+3*P1     :4+4*P2+4*P1     ) .= sqrt(2) .* imag.(view(v, 2:M1, M2+1))
    view(beta, 5+4*P2+4*P1     :4+4*P2+4*P1+  PP) .= sqrt(2) .* real.(view(v, 2:M1, 2:M2) |> vec)
    view(beta, 5+4*P2+4*P1+  PP:4+4*P2+4*P1+2*PP) .= sqrt(2) .* imag.(view(v, 2:M1, 2:M2) |> vec)
    view(beta, 5+4*P2+4*P1+2*PP:4+4*P2+4*P1+3*PP) .= sqrt(2) .* real.(view(v, 2:M1, M2+2:N2) |> vec)
    view(beta, 5+4*P2+4*P1+3*PP:4+4*P2+4*P1+4*PP) .= sqrt(2) .* imag.(view(v, 2:M1, M2+2:N2) |> vec)
    return beta
end

function DFT_to_beta_2d(v::Array{ComplexF64}, size)
    N = prod(size)
    beta = Vector{Float64}(undef, N)
    DFT_to_beta_2d!(beta, v, size)
end

function DFT_to_beta_2d(v::CuArray{ComplexF64}, size)
    N = prod(size)
    beta = CuVector{Float64}(undef, N)
    DFT_to_beta_2d!(beta, v, size)
end

# dim = 3

function DFT_to_beta_3d!(beta::Array{Float64}, v, size)
    N1 = size[1]
    N2 = size[2]
    N3 = size[3]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    M3 = N3 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    P3 = M3 - 1
    P23 = P2 * P3
    P13 = P1 * P3
    P12 = P1 * P2
    P123 = P1 * P2 * P3
    beta[1] = real(v[1   , 1   , 1   ])
    beta[2] = real(v[1   , 1   , M3+1])
    beta[3] = real(v[1   , M2+1, 1   ])
    beta[4] = real(v[1   , M2+1, M3+1])
    beta[5] = real(v[M1+1, 1   , 1   ])
    beta[6] = real(v[M1+1, 1   , M3+1])
    beta[7] = real(v[M1+1, M2+1, 1   ])
    beta[8] = real(v[M1+1, M2+1, M3+1])
    index = 8

    for l = index+1 : index+P3
        t = l - index + 1
        beta[l     ] = sqrt(2) * real(v[1   , 1   , t])
        beta[l+  P3] = sqrt(2) * imag(v[1   , 1   , t])
        beta[l+2*P3] = sqrt(2) * real(v[1   , M2+1, t])
        beta[l+3*P3] = sqrt(2) * imag(v[1   , M2+1, t])
        beta[l+4*P3] = sqrt(2) * real(v[M1+1, 1   , t])
        beta[l+5*P3] = sqrt(2) * imag(v[M1+1, 1   , t])
        beta[l+6*P3] = sqrt(2) * real(v[M1+1, M2+1, t])
        beta[l+7*P3] = sqrt(2) * imag(v[M1+1, M2+1, t])
    end
    index = index + 8 * P1

    for l = index+1 : index+P2
        t = l - index + 1
        beta[l     ] = sqrt(2) * real(v[1   , t, 1   ])
        beta[l+  P2] = sqrt(2) * imag(v[1   , t, 1   ])
        beta[l+2*P2] = sqrt(2) * real(v[1   , t, M3+1])
        beta[l+3*P2] = sqrt(2) * imag(v[1   , t, M3+1])
        beta[l+4*P2] = sqrt(2) * real(v[M1+1, t, 1   ])
        beta[l+5*P2] = sqrt(2) * imag(v[M1+1, t, 1   ])
        beta[l+6*P2] = sqrt(2) * real(v[M1+1, t, M3+1])
        beta[l+7*P2] = sqrt(2) * imag(v[M1+1, t, M3+1])
    end
    index = index + 8 * P2

    for l = index+1 : index+P1
        t = l - index + 1
        beta[l     ] = sqrt(2) * real(v[t, 1   , 1   ])
        beta[l+  P1] = sqrt(2) * imag(v[t, 1   , 1   ])
        beta[l+2*P1] = sqrt(2) * real(v[t, 1   , M3+1])
        beta[l+3*P1] = sqrt(2) * imag(v[t, 1   , M3+1])
        beta[l+4*P1] = sqrt(2) * real(v[t, M2+1, 1   ])
        beta[l+5*P1] = sqrt(2) * imag(v[t, M2+1, 1   ])
        beta[l+6*P1] = sqrt(2) * real(v[t, M2+1, M3+1])
        beta[l+7*P1] = sqrt(2) * imag(v[t, M2+1, M3+1])
    end
    index = index + 8 * P1

    l = index
    for k = 2:M3
        for j = 2:M2
            l = l+1
            beta[l      ] = sqrt(2) * real(v[1   , j   , k])
            beta[l+  P23] = sqrt(2) * imag(v[1   , j   , k])
            beta[l+2*P23] = sqrt(2) * real(v[1   , j+M2, k])
            beta[l+3*P23] = sqrt(2) * imag(v[1   , j+M2, k])
            beta[l+4*P23] = sqrt(2) * real(v[M1+1, j   , k])
            beta[l+5*P23] = sqrt(2) * imag(v[M1+1, j   , k])
            beta[l+6*P23] = sqrt(2) * real(v[M1+1, j+M2, k])
            beta[l+7*P23] = sqrt(2) * imag(v[M1+1, j+M2, k])
        end
    end
    index = index + 8 * P23

    l = index
    for k = 2:M3
        for i = 2:M1
            l = l+1
            beta[l      ] = sqrt(2) * real(v[i   , 1   , k])
            beta[l+  P13] = sqrt(2) * imag(v[i   , 1   , k])
            beta[l+2*P13] = sqrt(2) * real(v[i+M1, 1   , k])
            beta[l+3*P13] = sqrt(2) * imag(v[i+M1, 1   , k])
            beta[l+4*P13] = sqrt(2) * real(v[i   , M2+1, k])
            beta[l+5*P13] = sqrt(2) * imag(v[i   , M2+1, k])
            beta[l+6*P13] = sqrt(2) * real(v[i+M1, M2+1, k])
            beta[l+7*P13] = sqrt(2) * imag(v[i+M1, M2+1, k])
        end
    end
    index = index + 8 * P13

    l = index
    for j = 2:M2
        for i = 2:M1
            l = l+1
            beta[l      ] = sqrt(2) * real(v[i   , j   , 1])
            beta[l+  P12] = sqrt(2) * imag(v[i   , j   , 1])
            beta[l+2*P12] = sqrt(2) * real(v[i+M1, j   , 1])
            beta[l+3*P12] = sqrt(2) * imag(v[i+M1, j   , 1])
            beta[l+4*P12] = sqrt(2) * real(v[i   , j, M3+1])
            beta[l+5*P12] = sqrt(2) * imag(v[i   , j, M3+1])
            beta[l+6*P12] = sqrt(2) * real(v[i+M1, j, M3+1])
            beta[l+7*P12] = sqrt(2) * imag(v[i+M1, j, M3+1])
        end
    end
    index = index + 8 * P12

    l = index
    for k = 2:M3
        for j = 2:M2
            for i = 2:M1
                l = l+1
                beta[l       ] = sqrt(2) * real(v[i   , j   , k])
                beta[l+  P123] = sqrt(2) * imag(v[i   , j   , k])
                beta[l+2*P123] = sqrt(2) * real(v[i+M1, j   , k])
                beta[l+3*P123] = sqrt(2) * imag(v[i+M1, j   , k])
                beta[l+4*P123] = sqrt(2) * real(v[i   , j+M2, k])
                beta[l+5*P123] = sqrt(2) * imag(v[i   , j+M2, k])
                beta[l+6*P123] = sqrt(2) * real(v[i+M1, j+M2, k])
                beta[l+7*P123] = sqrt(2) * imag(v[i+M1, j+M2, k])
            end
        end
    end
    return beta
end

function DFT_to_beta_3d!(beta::CuArray{Float64}, v, size)
    N1 = size[1]
    N2 = size[2]
    N3 = size[3]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    M3 = N3 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    P3 = M3 - 1
    P23 = P2 * P3
    P13 = P1 * P3
    P12 = P1 * P2
    P123 = P1 * P2 * P3
    view(beta, 1:2) .= real.(view(v, 1   , 1   , 1:M3:M3+1))
    view(beta, 3:4) .= real.(view(v, 1   , M2+1, 1:M3:M3+1))
    view(beta, 5:6) .= real.(view(v, M1+1, 1   , 1:M3:M3+1))
    view(beta, 7:8) .= real.(view(v, M1+1, M2+1, 1:M3:M3+1))
    view(beta,9                                        :8+ P3                                    ) .= sqrt(2) .* real.(view(v,1, 1, 2:M3))
    view(beta,9+  P3                                   :8+2*P3                                   ) .= sqrt(2) .* imag.(view(v,1, 1, 2:M3))
    view(beta,9+2*P3                                   :8+3*P3                                   ) .= sqrt(2) .* real.(view(v,1, M2+1, 2:M3))
    view(beta,9+3*P3                                   :8+4*P3                                   ) .= sqrt(2) .* imag.(view(v,1, M2+1, 2:M3))
    view(beta,9+4*P3                                   :8+5*P3                                   ) .= sqrt(2) .* real.(view(v,M1+1, 1, 2:M3))
    view(beta,9+5*P3                                   :8+6*P3                                   ) .= sqrt(2) .* imag.(view(v,M1+1, 1, 2:M3))
    view(beta,9+6*P3                                   :8+7*P3                                   ) .= sqrt(2) .* real.(view(v,M1+1, M2+1, 2:M3))
    view(beta,9+7*P3                                   :8+8*P3                                   ) .= sqrt(2) .* imag.(view(v,M1+1, M2+1, 2:M3))
    view(beta,9+8*P3                                   :8+8*P3+  P2                              ) .= sqrt(2) .* real.(view(v,1, 2:M2, 1))
    view(beta,9+8*P3+  P2                              :8+8*P3+2*P2                              ) .= sqrt(2) .* imag.(view(v,1, 2:M2, 1))
    view(beta,9+8*P3+2*P2                              :8+8*P3+3*P2                              ) .= sqrt(2) .* real.(view(v,1, 2:M2, M3+1))
    view(beta,9+8*P3+3*P2                              :8+8*P3+4*P2                              ) .= sqrt(2) .* imag.(view(v,1, 2:M2, M3+1))
    view(beta,9+8*P3+4*P2                              :8+8*P3+5*P2                              ) .= sqrt(2) .* real.(view(v,M1+1, 2:M2, 1))
    view(beta,9+8*P3+5*P2                              :8+8*P3+6*P2                              ) .= sqrt(2) .* imag.(view(v,M1+1, 2:M2, 1))
    view(beta,9+8*P3+6*P2                              :8+8*P3+7*P2                              ) .= sqrt(2) .* real.(view(v,M1+1, 2:M2, M3+1))
    view(beta,9+8*P3+7*P2                              :8+8*P3+8*P2                              ) .= sqrt(2) .* imag.(view(v,M1+1, 2:M2, M3+1))
    view(beta,9+8*P3+8*P2                              :8+8*P3+8*P2+  P1                         ) .= sqrt(2) .* real.(view(v,2:M1, 1, 1))
    view(beta,9+8*P3+8*P2+  P1                         :8+8*P3+8*P2+2*P1                         ) .= sqrt(2) .* imag.(view(v,2:M1, 1, 1))
    view(beta,9+8*P3+8*P2+2*P1                         :8+8*P3+8*P2+3*P1                         ) .= sqrt(2) .* real.(view(v,2:M1, 1, M3+1))
    view(beta,9+8*P3+8*P2+3*P1                         :8+8*P3+8*P2+4*P1                         ) .= sqrt(2) .* imag.(view(v,2:M1, 1, M3+1))
    view(beta,9+8*P3+8*P2+4*P1                         :8+8*P3+8*P2+5*P1                         ) .= sqrt(2) .* real.(view(v,2:M1, M2+1, 1))
    view(beta,9+8*P3+8*P2+5*P1                         :8+8*P3+8*P2+6*P1                         ) .= sqrt(2) .* imag.(view(v,2:M1, M2+1, 1))
    view(beta,9+8*P3+8*P2+6*P1                         :8+8*P3+8*P2+7*P1                         ) .= sqrt(2) .* real.(view(v,2:M1, M2+1, M3+1))
    view(beta,9+8*P3+8*P2+7*P1                         :8+8*P3+8*P2+8*P1                         ) .= sqrt(2) .* imag.(view(v,2:M1, M2+1, M3+1))
    view(beta,9+8*P3+8*P2+8*P1                         :8+8*P3+8*P2+8*P1+  P23                   ) .= sqrt(2) .* real.(view(v,1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+  P23                   :8+8*P3+8*P2+8*P1+2*P23                   ) .= sqrt(2) .* imag.(view(v,1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+2*P23                   :8+8*P3+8*P2+8*P1+3*P23                   ) .= sqrt(2) .* real.(view(v,1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+3*P23                   :8+8*P3+8*P2+8*P1+4*P23                   ) .= sqrt(2) .* imag.(view(v,1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+4*P23                   :8+8*P3+8*P2+8*P1+5*P23                   ) .= sqrt(2) .* real.(view(v,M1+1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+5*P23                   :8+8*P3+8*P2+8*P1+6*P23                   ) .= sqrt(2) .* imag.(view(v,M1+1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+6*P23                   :8+8*P3+8*P2+8*P1+7*P23                   ) .= sqrt(2) .* real.(view(v,M1+1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+7*P23                   :8+8*P3+8*P2+8*P1+8*P23                   ) .= sqrt(2) .* imag.(view(v,M1+1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23                   :8+8*P3+8*P2+8*P1+8*P23+  P13             ) .= sqrt(2) .* real.(view(v,2:M1, 1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+  P13             :8+8*P3+8*P2+8*P1+8*P23+2*P13             ) .= sqrt(2) .* imag.(view(v,2:M1, 1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+2*P13             :8+8*P3+8*P2+8*P1+8*P23+3*P13             ) .= sqrt(2) .* real.(view(v,M1+2:N1, 1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+3*P13             :8+8*P3+8*P2+8*P1+8*P23+4*P13             ) .= sqrt(2) .* imag.(view(v,M1+2:N1, 1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+4*P13             :8+8*P3+8*P2+8*P1+8*P23+5*P13             ) .= sqrt(2) .* real.(view(v,2:M1, M2+1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+5*P13             :8+8*P3+8*P2+8*P1+8*P23+6*P13             ) .= sqrt(2) .* imag.(view(v,2:M1, M2+1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+6*P13             :8+8*P3+8*P2+8*P1+8*P23+7*P13             ) .= sqrt(2) .* real.(view(v,M1+2:N1, M2+1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+7*P13             :8+8*P3+8*P2+8*P1+8*P23+8*P13             ) .= sqrt(2) .* imag.(view(v,M1+2:N1, M2+1, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13             :8+8*P3+8*P2+8*P1+8*P23+8*P13+  P12       ) .= sqrt(2) .* real.(view(v,2:M1, 2:M2, 1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+  P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12       ) .= sqrt(2) .* imag.(view(v,2:M1, 2:M2, 1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12       ) .= sqrt(2) .* real.(view(v,M1+2:N1, 2:M2, 1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12       ) .= sqrt(2) .* imag.(view(v,M1+2:N1, 2:M2, 1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12       ) .= sqrt(2) .* real.(view(v,2:M1, 2:M2, M3+1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12       ) .= sqrt(2) .* imag.(view(v,2:M1, 2:M2, M3+1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12       ) .= sqrt(2) .* real.(view(v,M1+2:N1, 2:M2, M3+1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12       ) .= sqrt(2) .* imag.(view(v,M1+2:N1, 2:M2, M3+1) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12       :8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+  P123) .= sqrt(2) .* real.(view(v,2:M1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+  P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123) .= sqrt(2) .* imag.(view(v,2:M1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123) .= sqrt(2) .* real.(view(v,M1+2:N1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123) .= sqrt(2) .* imag.(view(v,M1+2:N1, 2:M2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123) .= sqrt(2) .* real.(view(v,2:M1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123) .= sqrt(2) .* imag.(view(v,2:M1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+7*P123) .= sqrt(2) .* real.(view(v,M1+2:N1, M2+2:N2, 2:M3) |> vec)
    view(beta,9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+7*P123:8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+8*P123) .= sqrt(2) .* imag.(view(v,M1+2:N1, M2+2:N2, 2:M3) |> vec)
    return beta
end

function DFT_to_beta_3d(v::Array{ComplexF64}, size)
    N = prod(size)
    beta = Vector{Float64}(undef, N)
    DFT_to_beta_3d!(beta, v, size)
end

function DFT_to_beta_3d(v::CuArray{ComplexF64}, size)
    N = prod(size)
    beta = CuVector{Float64}(undef, N)
    DFT_to_beta_3d!(beta, v, size)
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
        return beta_to_DFT_1d!(v, beta, size)
    elseif (dim == 2)
        return beta_to_DFT_2d!(v, beta, size)
    else
        error("Dimension not supported")
    end
    return v
end

function beta_to_DFT(dim, size, beta)
    cpu = beta isa Array
    if !cpu && (dim == 3)
        beta = Array(beta)
    end
    if (dim == 1)
        return beta_to_DFT_1d(beta, size)
    elseif (dim == 2)
        return beta_to_DFT_2d(beta, size)
    elseif (dim == 3)
        return beta_to_DFT_3d(beta, size)
    end
    if !cpu && (dim == 3)
        v = CuArray(v)
    end
    return v
end

# dim = 1
function beta_to_DFT_1d!(v::Vector{ComplexF64}, beta, size)
    N = size[1]
    M = N ÷ 2
    v[1  ] = beta[1]
    v[M+1] = beta[2]
    for i = 2:M
        v[i    ] = (beta[i+1] + im * beta[M+i]) / sqrt(2)
        v[N-i+2] = (beta[i+1] - im * beta[M+i]) / sqrt(2)
    end
    return v
end

function beta_to_DFT_1d!(v::CuVector{ComplexF64}, beta, size)
    N = size[1]
    M = N ÷ 2
    view(v, 1:M:M+1) .= view(beta, 1:2)
    beta_r = view(beta, 3:M+1)
    beta_c = view(beta, M+2:N)
    view(v, 2:M     ) .= (beta_r .+ im .* beta_c) ./ sqrt(2)
    view(v, N:-1:M+2) .= (beta_r .- im .* beta_c) ./ sqrt(2)
    return v
end

function beta_to_DFT_1d(beta::StridedArray{Float64}, size)
    N = size[1]
    v = Vector{ComplexF64}(undef, N)
    beta_to_DFT_1d!(v, beta, size)
end

function beta_to_DFT_1d(beta::StridedCuArray{Float64}, size)
    N = size[1]
    v = CuVector{ComplexF64}(undef, N)
    beta_to_DFT_1d!(v, beta, size)
end

# dim = 2
function beta_to_DFT_2d!(v::Matrix{ComplexF64}, beta, size)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    v[1   ,1   ] = beta[1]
    v[1   ,M2+1] = beta[2]
    v[M1+1,1   ] = beta[3]
    v[M1+1,M2+1] = beta[4]

    for i = 1:P1
        v[i+1   ,1   ] = (beta[4+4*P2+i     ] + im * beta[4+4*P2+P1+i  ]) / sqrt(2)
        v[N1-i+1,1   ] = (beta[4+4*P2+i     ] - im * beta[4+4*P2+P1+i  ]) / sqrt(2)
        v[i+1   ,M2+1] = (beta[4+4*P2+2*P1+i] + im * beta[4+4*P2+3*P1+i]) / sqrt(2)
        v[N1-i+1,M2+1] = (beta[4+4*P2+2*P1+i] - im * beta[4+4*P2+3*P1+i]) / sqrt(2)
    end

    for i = 1:P2
        v[1   , i+1] = (beta[4+i     ] + im * beta[4+P2+i  ]) / sqrt(2)
        v[M1+1, i+1] = (beta[4+2*P2+i] + im * beta[4+3*P2+i]) / sqrt(2)
    end

    j = 0
    c = 0
    for col = 2:M2
        for row = 2:M1
            j = j+1
            v[row   , col] = (beta[4+4*P2+4*P1+j] + im * beta[4+4*P2+4*P1+P1*P2+j]) / sqrt(2)
            v[row+M1, col] = (beta[4+4*P2+4*P1+3*P1*P2-c] - im * beta[N1*N2-c]) / sqrt(2)
            c = c+1
        end
    end

    for i = 1:P2
        v[1   , M2+1+i] = conj(v[   1,M2-i+1])
        v[M1+1, M2+1+i] = conj(v[M1+1,M2-i+1])
    end

    for i = 1:P1
        for j = 1:P2
            v[   i+1, M2+1+j] = conj(v[N1-i+1,M2-j+1])
            v[M1+i+1, M2+1+j] = conj(v[M1-i+1,M2-j+1])
        end
    end
    return v
end

function beta_to_DFT_2d!(v::CuMatrix{ComplexF64}, beta, size)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1

    view(v,1:M1:M1+1,1   ) .= view(beta, 1:2:3)
    view(v,1:M1:M1+1,M2+1) .= view(beta, 2:2:4)

    view(v,1   ,2:M2) .= (view(beta,4+1:4+M2-1) .+ im .* view(beta,4+P2+1:4+2*P2)) ./ sqrt(2)
    view(v,M1+1,2:M2) .= (view(beta,4+2*P2+1:4+3*P2) .+ im .* view(beta,4+3*P2+1:4+4*P2)) ./ sqrt(2)

    beta_r = view(beta,4+4*P2+1:4+4*P2+P1)
    beta_c = view(beta,4+4*P2+P1+1:4+4*P2+2*P1)
    view(v,2:M1,1)       .= (beta_r .+ im .* beta_c) ./ sqrt(2)
    view(v,N1:-1:M1+2,1) .= (beta_r .- im .* beta_c) ./ sqrt(2)

    beta_r = view(beta,4+4*P2+2*P1+1:4+4*P2+3*P1)
    beta_c = view(beta,4+4*P2+3*P1+1:4+4*P2+4*P1)
    view(v,2:M1      ,M2+1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)
    view(v,N1:-1:M1+2,M2+1) .= (beta_r .- im .* beta_c) ./ sqrt(2)

    view(v,1      ,M2+2:N2) .= conj.(view(v,1         ,M2:-1:2))
    view(v,2:M1   ,M2+2:N2) .= conj.(view(v,N1:-1:M1+2,M2:-1:2))
    view(v,M1+1   ,M2+2:N2) .= conj.(view(v,M1+1      ,M2:-1:2))
    view(v,M1+2:N1,M2+2:N2) .= conj.(view(v,M1:-1:2   ,M2:-1:2))

    beta_r = reshape(view(beta,4+4*P2+4*P1+1:4+4*P2+4*P1+P1*P2), P1, P2)
    beta_c = reshape(view(beta,4+4*P2+4*P1+P1*P2+1:4+4*P2+4*P1+2*P1*P2), P1, P2)
    view(v,2:M1,2:M2) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,4+4*P2+4*P1+3*P1*P2:-1:4+4*P2+4*P1+2*P1*P2+1), P1, P2)
    beta_c = reshape(view(beta,N1*N2:-1:4+4*P2+4*P1+3*P1*P2+1), P1, P2)
    view(v,M1+2:N1,2:M2) .= (beta_r .- im .* beta_c) ./ sqrt(2)
    return v
end

function beta_to_DFT_2d(beta::StridedArray{Float64}, size)
    N1 = size[1]
    N2 = size[2]
    v = Matrix{ComplexF64}(undef, N1, N2)
    beta_to_DFT_2d!(v, beta, size)
end

function beta_to_DFT_2d(beta::StridedCuArray{Float64}, size)
    N1 = size[1]
    N2 = size[2]
    v = CuMatrix{ComplexF64}(undef, N1, N2)
    beta_to_DFT_2d!(v, beta, size)
end

# dim = 3
function beta_to_DFT_3d!(v, beta, size)
    N1 = size[1]
    N2 = size[2]
    N3 = size[3]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    M3 = N3 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    P3 = M3 - 1
    P23 = P2 * P3
    P13 = P1 * P3
    P12 = P1 * P2
    P123 = P1 * P2 * P3

    v[1   , 1   , 1   ] = beta[1]
    v[1   , 1   , M3+1] = beta[2]
    v[1   , M2+1, 1   ] = beta[3]
    v[1   , M2+1, M3+1] = beta[4]
    v[M1+1, 1   , 1   ] = beta[5]
    v[M1+1, 1   , M3+1] = beta[6]
    v[M1+1, M2+1, 1   ] = beta[7]
    v[M1+1, M2+1, M3+1] = beta[8]

    v[2:M1   , 1, 1] = (beta[8+8*P3+8*P2+1:8+8*P3+8*P2+P1] .+ im .* beta[8+8*P3+8*P2+P1+1:8+8*P3+8*P2+2*P1]) ./ sqrt(2)
    v[M1+2:N1, 1, 1] = (beta[8+8*P3+8*P2+P1:-1:8+8*P3+8*P2+1] .- im .* beta[8+8*P3+8*P2+2*P1:-1:8+8*P3+8*P2+P1+1]) ./ sqrt(2)

    v[1, 2:M2, 1] = (beta[8+8*P3+1:8+8*P3+P2] .+ im .* beta[8+8*P3+P2+1:8+8*P3+2*P2]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+P1*P2], P1, P2)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+2*P1*P2], P1, P2)
    v[2:M1, 2:M2, 1] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+1, 2:M2, 1] = (beta[8+8*P3+4*P2+1:8+8*P3+5*P2] .+ im .* beta[8+8*P3+5*P2+1:8+8*P3+6*P2]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+2*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+3*P1*P2], P1, P2)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+3*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+4*P1*P2], P1, P2)
    v[M1+2:N1, 2:M2, 1] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[2:M1   , M2+1, 1] = (beta[8+8*P3+8*P2+4*P1+1:8+8*P3+8*P2+5*P1] .+ im .* beta[8+8*P3+8*P2+5*P1+1:8+8*P3+8*P2+6*P1]) ./ sqrt(2)
    v[M1+2:N1, M2+1, 1] = (beta[8+8*P3+8*P2+5*P1:-1:8+8*P3+8*P2+4*P1+1] .- im .* beta[8+8*P3+8*P2+6*P1:-1:8+8*P3+8*P2+5*P1+1]) ./ sqrt(2)

    v[1      , M2+2:N2, 1] = conj.(v[1, M2:-1:2, 1])
    v[2:M1   , M2+2:N2, 1] = conj.(v[N1:-1:M1+2, M2:-1:2, 1])
    v[M1+1   , M2+2:N2, 1] = conj.(v[M1+1, M2:-1:2, 1])
    v[M1+2:N1, M2+2:N2, 1] = conj.(v[M1:-1:2, M2:-1:2, 1])

    v[2:M1   , 1, M3+1] = (beta[8+8*P3+8*P2+2*P1+1:8+8*P3+8*P2+3*P1] .+ im .* beta[8+8*P3+8*P2+3*P1+1:8+8*P3+8*P2+4*P1]) ./ sqrt(2)
    v[M1+2:N1, 1, M3+1] = (beta[8+8*P3+8*P2+3*P1:-1:8+8*P3+8*P2+2*P1+1] .- im .* beta[8+8*P3+8*P2+4*P1:-1:8+8*P3+8*P2+3*P1+1]) ./ sqrt(2)

    v[1, 2:M2, M3+1] = (beta[8+8*P3+2*P2+1:8+8*P3+3*P2] .+ im .* beta[8+8*P3+3*P2+1:8+8*P3+4*P2]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+4*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+5*P1*P2], P1, P2)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+5*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+6*P1*P2], P1, P2)
    v[2:M1, 2:M2, M3+1] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+1, 2:M2, M3+1] = (beta[8+8*P3+6*P2+1:8+8*P3+7*P2] .+ im .* beta[8+8*P3+7*P2+1:8+8*P3+8*P2]) ./sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+6*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+7*P1*P2], P1, P2)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+7*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2], P1, P2)
    v[M1+2:N1, 2:M2, M3+1] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[2:M1   , M2+1, M3+1] = (beta[8+8*P3+8*P2+6*P1+1:8+8*P3+8*P2+7*P1] .+ im .* beta[8+8*P3+8*P2+7*P1+1:8+8*P3+8*P2+8*P1]) ./ sqrt(2)
    v[M1+2:N1, M2+1, M3+1] = (beta[8+8*P3+8*P2+7*P1:-1:8+8*P3+8*P2+6*P1+1] .- im .* beta[8+8*P3+8*P2+8*P1:-1:8+8*P3+8*P2+7*P1+1]) ./ sqrt(2)

    v[1      , M2+2:N2, M3+1] = conj.(v[1, M2:-1:2, M3+1])
    v[2:M1   , M2+2:N2, M3+1] = conj.(v[N1:-1:M1+2, M2:-1:2, M3+1])
    v[M1+1   , M2+2:N2, M3+1] = conj.(v[M1+1, M2:-1:2, M3+1])
    v[M1+2:N1, M2+2:N2, M3+1] = conj.(v[M1:-1:2, M2:-1:2, M3+1])

    v[1, 1, 2:M3] = (beta[9:8+P3] .+ im .* beta[8+P3+1:8+2*P3]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+P1*P3], P1, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+2*P1*P3], P1, P3)
    v[2:M1, 1, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+1, 1, 2:M3] = (beta[8+4*P3+1:8+5*P3] .+ im .* beta[8+5*P3+1:8+6*P3]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+2*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+3*P1*P3], P1, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+3*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+4*P1*P3], P1, P3)
    v[M1+2:N1, 1, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[1      , 1, M3+2:N3] = conj.(v[1, 1, M3:-1:2])
    v[2:M1   , 1, M3+2:N3] = conj.(v[N1:-1:M1+2, 1, M3:-1:2])
    v[M1+1   , 1, M3+2:N3] = conj.(v[M1+1, 1, M3:-1:2])
    v[M1+2:N1, 1, M3+2:N3] = conj.(v[M1:-1:2, 1, M3:-1:2])

    v[1, M2+1, 2:M3] = (beta[8+2*P3+1:8+3*P3] .+ im .* beta[8+3*P3+1:8+4*P3]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+4*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+5*P1*P3], P1, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+5*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+6*P1*P3], P1, P3)
    v[2:M1, M2+1, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+1, M2+1, 2:M3] = (beta[8+6*P3+1:8+7*P3] .+ im .* beta[8+7*P3+1:8+8*P3]) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+6*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+7*P1*P3], P1, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+7*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3], P1, P3)
    v[M1+2:N1, M2+1, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[1      , M2+1, M3+2:N3] = conj.(v[1, M2+1, M3:-1:2])
    v[2:M1   , M2+1, M3+2:N3] = conj.(v[N1:-1:M1+2, M2+1, M3:-1:2])
    v[M1+1   , M2+1, M3+2:N3] = conj.(v[M1+1, M2+1, M3:-1:2])
    v[M1+2:N1, M2+1, M3+2:N3] = conj.(v[M1:-1:2, M2+1, M3:-1:2])

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+1:8+8*P3+8*P2+8*P1+P2*P3], P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+P2*P3+1:8+8*P3+8*P2+8*P1+2*P2*P3], P2, P3)
    v[1, 2:M2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+2*P2*P3+1:8+8*P3+8*P2+8*P1+3*P2*P3], P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+3*P2*P3+1:8+8*P3+8*P2+8*P1+4*P2*P3], P2, P3)
    v[1, M2+2:N2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[1, 2:M2   , M3+2:N3] = conj.(v[1, N2:-1:M2+2, M3:-1:2])
    v[1, M2+2:N2, M3+2:N3] = conj.(v[1, M2:-1:2, M3:-1:2])

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+4*P2*P3+1:8+8*P3+8*P2+8*P1+5*P2*P3], P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+5*P2*P3+1:8+8*P3+8*P2+8*P1+6*P2*P3], P2, P3)
    v[M1+1, 2:M2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+6*P2*P3+1:8+8*P3+8*P2+8*P1+7*P2*P3], P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+7*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3], P2, P3)
    v[M1+1, M2+2:N2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+1, 2:M2   , M3+2:N3] = conj.(v[M1+1, N2:-1:M2+2, M3:-1:2])
    v[M1+1, M2+2:N2, M3+2:N3] = conj.(v[M1+1, M2:-1:2, M3:-1:2])

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+P1*P2*P3], P1, P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+2*P1*P2*P3], P1, P2, P3)
    v[2:M1, 2:M2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+2:N1, M2+2:N2, M3+2:N3] = conj.(v[M1:-1:2, M2:-1:2, M3:-1:2])

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+2*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+3*P1*P2*P3], P1, P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+3*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+4*P1*P2*P3], P1, P2, P3)
    v[M1+2:N1, 2:M2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[2:M1, M2+2:N2, M3+2:N3] = conj.(v[N1:-1:M1+2, M2:-1:2, M3:-1:2])

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+4*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+5*P1*P2*P3], P1, P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+5*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+6*P1*P2*P3], P1, P2, P3)
    v[2:M1, M2+2:N2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[M1+2:N1, 2:M2, M3+2:N3] = conj.(v[M1:-1:2, N2:-1:M2+2, M3:-1:2])

    beta_r = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+6*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+7*P1*P2*P3], P1, P2, P3)
    beta_c = reshape(beta[8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+7*P1*P2*P3+1:N1*N2*N3], P1, P2, P3)
    v[M1+2:N1, M2+2:N2, 2:M3] = (beta_r .+ im .* beta_c) ./ sqrt(2)

    v[2:M1, 2:M2, M3+2:N3] = conj.(v[N1:-1:M1+2, N2:-1:M2+2, M3:-1:2])
    return v
end

function beta_to_DFT_3d(beta::StridedArray{Float64}, size)
    N1 = size[1]
    N2 = size[2]
    N3 = size[3]
    v = Array{ComplexF64, 3}(undef, N1, N2, N3)
    beta_to_DFT_3d!(v, beta, size)
end

function beta_to_DFT_3d(beta::StridedCuArray{Float64}, size)
    N1 = size[1]
    N2 = size[2]
    N3 = size[3]
    v = CuArray{ComplexF64, 3}(undef, N1, N2, N3)
    beta_to_DFT_3d!(v, beta, size)
end
