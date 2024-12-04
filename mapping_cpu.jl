# DFT_to_beta
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

function DFT_to_beta_1d(v::Array{ComplexF64}, size)
    N = size[1]
    beta = Vector{Float64}(undef, N)
    DFT_to_beta_1d!(beta, v, size)
end

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

function DFT_to_beta_2d(v::Array{ComplexF64}, size)
    N = prod(size)
    beta = Vector{Float64}(undef, N)
    DFT_to_beta_2d!(beta, v, size)
end

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

function DFT_to_beta_3d(v::Array{ComplexF64}, size)
    N = prod(size)
    beta = Vector{Float64}(undef, N)
    DFT_to_beta_3d!(beta, v, size)
end

# beta_to_DFT
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

function beta_to_DFT_1d(beta::StridedArray{Float64}, size)
    N = size[1]
    v = Vector{ComplexF64}(undef, N)
    beta_to_DFT_1d!(v, beta, size)
end

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

function beta_to_DFT_2d(beta::StridedArray{Float64}, size)
    N1 = size[1]
    N2 = size[2]
    v = Matrix{ComplexF64}(undef, N1, N2)
    beta_to_DFT_2d!(v, beta, size)
end

function beta_to_DFT_3d!(v::Array{ComplexF64, 3}, beta, size)
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

    v[1   ,1   ,1   ] = beta[1]
    v[1   ,1   ,M3+1] = beta[2]
    v[1   ,M2+1,1   ] = beta[3]
    v[1   ,M2+1,M3+1] = beta[4]
    v[M1+1,1   ,1   ] = beta[5]
    v[M1+1,1   ,M3+1] = beta[6]
    v[M1+1,M2+1,1   ] = beta[7]
    v[M1+1,M2+1,M3+1] = beta[8]

    for i = 1:P1
        index_r = 8+8*P3+8*P2+i
        index_c = 8+8*P3+8*P2+P1+i
        v[i+1,1,1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+P1-i+1
        index_c = 8+8*P3+8*P2+2*P1-i+1
        v[M1+1+i,1,1] = (beta[index_r] - im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        index_r = 8+8*P3+j
        index_c = 8+8*P3+P2+j
        v[1,j+1,1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        for i = 1:P1
            ij = (j-1)*P1 + i
            index_r = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+ij
            index_c = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+P1*P2+ij
            v[i+1,j+1,1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
        end
    end

    for j = 1:P2
        index_r = 8+8*P3+4*P2+j
        index_c = 8+8*P3+5*P2+j
        v[M1+1,j+1,1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        for i = 1:P1
            ij = (j-1)*P1 + i
            index_r = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+2*P1*P2+ij
            index_c = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+3*P1*P2+ij
            v[M1+1+i,j+1,1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
        end
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+4*P1+i
        index_c = 8+8*P3+8*P2+5*P1+i
        v[i+1,M2+1,1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+5*P1-i+1
        index_c = 8+8*P3+8*P2+6*P1-i+1
        v[M1+i+1,M2+1,1] = (beta[index_r] - im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        v[1   ,M2+j+1,1] = conj(v[1   , M2-j+1, 1])
        v[M1+1,M2+j+1,1] = conj(v[M1+1, M2-j+1, 1])
    end

    for j = 1:P2
        for i = 1:P1
            v[i+1   , M2+j+1, 1] = conj(v[N1-i+1, M2-j+1, 1])
            v[M1+i+1, M2+j+1, 1] = conj(v[M1-i+1, M2-j+1, 1])
        end
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+2*P1+i
        index_c = 8+8*P3+8*P2+3*P1+i
        v[i+1,1,M3+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+3*P1-i+1
        index_c = 8+8*P3+8*P2+4*P1-i+1
        v[M1+i+1,1,M3+1] = (beta[index_r] - im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        index_r = 8+8*P3+2*P2+j
        index_c = 8+8*P3+3*P2+j
        v[1,j+1,M3+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        for i = 1:P1
            ij = (j-1)*P1 + i
            index_r = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+4*P1*P2+ij
            index_c = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+5*P1*P2+ij
            v[i+1,j+1,M3+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
        end
    end

    for j = 1:P2
        index_r = 8+8*P3+6*P2+j
        index_c = 8+8*P3+7*P2+j
        v[M1+1,j+1,M3+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        for i = 1:P1
            ij = (j-1)*P1 + i
            index_r = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+6*P1*P2+ij
            index_c = 8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+7*P1*P2+ij
            v[M1+i+1,j+1,M3+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
        end
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+6*P1+i
        index_c = 8+8*P3+8*P2+7*P1+i
        v[i+1,M2+1,M3+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for i = 1:P1
        index_r = 8+8*P3+8*P2+7*P1-i+1
        index_c = 8+8*P3+8*P2+8*P1-i+1
        v[M1+i+1,M2+1,M3+1] = (beta[index_r] - im * beta[index_c]) / sqrt(2)
    end

    for j = 1:P2
        v[1   ,M2+j+1,M3+1] = conj(v[1   , M2-j+1, M3+1])
        v[M1+1,M2+j+1,M3+1] = conj(v[M1+1, M2-j+1, M3+1])
    end

    for j = 1:P2
        for i = 1:P1
            v[i+1   , M2+j+1, M3+1] = conj(v[N1-i+1, M2-j+1, M3+1])
            v[M1+i+1, M2+j+1, M3+1] = conj(v[M1-i+1, M2-j+1, M3+1])
        end
    end

    for k = 1:P3
        index_r = 8+k
        index_c = 8+P3+k
        v[1,1,k+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for k = 1:P3
        for i = 1:P1
            ik = (k-1)*P1 + i
            index_r = 8+8*P3+8*P2+8*P1+8*P2*P3+ik
            index_c = 8+8*P3+8*P2+8*P1+8*P2*P3+P1*P3+ik
            v[i+1,1,k+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
        end
    end

    for k = 1:P3
        index_r = 8+4*P3+k
        index_c = 8+5*P3+k
        v[M1+1,1,k+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
    end

    for k = 1:P3
        for i = 1:P1
            ik = (k-1)*P1 + i
            index_r = 8+8*P3+8*P2+8*P1+8*P2*P3+2*P1*P3+ik
            index_c = 8+8*P3+8*P2+8*P1+8*P2*P3+3*P1*P3+ik
            v[M1+i+1,1,k+1] = (beta[index_r] + im * beta[index_c]) / sqrt(2)
        end
    end

    view(v,1      , 1, M3+2:N3) .= conj.(view(v,1, 1, M3:-1:2))
    view(v,2:M1   , 1, M3+2:N3) .= conj.(view(v,N1:-1:M1+2, 1, M3:-1:2))
    view(v,M1+1   , 1, M3+2:N3) .= conj.(view(v,M1+1, 1, M3:-1:2))
    view(v,M1+2:N1, 1, M3+2:N3) .= conj.(view(v,M1:-1:2, 1, M3:-1:2))

    view(v,1, M2+1, 2:M3) .= (view(beta,8+2*P3+1:8+3*P3) .+ im .* view(beta,8+3*P3+1:8+4*P3)) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+4*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+5*P1*P3), P1, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+5*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+6*P1*P3), P1, P3)
    view(v,2:M1, M2+1, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,M1+1, M2+1, 2:M3) .= (view(beta,8+6*P3+1:8+7*P3) .+ im .* view(beta,8+7*P3+1:8+8*P3)) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+6*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+7*P1*P3), P1, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+7*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3), P1, P3)
    view(v,M1+2:N1, M2+1, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,1      , M2+1, M3+2:N3) .= conj.(view(v,1, M2+1, M3:-1:2))
    view(v,2:M1   , M2+1, M3+2:N3) .= conj.(view(v,N1:-1:M1+2, M2+1, M3:-1:2))
    view(v,M1+1   , M2+1, M3+2:N3) .= conj.(view(v,M1+1, M2+1, M3:-1:2))
    view(v,M1+2:N1, M2+1, M3+2:N3) .= conj.(view(v,M1:-1:2, M2+1, M3:-1:2))

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+1:8+8*P3+8*P2+8*P1+P2*P3), P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+P2*P3+1:8+8*P3+8*P2+8*P1+2*P2*P3), P2, P3)
    view(v,1, 2:M2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+2*P2*P3+1:8+8*P3+8*P2+8*P1+3*P2*P3), P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+3*P2*P3+1:8+8*P3+8*P2+8*P1+4*P2*P3), P2, P3)
    view(v,1, M2+2:N2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,1, 2:M2   , M3+2:N3) .= conj.(view(v,1, N2:-1:M2+2, M3:-1:2))
    view(v,1, M2+2:N2, M3+2:N3) .= conj.(view(v,1, M2:-1:2, M3:-1:2))

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+4*P2*P3+1:8+8*P3+8*P2+8*P1+5*P2*P3), P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+5*P2*P3+1:8+8*P3+8*P2+8*P1+6*P2*P3), P2, P3)
    view(v,M1+1, 2:M2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+6*P2*P3+1:8+8*P3+8*P2+8*P1+7*P2*P3), P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+7*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3), P2, P3)
    view(v,M1+1, M2+2:N2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,M1+1, 2:M2   , M3+2:N3) .= conj.(view(v,M1+1, N2:-1:M2+2, M3:-1:2))
    view(v,M1+1, M2+2:N2, M3+2:N3) .= conj.(view(v,M1+1, M2:-1:2, M3:-1:2))

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+P1*P2*P3), P1, P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+2*P1*P2*P3), P1, P2, P3)
    view(v,2:M1, 2:M2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,M1+2:N1, M2+2:N2, M3+2:N3) .= conj.(view(v,M1:-1:2, M2:-1:2, M3:-1:2))

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+2*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+3*P1*P2*P3), P1, P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+3*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+4*P1*P2*P3), P1, P2, P3)
    view(v,M1+2:N1, 2:M2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,2:M1, M2+2:N2, M3+2:N3) .= conj.(view(v,N1:-1:M1+2, M2:-1:2, M3:-1:2))

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+4*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+5*P1*P2*P3), P1, P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+5*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+6*P1*P2*P3), P1, P2, P3)
    view(v,2:M1, M2+2:N2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,M1+2:N1, 2:M2, M3+2:N3) .= conj.(view(v,M1:-1:2, N2:-1:M2+2, M3:-1:2))

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+6*P1*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+7*P1*P2*P3), P1, P2, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2+7*P1*P2*P3+1:N1*N2*N3), P1, P2, P3)
    view(v,M1+2:N1, M2+2:N2, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,2:M1, 2:M2, M3+2:N3) .= conj.(view(v,N1:-1:M1+2, N2:-1:M2+2, M3:-1:2))
    return v
end

function beta_to_DFT_3d(beta::StridedArray{Float64}, size)
    N1 = size[1]
    N2 = size[2]
    N3 = size[3]
    v = Array{ComplexF64, 3}(undef, N1, N2, N3)
    beta_to_DFT_3d!(v, beta, size)
end
