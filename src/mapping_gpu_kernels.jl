# dim = 1
function DFT_to_beta_1d!(beta::CuVector{Float64}, v::CuVector{ComplexF64}, size; rdft::Bool=false)
    N = size[1]
    M = N ÷ 2
    backend = KA.get_backend(beta)
    kernel = kernel_DFT_to_beta_1d!(backend)
    kernel(beta, v, N, M; rdft, ndrange=N)
    KA.synchronize(backend)
    return beta
end

@kernel function kernel_DFT_to_beta_1d!(beta::CuVector{Float64}, v, N, M; rdft::Bool=false)
    i = @index(Global)
    if i == 1
        beta[i] = real(v[1])
    elseif i == 2
        beta[i] = real(v[M+1])
    elseif 3 <= i <= M+1
        beta[i] = sqrt(2) * real(v[i-1])
    else
        beta[i] = sqrt(2) * imag(v[i-M])
    end
    return nothing
end

# dim = 2
function DFT_to_beta_2d!(beta::CuVector{Float64}, v::CuMatrix{ComplexF64}, size; rdft::Bool=false)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    PP = P1 * P2
    backend = KA.get_backend(beta)
    kernel = kernel_DFT_to_beta_2d!(backend)
    kernel(beta, v, N1, N2, M1, M2, P1, P2, PP; rdft, ndrange=N1*N2)
    KA.synchronize(backend)
    return beta
end

@kernel function kernel_DFT_to_beta_2d!(beta::CuVector{Float64}, v::CuMatrix{ComplexF64}, N1, N2, M1, M2, P1, P2, PP; rdft::Bool=false)
    i = @index(Global)
    # vertex
    if i == 1
        beta[i] = real(v[1, 1])
    elseif i == 2
        beta[i] = real(v[1, M2+1])
    elseif i == 3
        beta[i] = real(v[M1+1, 1])
    elseif i == 4
        beta[i] = real(v[M1+1, M2+1])
    # edge
    elseif 5 <= i <= (4+P2)
        beta[i] = sqrt(2) * real(v[1, i-3])
    elseif (5+P2) <= i <= (4+2*P2)
        beta[i] = sqrt(2) * imag(v[1, i-P2-3])
    elseif (5+2*P2) <= i <= (4+3*P2)
        beta[i] = sqrt(2) * real(v[M1+1, i-2*P2-3])
    elseif (5+3*P2) <= i <= (4+4*P2)
        beta[i] = sqrt(2) * imag(v[M1+1, i-3*P2-3])
    elseif (5+4*P2) <= i <= (4+4*P2+P1)
        beta[i] = sqrt(2) * real(v[i-4*P2-3, 1])
    elseif (5+4*P2+P1) <= i <= (4+4*P2+2*P1)
        beta[i] = sqrt(2) * imag(v[i-4*P2-P1-3, 1])
    elseif (5+4*P2+2*P1) <= i <= (4+4*P2+3*P1)
        beta[i] = sqrt(2) * real(v[i-4*P2-2*P1-3, M2+1])
    elseif (5+4*P2+3*P1) <= i <= (4+4*P2+4*P1)
        beta[i] = sqrt(2) * imag(v[i-4*P2-3*P1-3, M2+1])
    # center
    elseif (5+4*P2+4*P1) <= i <= (4+4*P2+4*P1+PP)
        j = i - (4+4*P2+4*P1)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, i2+1])
    elseif (5+4*P2+4*P1+PP) <= i <= (4+4*P2+4*P1+2*PP)
        j = i - (4+4*P2+4*P1+PP)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, i2+1])
    elseif (5+4*P2+4*P1+2*PP) <= i <= (4+4*P2+4*P1+3*PP)
        j = i - (4+4*P2+4*P1+2*PP)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, i2+(M2+1)])
    else
        j = i - (4+4*P2+4*P1+3*PP)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, i2+(M2+1)])
    end
    return nothing
end

function DFT_to_beta_3d!(beta::CuVector{Float64}, v::CuArray{ComplexF64,3}, size; rdft::Bool=false)
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
    backend = KA.get_backend(beta)
    kernel = kernel_DFT_to_beta_3d!(backend)
    kernel(beta, v, N1, N2, N3, M1, M2, M3, P1, P2, P3, P23, P13, P12, P123; rdft, ndrange=N1*N2*N3)
    KA.synchronize(backend)
    return beta
end

@kernel function kernel_DFT_to_beta_3d!(beta::CuVector{Float64}, v::CuArray{ComplexF64,3}, N1, N2, N3, M1, M2, M3, P1, P2, P3, P23, P13, P12, P123; rdft::Bool=false)
    i = @index(Global)
    if i == 1
        beta[i] = real(v[1   , 1   , 1   ])
    elseif i == 2
        beta[i] = real(v[1   , 1   , M3+1])
    elseif i == 3
        beta[i] = real(v[1   , M2+1, 1   ])
    elseif i == 4
        beta[i] = real(v[1   , M2+1, M3+1])
    elseif i == 5
        beta[i] = real(v[M1+1, 1   , 1   ])
    elseif i == 6
        beta[i] = real(v[M1+1, 1   , M3+1])
    elseif i == 7
        beta[i] = real(v[M1+1, M2+1, 1   ])
    elseif i == 8
        beta[i] = real(v[M1+1, M2+1, M3+1])
    elseif 9 <= i <= 8+P3
        beta[i] = sqrt(2) * real(v[1, 1, i-7])
    elseif 9+P3 <= i <= 8+2*P3
        beta[i] = sqrt(2) * imag(v[1, 1, i-P3-7])
    elseif 9+2*P3 <= i <= 8+3*P3
        beta[i] = sqrt(2) * real(v[1, M2+1, i-2*P3-7])
    elseif 9+3*P3 <= i <= 8+4*P3
        beta[i] = sqrt(2) * imag(v[1, M2+1, i-3*P3-7])
    elseif 9+4*P3 <= i <= 8+5*P3
        beta[i] = sqrt(2) * real(v[M1+1, 1, i-4*P3-7])
    elseif 9+5*P3 <= i <= 8+6*P3
        beta[i] = sqrt(2) * imag(v[M1+1, 1, i-5*P3-7])
    elseif 9+6*P3 <= i <= 8+7*P3
        beta[i] = sqrt(2) * real(v[M1+1, M2+1, i-6*P3-7])
    elseif 9+7*P3 <= i <= 8+8*P3
        beta[i] = sqrt(2) * imag(v[M1+1, M2+1, i-7*P3-7])
    elseif 9+8*P3 <= i <= 8+8*P3+P2
        beta[i] = sqrt(2) * real(v[1, i-8*P3-7, 1])
    elseif 9+8*P3+P2 <= i <= 8+8*P3+2*P2
        beta[i] = sqrt(2) * imag(v[1, i-8*P3-P2-7, 1])
    elseif 9+8*P3+2*P2 <= i <= 8+8*P3+3*P2
        beta[i] = sqrt(2) * real(v[1, i-8*P3-2*P2-7, M3+1])
    elseif 9+8*P3+3*P2 <= i <= 8+8*P3+4*P2
        beta[i] = sqrt(2) * imag(v[1, i-8*P3-3*P2-7, M3+1])
    elseif 9+8*P3+4*P2 <= i <= 8+8*P3+5*P2
        beta[i] = sqrt(2) * real(v[M1+1, i-8*P3-4*P2-7, 1])
    elseif 9+8*P3+5*P2 <= i <= 8+8*P3+6*P2
        beta[i] = sqrt(2) * imag(v[M1+1, i-8*P3-5*P2-7, 1])
    elseif 9+8*P3+6*P2 <= i <= 8+8*P3+7*P2
        beta[i] = sqrt(2) * real(v[M1+1, i-8*P3-6*P2-7, M3+1])
    elseif 9+8*P3+7*P2 <= i <= 8+8*P3+8*P2
        beta[i] = sqrt(2) * imag(v[M1+1, i-8*P3-7*P2-7, M3+1])
    elseif 9+8*P3+8*P2 <= i <= 8+8*P3+8*P2+P1
        beta[i] = sqrt(2) * real(v[i-8*P3-8*P2-7, 1, 1])
    elseif 9+8*P3+8*P2+P1 <= i <= 8+8*P3+8*P2+2*P1
        beta[i] = sqrt(2) * imag(v[i-8*P3-8*P2-P1-7, 1, 1])
    elseif 9+8*P3+8*P2+2*P1 <= i <= 8+8*P3+8*P2+3*P1
        beta[i] = sqrt(2) * real(v[i-8*P3-8*P2-2*P1-7, 1, M3+1])
    elseif 9+8*P3+8*P2+3*P1 <= i <= 8+8*P3+8*P2+4*P1
        beta[i] = sqrt(2) * imag(v[i-8*P3-8*P2-3*P1-7, 1, M3+1])
    elseif 9+8*P3+8*P2+4*P1 <= i <= 8+8*P3+8*P2+5*P1
        beta[i] = sqrt(2) * real(v[i-8*P3-8*P2-4*P1-7, M2+1, 1])
    elseif 9+8*P3+8*P2+5*P1 <= i <= 8+8*P3+8*P2+6*P1
        beta[i] = sqrt(2) * imag(v[i-8*P3-8*P2-5*P1-7, M2+1, 1])
    elseif 9+8*P3+8*P2+6*P1 <= i <= 8+8*P3+8*P2+7*P1
        beta[i] = sqrt(2) * real(v[i-8*P3-8*P2-6*P1-7, M2+1, M3+1])
    elseif 9+8*P3+8*P2+7*P1 <= i <= 8+8*P3+8*P2+8*P1
        beta[i] = sqrt(2) * imag(v[i-8*P3-8*P2-7*P1-7, M2+1, M3+1])
    elseif 9+8*P3+8*P2+8*P1 <= i <= 8+8*P3+8*P2+8*P1+P23
        j = i-(8+8*P3+8*P2+8*P1)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * real(v[1, i2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+P23 <= i <= 8+8*P3+8*P2+8*P1+2*P23
        j = i-(8+8*P3+8*P2+8*P1+P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * imag(v[1, i2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+2*P23 <= i <= 8+8*P3+8*P2+8*P1+3*P23
        j = i-(8+8*P3+8*P2+8*P1+2*P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * real(v[1, i2+(M2+1), i3+1])
    elseif 9+8*P3+8*P2+8*P1+3*P23 <= i <= 8+8*P3+8*P2+8*P1+4*P23
        j = i-(8+8*P3+8*P2+8*P1+3*P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * imag(v[1, i2+(M2+1), i3+1])
    elseif 9+8*P3+8*P2+8*P1+4*P23 <= i <= 8+8*P3+8*P2+8*P1+5*P23
        j = i-(8+8*P3+8*P2+8*P1+4*P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * real(v[M1+1, i2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+5*P23 <= i <= 8+8*P3+8*P2+8*P1+6*P23
        j = i-(8+8*P3+8*P2+8*P1+5*P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * imag(v[M1+1, i2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+6*P23 <= i <= 8+8*P3+8*P2+8*P1+7*P23
        j = i-(8+8*P3+8*P2+8*P1+6*P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * real(v[M1+1, i2+(M2+1), i3+1])
    elseif 9+8*P3+8*P2+8*P1+7*P23 <= i <= 8+8*P3+8*P2+8*P1+8*P23
        j = i-(8+8*P3+8*P2+8*P1+7*P23)
        i3 = div(j-1, P2) + 1
        i2 = mod(j-1, P2) + 1
        beta[i] = sqrt(2) * imag(v[M1+1, i2+(M2+1), i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23 <= i <= 8+8*P3+8*P2+8*P1+8*P23+P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, 1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+2*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, 1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+2*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+3*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+2*P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[M1+1-i1, 1, N3+1-i3])
    elseif 9+8*P3+8*P2+8*P1+8*P23+3*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+4*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+3*P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = -sqrt(2) * imag(v[M1+1-i1, 1, N3+1-i3])
    elseif 9+8*P3+8*P2+8*P1+8*P23+4*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+5*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+4*P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, M2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+5*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+6*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+5*P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, M2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+6*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+7*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+6*P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[M1+1-i1, M2+1, N3+1-i3])
    elseif 9+8*P3+8*P2+8*P1+8*P23+7*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13
        j = i-(8+8*P3+8*P2+8*P1+8*P23+7*P13)
        i3 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = -sqrt(2) * imag(v[M1+1-i1, M2+1, N3+1-i3])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, i2+1, 1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, i2+1, 1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[M1+1-i1, N2+1-i2, 1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = -sqrt(2) * imag(v[M1+1-i1, N2+1-i2, 1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, i2+1, M3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, i2+1, M3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = sqrt(2) * real(v[M1+1-i1, N2+1-i2, M3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12)
        i2 = div(j-1, P1) + 1
        i1 = mod(j-1, P1) + 1
        beta[i] = -sqrt(2) * imag(v[M1+1-i1, N2+1-i2, M3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, i2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+P123 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, i2+1, i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = sqrt(2) * real(v[M1+1-i1, N2+1-i2, N3+1-i3])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = -sqrt(2) * imag(v[M1+1-i1, N2+1-i2, N3+1-i3])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = sqrt(2) * real(v[i1+1, i2+(M2+1), i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = sqrt(2) * imag(v[i1+1, i2+(M2+1), i3+1])
    elseif 9+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123 <= i <= 8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+7*P123
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = sqrt(2) * real(v[M1+1-i1, M2+1-i2, N3+1-i3])
    else
        j = i-(8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+7*P123)
        i3 = div(j-1, P12) + 1
        k = mod(j-1, P12) + 1
        i2 = div(k-1, P1) + 1
        i1 = mod(k-1, P1) + 1
        beta[i] = -sqrt(2) * imag(v[M1+1-i1, M2+1-i2, N3+1-i3])
    end
    return nothing
end

# dim = 1
function beta_to_DFT_1d!(v::CuVector{ComplexF64}, beta::StridedCuVector{Float64}, size; rdft::Bool=false)
    N = size[1]
    M = N ÷ 2
    backend = KA.get_backend(v)
    kernel = kernel_beta_to_DFT_1d!(backend)
    kernel(v, beta, N, M; rdft, ndrange = rdft ? M+1 : N)
    KA.synchronize(backend)
    return v
end

@kernel function kernel_beta_to_DFT_1d!(v::CuVector{ComplexF64}, beta::StridedCuVector{Float64}, N, M; rdft::Bool=false)
    i = @index(Global)
    if i == 1
        v[i] = beta[1]
    elseif i == M+1
        v[i] = beta[2]
    elseif 2 <= i <= M
        v[i] = (beta[i+1] + im*beta[M+i]) / sqrt(2)
    else
        v[i] = (beta[N+3-i] - im*beta[(N+M+2)-i]) / sqrt(2)
    end
    return nothing
end

# dim = 2
function beta_to_DFT_2d!(v::CuMatrix{ComplexF64}, beta::StridedCuVector{Float64}, size; rdft::Bool=false)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1
    PP = P1 * P2
    backend = KA.get_backend(v)
    kernel = kernel_beta_to_DFT_2d!(backend)
    kernel(v, beta, N1, N2, M1, M2, P1, P2, PP; rdft, ndrange = rdft ? (M1+1,N2) : (N1,N2))
    KA.synchronize(backend)
    return v
end

@kernel function kernel_beta_to_DFT_2d!(v::CuMatrix{ComplexF64}, beta::StridedCuVector{Float64}, N1, N2, M1, M2, P1, P2, PP; rdft::Bool=false)
    i, j = @index(Global)
    # vertex
    if i == 1
        if j == 1
            v[i,j] = beta[1]
        elseif j == M2+1
            v[i,j] = beta[2]
        end
    elseif i == M1+1
        if j == 1
            v[i,j] = beta[3]
        elseif j == M2+1
            v[i,j] = beta[4]
        end
    end

    # edge
    if i == 1
        if 2 <= j <= M2
            offset = 3
            v[i,j] = (beta[j+offset] + im * beta[j+offset+P2]) / sqrt(2)
        elseif M2+2 <= j <= N2
            offset = N2+5
            v[i,j] = (beta[offset-j] - im * beta[offset+P2-j]) / sqrt(2)
        end
    elseif i == M1+1
        if 2 <= j <= M2
            offset = 3 + 2*P2
            v[i,j] = (beta[j+offset] + im * beta[j+offset+P2]) / sqrt(2)
        elseif M2+2 <= j <= N2
            offset = N2+5 + 2*P2
            v[i,j] = (beta[offset-j] - im * beta[offset+P2-j]) / sqrt(2)
        end
    end

    if j == 1
        if 2 <= i <= M1
            offset = 3 + 4*P2
            v[i,j] = (beta[i+offset] + im * beta[i+offset+P1]) / sqrt(2)
        elseif M1+2 <= i <= N1
            offset = N1+5 + 4*P2
            v[i,j] = (beta[offset-i] - im * beta[offset+P1-i]) / sqrt(2)
        end
    elseif j == M2+1
        if 2 <= i <= M1
            offset = 3 + 4*P2 + 2*P1
            v[i,j] = (beta[i+offset] + im * beta[i+offset+P1]) / sqrt(2)
        elseif M1+2 <= i <= N1
            offset = N1+5 + 4*P2 + 2*P1
            v[i,j] = (beta[offset-i] - im * beta[offset+P1-i]) / sqrt(2)
        end
    end

    # center
    if 2 <= i <= M1
        if 2 <= j <= M2
            offset = 4 + 4*P2 + 4*P1
            index = (j-2)*P1 + (i-1)
            v[i,j] = (beta[offset+index] + im * beta[offset+PP+index]) / sqrt(2)
        elseif M2+2 <= j <= N2
            offset = 4 + 4*P2 + 4*P1 + 2*PP
            index = (j-M2-2)*P1 + (i-1)
            v[i,j] = (beta[offset+index] + im * beta[offset+PP+index]) / sqrt(2)
        end
    elseif M1+2 <= i <= N1
        if 2 <= j <= M2
            offset = 4 + 4*P2 + 4*P1 + 3*PP
            index = (j-2)*P1 + (i-M1-2)
            v[i,j] = (beta[offset-index] - im * beta[offset+PP-index]) / sqrt(2)
        elseif M2+2 <= j <= N2
            offset = 4 + 4*P2 + 4*P1 + PP
            index = (j-M2-2)*P1 + (i-M1-2)
            v[i,j] = (beta[offset-index] - im * beta[offset+PP-index]) / sqrt(2)
        end
    end
    return nothing
end

# dim = 3
function beta_to_DFT_3d!(v::CuArray{ComplexF64, 3}, beta::StridedCuVector{Float64}, size; rdft::Bool=false)
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
    backend = KA.get_backend(v)
    kernel = kernel_beta_to_DFT_3d!(backend)
    kernel(v, beta, N1, N2, N3, M1, M2, M3, P1, P2, P3, P23, P13, P12, P123, rdft, ndrange = rdft ? (M1+1,N2,N3) : (N1,N2,N3))
    KA.synchronize(backend)
    return v
end

@kernel function kernel_beta_to_DFT_3d!(v::CuArray{ComplexF64, 3}, beta::StridedCuVector{Float64}, N1, N2, N3, M1, M2, M3, P1, P2, P3, P23, P13, P12, P123; rdft::Bool=false)
    i, j, k = @index(Global)
    #vertex
    if i == 1
        if j == 1
            if k == 1
                v[i,j,k] = beta[1]
            elseif k == M3+1
                v[i,j,k] = beta[2]
            end
        elseif j == M2+1
            if k == 1
                v[i,j,k] = beta[3]
            elseif k == M3+1
                v[i,j,k] = beta[4]
            end
        end
    elseif i == M1+1
        if j == 1
            if k == 1
                v[i,j,k] = beta[5]
            elseif k == M3+1
                v[i,j,k] = beta[6]
            end
        elseif j == M2+1
            if k == 1
                v[i,j,k] = beta[7]
            elseif k == (M3+1)
                v[i,j,k] = beta[8]
            end
        end
    end

    #edge
    if i == 1
        if j == 1
            if 2 <= k <= M3
                v[i,j,k] = (beta[k+7] + im*beta[k+7+P3])/sqrt(2)
            elseif (M3+2) <= k <= N3
                v[i,j,k] = (beta[N3+9-k] - im*beta[N3+9+P3-k])/sqrt(2)
            end
        elseif j == (M2+1)
            if 2 <= k <= M3
                v[i,j,k] = (beta[k+7+2*P3] + im*beta[k+7+3*P3])/sqrt(2)
            elseif (M3+2) <= k <= N3
                v[i,j,k] = (beta[N3+9+2*P3-k] - im*beta[N3+9+3*P3-k])/sqrt(2)
            end
        end
    elseif i == (M1+1)
        if j == 1
            if 2 <= k <= M3
                v[i,j,k] = (beta[k+7+4*P3] + im*beta[k+7+5*P3])/sqrt(2)
            elseif (M3+2) <= k <= N3
                v[i,j,k] = (beta[N3+9+4*P3-k] - im*beta[N3+9+5*P3-k])/sqrt(2)
            end
        elseif j == (M2+1)
            if 2 <= k <= M3
                v[i,j,k] = (beta[k+7+6*P3] + im*beta[k+7+7*P3])/sqrt(2)
            elseif (M3+2) <= k <= N3
                v[i,j,k] = (beta[N3+9+6*P3-k] - im*beta[N3+9+7*P3-k])/sqrt(2)
            end
        end
    end

    if i == 1
        if k == 1
            if 2 <= j <= M2
                v[i,j,k] = (beta[j+7+8*P3] + im*beta[j+7+8*P3+P2])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[N2+9+8*P3-j] - im*beta[N2+9+8*P3+P2-j])/sqrt(2)
            end
        elseif k == (M3+1)
            if 2 <= j <= M2
                v[i,j,k] = (beta[j+7+8*P3+2*P2] + im*beta[j+7+8*P3+3*P2])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[N2+9+8*P3+2*P2-j] - im*beta[N2+9+8*P3+3*P2-j])/sqrt(2)
            end
        end
    elseif i == (M1+1)
        if k == 1
            if 2 <= j <= M2
                v[i,j,k] = (beta[j+7+8*P3+4*P2] + im*beta[j+7+8*P3+5*P2])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[N2+9+8*P3+4*P2-j] - im*beta[N2+9+8*P3+5*P2-j])/sqrt(2)
            end
        elseif k == (M3+1)
            if 2 <= j <= M2
                v[i,j,k] = (beta[j+7+8*P3+6*P2] + im*beta[j+7+8*P3+7*P2])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[N2+9+8*P3+6*P2-j] - im*beta[N2+9+8*P3+7*P2-j])/sqrt(2)
            end
        end
    end

    if j == 1
        if k == 1
            if 2 <= i <= M1
                v[i,j,k] = (beta[i+7+8*P3+8*P2] + im*beta[i+7+8*P3+8*P2+P1])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[N1+9+8*P3+8*P2-i] - im*beta[N1+9+8*P3+8*P2+P1-i])/sqrt(2)
            end
        elseif k == (M3+1)
            if 2 <= i <= M1
                v[i,j,k] = (beta[i+7+8*P3+8*P2+2*P1] + im*beta[i+7+8*P3+8*P2+3*P1])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[N1+9+8*P3+8*P2+2*P1-i] - im*beta[N1+9+8*P3+8*P2+3*P1-i])/sqrt(2)
            end
        end
    elseif j == (M2+1)
        if k == 1
            if 2 <= i <= M1
                v[i,j,k] = (beta[i+7+8*P3+8*P2+4*P1] + im*beta[i+7+8*P3+8*P2+5*P1])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[N1+9+8*P3+8*P2+4*P1-i] - im*beta[N1+9+8*P3+8*P2+5*P1-i])/sqrt(2)
            end
        elseif k == (M3+1)
            if 2 <= i <= M1
                v[i,j,k] = (beta[i+7+8*P3+8*P2+6*P1] + im*beta[i+7+8*P3+8*P2+7*P1])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[N1+9+8*P3+8*P2+6*P1-i] - im*beta[N1+9+8*P3+8*P2+7*P1-i])/sqrt(2)
            end
        end
    end

    #face
    if i == 1
        if 2 <= k <= M3
            if 2 <= j <= M2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+(k-2)*P2+(j-1)] + im*beta[8+8*P3+8*P2+8*P1+P23+(k-2)*P2+(j-1)])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+2*P23+(k-2)*P2+(j-(M2+1))] + im*beta[8+8*P3+8*P2+8*P1+3*P23+(k-2)*P2+(j-(M2+1))])/sqrt(2)
            end
        elseif (M3+2) <= k <= N3
            if 2 <= j <= M2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+2*P23+(N3-k)*P2+(M2+1-j)] - im*beta[8+8*P3+8*P2+8*P1+3*P23+(N3-k)*P2+(M2+1-j)]) / sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+(N3-k)*P2+(N2+1-j)] - im*beta[8+8*P3+8*P2+8*P1+P23+(N3-k)*P2+(N2+1-j)])/sqrt(2)
            end
        end
    elseif i == (M1+1)
        if 2 <= k <= M3
            if 2 <= j <= M2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+4*P23+(k-2)*P2+(j-1)]+im*beta[8+8*P3+8*P2+8*P1+5*P23+(k-2)*P2+(j-1)])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+6*P23+(k-2)*P2+(j-(M2+1))] + im*beta[8+8*P3+8*P2+8*P1+7*P23+(k-2)*P2+(j-(M2+1))])/sqrt(2)
            end
        elseif (M3+2) <= k <= N3
            if 2 <= j <= M2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+6*P23+(N3-k)*P2+(M2+1-j)] - im*beta[8+8*P3+8*P2+8*P1+7*P23+(N3-k)*P2+(M2+1-j)])/sqrt(2)
            elseif (M2+2) <= j <= N2
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+4*P23+(N3-k)*P2+(N2+1-j)] - im*beta[8+8*P3+8*P2+8*P1+5*P23+(N3-k)*P2+(N2+1-j)])/sqrt(2)
            end
        end
    end

    if j == 1
        if 2 <= k <= M3
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+(k-2)*P1+(i-1)] + im*beta[8+8*P3+8*P2+8*P1+8*P23+P13+(k-2)*P1+(i-1)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+2*P13+(k-2)*P1+(i-(M1+1))] + im*beta[8+8*P3+8*P2+8*P1+8*P23+3*P13+(k-2)*P1+(i-(M1+1))]) /sqrt(2)
            end
        elseif (M3+2) <= k <= N3
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+2*P13+(N3-k)*P1+(M1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+3*P13+(N3-k)*P1+(M1+1-i)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+(N3-k)*P1+(N1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+P13+(N3-k)*P1+(N1+1-i)])/sqrt(2)
            end
        end
    elseif j == (M2+1)
        if 2 <= k <= M3
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+4*P13+(k-2)*P1+(i-1)] + im*beta[8+8*P3+8*P2+8*P1+8*P23+5*P13+(k-2)*P1+(i-1)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+6*P13+(k-2)*P1+(i-(M1+1))] + im*beta[8+8*P3+8*P2+8*P1+8*P23+7*P13+(k-2)*P1+(i-(M1+1))]) /sqrt(2)
            end
        elseif (M3+2) <= k <= N3
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+6*P13+(N3-k)*P1+(M1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+7*P13+(N3-k)*P1+(M1+1-i)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+4*P13+(N3-k)*P1+(N1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+5*P13+(N3-k)*P1+(N1+1-i)])/sqrt(2)
            end
        end
    end

    if k == 1
        if 2 <= j <= M2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+(j-2)*P1+(i-1)] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+P12+(j-2)*P1+(i-1)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12+(j-2)*P1+(i-(M1+1))] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12+(j-2)*P1+(i-(M1+1))])/sqrt(2)
            end
        elseif (M2+2) <= j <= N2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+2*P12+(N2-j)*P1+(M1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+3*P12+(N2-j)*P1+(M1+1-i)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+(N2-j)*P1+(N1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+P12+(N2-j)*P1+(N1+1-i)])/sqrt(2)
            end
        end
    elseif k == (M3+1)
        if 2 <= j <= M2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12+(j-2)*P1+(i-1)] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12+(j-2)*P1+(i-1)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12+(j-2)*P1+(i-(M1+1))] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12+(j-2)*P1+(i-(M1+1))])/sqrt(2)
            end
        elseif (M2+2) <= j <= N2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+6*P12+(N2-j)*P1+(M1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+7*P12+(N2-j)*P1+(M1+1-i)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+4*P12+(N2-j)*P1+(N1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+5*P12+(N2-j)*P1+(N1+1-i)])/sqrt(2)
            end
        end
    end

    #center
    if 2 <= k <= M3
        if 2 <= j <= M2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+(k-2)*P12+(j-2)*P1+(i-1)] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+P123+(k-2)*P12+(j-2)*P1+(i-1)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123+(k-2)*P12+(j-2)*P1+(i-(M1+1))] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123+(k-2)*P12+(j-2)*P1+(i-(M1+1))])/sqrt(2)
            end
        elseif (M2+2) <= j <= N2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123+(k-2)*P12+(j-(M2+2))*P1+(i-1)] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123+(k-2)*P12+(j-(M2+2))*P1+(i-1)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123+(k-2)*P12+(j-(M2+2))*P1+(i-(M1+1))] + im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+7*P123+(k-2)*P12+(j-(M2+2))*P1+(i-(M1+1))])/sqrt(2)
            end
        end
    elseif (M3+2) <= k <= N3
        if 2 <= j <= M2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+6*P123+(N3-k)*P12+(M2-j)*P1+(M1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+7*P123+(N3-k)*P12+(M2-j)*P1+(M1+1-i)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+4*P123+(N3-k)*P12+(M2-j)*P1+(N1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+5*P123+(N3-k)*P12+(M2-j)*P1+(N1+1-i)])/sqrt(2)
            end
        elseif (M2+2) <= j <= N2
            if 2 <= i <= M1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+2*P123+(N3-k)*P12+(N2-j)*P1+(M1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+3*P123+(N3-k)*P12+(N2-j)*P1+(M1+1-i)])/sqrt(2)
            elseif (M1+2) <= i <= N1
                v[i,j,k] = (beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+(N3-k)*P12+(N2-j)*P1+(N1+1-i)] - im*beta[8+8*P3+8*P2+8*P1+8*P23+8*P13+8*P12+P123+(N3-k)*P12+(N2-j)*P1+(N1+1-i)])/sqrt(2)
            end
        end
    end
    return nothing
end
