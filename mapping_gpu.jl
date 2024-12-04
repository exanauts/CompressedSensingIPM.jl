using CUDA

# DFT_to_beta
function DFT_to_beta_1d!(beta::CuVector{Float64}, v::CuVector{ComplexF64}, size; rdft::Bool=false)
    N = size[1]
    M = N ÷ 2
    view(beta, 1:2) .= real.(view(v, 1:M:M+1))
    view(beta, 3:M+1) .= sqrt(2) .* real.(view(v, 2:M))
    view(beta, M+2:N) .= sqrt(2) .* imag.(view(v, 2:M))
    return beta
end

function DFT_to_beta_2d!(beta::CuVector{Float64}, v::CuMatrix{ComplexF64}, size; rdft::Bool=false)
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

# beta_to_DFT
function beta_to_DFT_1d!(v::CuVector{ComplexF64}, beta::StridedCuVector{Float64}, size; rdft::Bool=false)
    N = size[1]
    M = N ÷ 2
    view(v, 1:M:M+1) .= view(beta, 1:2)
    beta_r = view(beta, 3:M+1)
    beta_c = view(beta, M+2:N)
    view(v, 2:M     ) .= (beta_r .+ im .* beta_c) ./ sqrt(2)
    if !rdft
        view(v, N:-1:M+2) .= (beta_r .- im .* beta_c) ./ sqrt(2)
    end
    return v
end

function beta_to_DFT_2d!(v::CuMatrix{ComplexF64}, beta::StridedCuVector{Float64}, size; rdft::Bool=false)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    P1 = M1 - 1
    P2 = M2 - 1

    view(v,1:M1:M1+1,M2+1) .= view(beta, 2:2:4)
    view(v,1:M1:M1+1) .= view(beta, 1:2:3)

    beta_r = view(beta,4+4*P2+1:4+4*P2+P1)
    beta_c = view(beta,4+4*P2+P1+1:4+4*P2+2*P1)
    view(v,2:M1,1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)
    if !rdft
        view(v,N1:-1:M1+2,1) .= (beta_r .- im .* beta_c) ./ sqrt(2)
    end

    beta_r = view(beta,4+1:4+M2-1)
    beta_c = view(beta,4+P2+1:4+2*P2)
    view(v,1, 2:M2) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,4+4*P2+4*P1+1:4+4*P2+4*P1+P1*P2), P1, P2)
    beta_c = reshape(view(beta,4+4*P2+4*P1+P1*P2+1:4+4*P2+4*P1+2*P1*P2), P1, P2)
    view(v,2:M1, 2:M2) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = view(beta,4+2*P2+1:4+3*P2)
    beta_c = view(beta,4+3*P2+1:4+4*P2)
    view(v,M1+1, 2:M2) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,4+4*P2+4*P1+2*P1*P2+1:4+4*P2+4*P1+3*P1*P2), P1, P2)
    beta_c = reshape(view(beta,4+4*P2+4*P1+3*P1*P2+1:N1*N2), P1, P2)
    view(v,2:M1,M2+2:N2) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = view(beta,4+4*P2+2*P1+1:4+4*P2+3*P1)
    beta_c = view(beta,4+4*P2+3*P1+1:4+4*P2+4*P1)
    view(v,2:M1,M2+1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)
    if !rdft
        view(v,N1:-1:M1+2,M2+1) .= (beta_r .- im .* beta_c) ./ sqrt(2)
    end

    view(v,1,M2+2:N2) .= conj.(view(v,1,M2:-1:2))
    view(v,M1+1,M2+2:N2) .= conj.(view(v,M1+1,M2:-1:2))
    if !rdft
        view(v,N1:-1:M1+2,M2:-1:2) .= conj.(view(v,2:M1,M2+2:N2))
        view(v,M1+2:N1,M2+2:N2) .= conj.(view(v,M1:-1:2,M2:-1:2))
    end
    return v
end

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

    view(v,1   , 1   , 1:M3:M3+1) .= view(beta,1:2)
    view(v,1   , M2+1, 1:M3:M3+1) .= view(beta,3:4)
    view(v,M1+1, 1   , 1:M3:M3+1) .= view(beta,5:6)
    view(v,M1+1, M2+1, 1:M3:M3+1) .= view(beta,7:8)

    beta_r = view(beta,8+8*P3+8*P2+1:8+8*P3+8*P2+P1)
    beta_c = view(beta,8+8*P3+8*P2+P1+1:8+8*P3+8*P2+2*P1)
    view(v,2:M1, 1, 1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = view(beta,8+8*P3+8*P2+P1:-1:8+8*P3+8*P2+1)
    beta_c = view(beta,8+8*P3+8*P2+2*P1:-1:8+8*P3+8*P2+P1+1)
    view(v,M1+2:N1, 1, 1) .= (beta_r .- im .* beta_c) ./ sqrt(2)

    beta_r = view(beta,8+8*P3+1:8+8*P3+P2)
    beta_c = view(beta,8+8*P3+P2+1:8+8*P3+2*P2)
    view(v,1, 2:M2, 1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+P1*P2), P1, P2)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+2*P1*P2), P1, P2)
    view(v,2:M1, 2:M2, 1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = view(beta,8+8*P3+4*P2+1:8+8*P3+5*P2)
    beta_c = view(beta,8+8*P3+5*P2+1:8+8*P3+6*P2)
    view(v,M1+1, 2:M2, 1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+2*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+3*P1*P2), P1, P2)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+3*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+4*P1*P2), P1, P2)
    view(v,M1+2:N1, 2:M2, 1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,2:M1   , M2+1, 1) .= (view(beta,8+8*P3+8*P2+4*P1+1:8+8*P3+8*P2+5*P1) .+ im .* view(beta,8+8*P3+8*P2+5*P1+1:8+8*P3+8*P2+6*P1)) ./ sqrt(2)
    view(v,M1+2:N1, M2+1, 1) .= (view(beta,8+8*P3+8*P2+5*P1:-1:8+8*P3+8*P2+4*P1+1) .- im .* view(beta,8+8*P3+8*P2+6*P1:-1:8+8*P3+8*P2+5*P1+1)) ./ sqrt(2)

    view(v,1      , M2+2:N2, 1) .= conj.(view(v,1, M2:-1:2, 1))
    view(v,2:M1   , M2+2:N2, 1) .= conj.(view(v,N1:-1:M1+2, M2:-1:2, 1))
    view(v,M1+1   , M2+2:N2, 1) .= conj.(view(v,M1+1, M2:-1:2, 1))
    view(v,M1+2:N1, M2+2:N2, 1) .= conj.(view(v,M1:-1:2, M2:-1:2, 1))

    view(v,2:M1   , 1, M3+1) .= (view(beta,8+8*P3+8*P2+2*P1+1:8+8*P3+8*P2+3*P1) .+ im .* view(beta,8+8*P3+8*P2+3*P1+1:8+8*P3+8*P2+4*P1)) ./ sqrt(2)
    view(v,M1+2:N1, 1, M3+1) .= (view(beta,8+8*P3+8*P2+3*P1:-1:8+8*P3+8*P2+2*P1+1) .- im .* view(beta,8+8*P3+8*P2+4*P1:-1:8+8*P3+8*P2+3*P1+1)) ./ sqrt(2)

    view(v,1, 2:M2, M3+1) .= (view(beta,8+8*P3+2*P2+1:8+8*P3+3*P2) .+ im .* view(beta,8+8*P3+3*P2+1:8+8*P3+4*P2)) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+4*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+5*P1*P2), P1, P2)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+5*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+6*P1*P2), P1, P2)
    view(v,2:M1, 2:M2, M3+1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,M1+1, 2:M2, M3+1) .= (view(beta,8+8*P3+6*P2+1:8+8*P3+7*P2) .+ im .* view(beta,8+8*P3+7*P2+1:8+8*P3+8*P2)) ./sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+6*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+7*P1*P2), P1, P2)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+7*P1*P2+1:8+8*P3+8*P2+8*P1+8*P2*P3+8*P1*P3+8*P1*P2), P1, P2)
    view(v,M1+2:N1, 2:M2, M3+1) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,2:M1   , M2+1, M3+1) .= (view(beta,8+8*P3+8*P2+6*P1+1:8+8*P3+8*P2+7*P1) .+ im .* view(beta,8+8*P3+8*P2+7*P1+1:8+8*P3+8*P2+8*P1)) ./ sqrt(2)
    view(v,M1+2:N1, M2+1, M3+1) .= (view(beta,8+8*P3+8*P2+7*P1:-1:8+8*P3+8*P2+6*P1+1) .- im .* view(beta,8+8*P3+8*P2+8*P1:-1:8+8*P3+8*P2+7*P1+1)) ./ sqrt(2)

    view(v,1      , M2+2:N2, M3+1) .= conj.(view(v,1, M2:-1:2, M3+1))
    view(v,2:M1   , M2+2:N2, M3+1) .= conj.(view(v,N1:-1:M1+2, M2:-1:2, M3+1))
    view(v,M1+1   , M2+2:N2, M3+1) .= conj.(view(v,M1+1, M2:-1:2, M3+1))
    view(v,M1+2:N1, M2+2:N2, M3+1) .= conj.(view(v,M1:-1:2, M2:-1:2, M3+1))

    view(v,1, 1, 2:M3) .= (view(beta,9:8+P3) .+ im .* view(beta,8+P3+1:8+2*P3)) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+P1*P3), P1, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+2*P1*P3), P1, P3)
    view(v,2:M1, 1, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

    view(v,M1+1, 1, 2:M3) .= (view(beta,8+4*P3+1:8+5*P3) .+ im .* view(beta,8+5*P3+1:8+6*P3)) ./ sqrt(2)

    beta_r = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+2*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+3*P1*P3), P1, P3)
    beta_c = reshape(view(beta,8+8*P3+8*P2+8*P1+8*P2*P3+3*P1*P3+1:8+8*P3+8*P2+8*P1+8*P2*P3+4*P1*P3), P1, P3)
    view(v,M1+2:N1, 1, 2:M3) .= (beta_r .+ im .* beta_c) ./ sqrt(2)

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
