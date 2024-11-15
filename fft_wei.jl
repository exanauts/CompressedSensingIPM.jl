function M_perp_tz_wei(dim, size, z_zero)
    N = prod(size)
    temp = fft(z_zero) ./ sqrt(N)
    beta = DFT_to_beta(dim, size, temp)
    return beta
end

function M_perp_beta_wei(dim, size, beta, idx_missing)
    N = prod(size)
    v = beta_to_DFT(dim, size, beta)
    temp = real.(ifft(v)) .* sqrt(N)
    temp[idx_missing] .= 0
    return temp
end
function M_perpt_M_perp_vec_wei(dim, size, vec, idx_missing)
    temp = M_perp_beta_wei(dim, size, vec, idx_missing)
    temp = M_perp_tz_wei(dim, size, temp)
    return temp
end

function DFT_to_beta_1d_wei(v, size)
    N = size[1]
    M = N ÷ 2
    beta = [real(v[1]);
            real(v[M+1]);
            sqrt(2) .* real.(v[2:M]);
            sqrt(2) .* imag.(v[2:M])]
    return beta
end

function DFT_to_beta_2d_wei(v, size)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    beta = [real(v[1, 1]);
            real(v[1, M2+1]);
            real(v[M1+1, 1]);
            real(v[M1+1, M2+1]);
            sqrt(2) .* real.(v[1, 2:M2]);
            sqrt(2) .* imag.(v[1, 2:M2]);
            sqrt(2) .* real.(v[M1+1, 2:M2]);
            sqrt(2) .* imag.(v[M1+1, 2:M2]);
            sqrt(2) .* real.(v[2:M1, 1]);
            sqrt(2) .* imag.(v[2:M1, 1]);
            sqrt(2) .* real.(v[2:M1, M2+1]);
            sqrt(2) .* imag.(v[2:M1, M2+1]);
            sqrt(2) .* reshape(real.(v[2:M1, 2:M2]), (M1-1) * (M2-1));
            sqrt(2) .* reshape(imag.(v[2:M1, 2:M2]), (M1-1) * (M2-1));
            sqrt(2) .* reshape(real.(v[2:M1, M2+2:N2]), (M1-1) * (M2-1));
            sqrt(2) .* reshape(imag.(v[2:M1, M2+2:N2]), (M1-1) * (M2-1))]
    return beta
end

function beta_to_DFT_1d_wei(beta, size)
    N = size[1]
    M = N ÷ 2
    v = [beta[1];
         (beta[3: M+1] .+ im .* beta[M+2:N]) ./ sqrt(2);
         beta[2];
         reverse((beta[3: M+1] .- im .* beta[M+2:N]) ./ sqrt(2))]
    return v
end

function beta_to_DFT_2d_wei(beta, size)
    N1 = size[1]
    N2 = size[2]
    M1 = N1 ÷ 2
    M2 = N2 ÷ 2
    v = Matrix{Complex{Float64}}(undef, N1, N2)
    v[:,1] = [beta[1];
              ((beta[4+4*(M2-1)+1:4+4*(M2-1)+(M1-1)]).+(im.*(beta[Int(4+4*(M2-1)+(M1-1)+1):Int(4+4*(M2-1)+2*(M1-1))])))./sqrt(2);
              beta[3];
              reverse(((beta[Int(4+4*(M2-1)+1):Int(4+4*(M2-1)+(M1-1))]).-(im.*(beta[Int(4+4*(M2-1)+(M1-1)+1):Int(4+4*(M2-1)+2*(M1-1))])))./sqrt(2))]
    v[:,2:M2] = [transpose((beta[Int(4+1):Int(4+M2-1)]).+(im.*(beta[Int(4+(M2-1)+1):Int(4+2*(M2-1))])))./sqrt(2);
                 reshape(((beta[Int(4+4*(M2-1)+4*(M1-1)+1):Int(4+4*(M2-1)+4*(M1-1)+(M1-1)*(M2-1))]).+(im.*(beta[Int(4+4*(M2-1)+4*(M1-1)+(M1-1)*(M2-1)+1):Int(4+4*(M2-1)+4*(M1-1)+2*(M1-1)*(M2-1))])))./sqrt(2), Int(M1-1), Int(M2-1));
                 transpose((beta[Int(4+2*(M2-1)+1):Int(4+3*(M2-1))]).+(im.*(beta[Int(4+3*(M2-1)+1):Int(4+4*(M2-1))])))./sqrt(2);
                 reverse(reverse(reshape(((beta[Int(4+4*(M2-1)+4*(M1-1)+2*(M1-1)*(M2-1)+1):Int(4+4*(M2-1)+4*(M1-1)+3*(M1-1)*(M2-1))]).-(im.*(beta[Int(4+4*(M2-1)+4*(M1-1)+3*(M1-1)*(M2-1)+1):Int(N1*N2)])))./sqrt(2), Int(M1-1), Int(M2-1)), dims = 1), dims = 2)]
    v[:,M2+1] = [beta[2];
                 ((beta[Int(4+4*(M2-1)+2*(M1-1)+1):Int(4+4*(M2-1)+3*(M1-1))]).+(im.*(beta[Int(4+4*(M2-1)+3*(M1-1)+1):Int(4+4*(M2-1)+4*(M1-1))])))./sqrt(2);
                 beta[4];
                 reverse(((beta[Int(4+4*(M2-1)+2*(M1-1)+1):Int(4+4*(M2-1)+3*(M1-1))]).-(im.*(beta[Int(4+4*(M2-1)+3*(M1-1)+1):Int(4+4*(M2-1)+4*(M1-1))])))./sqrt(2), dims = 1)]
    v[:, M2+2:N2] = [transpose(reverse(conj.(v[1,2:M2])));
                     reverse(reverse(conj.(v[M1+2:N1,2:M2]), dims = 2), dims = 1);
                     transpose(reverse(conj.(v[M1+1,2:M2])));
                     reverse(reverse(conj.(v[2:M1,2:M2]), dims = 2), dims = 1)]
    return v
end
