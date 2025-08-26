struct FFTOperator{R,C,OP,N,IM}
    buffer_real::R
    buffer_complex1::C
    buffer_complex2::C
    op::OP
    DFTdim::Int64
    DFTsize::NTuple{N,Int64}
    index_missing::IM
    fft_timer::Base.RefValue{Float64}
    mapping_timer::Base.RefValue{Float64}
    rdft::Bool
end

function FFTOperator{VT}(nβ, DFTdim, DFTsize, index_missing, rdft) where VT
    T = eltype(VT)
    A_vec = VT(undef, nβ)
    A = reshape(A_vec, DFTsize)
    buffer_real = A
    if rdft == true
        op = plan_rfft(A)
        M1 = (DFTsize[1] ÷ 2)
        if DFTdim == 1
            buffer_complex1 = Complex{T}.(A[1:M1+1])
        elseif DFTdim == 2
            buffer_complex1 = Complex{T}.(A[1:M1+1,:])
        else
            buffer_complex1 = Complex{T}.(A[1:M1+1,:,:])
        end
        buffer_complex2 = buffer_complex1
    else
        op = plan_fft(A)
        buffer_complex1 = Complex{T}.(A)
        buffer_complex2 = copy(buffer_complex1)
    end
    fft_timer = Ref{Float64}(0.0)
    mapping_timer = Ref{Float64}(0.0)

    return FFTOperator(buffer_real, buffer_complex1, buffer_complex2, op, DFTdim, DFTsize, index_missing, fft_timer, mapping_timer, rdft)
end

function M_perpt_z(op_fft::FFTOperator, z)
    return M_perpt_z(op_fft.buffer_real, op_fft.buffer_complex1, op_fft.buffer_complex2, op_fft.op,
                     op_fft.DFTdim, op_fft.DFTsize, z, op_fft.fft_timer, op_fft.mapping_timer; rdft=op_fft.rdft)
end

function M_perp_beta(op_fft::FFTOperator, beta)
    return M_perp_beta(op_fft.buffer_real, op_fft.buffer_complex1, op_fft.buffer_complex2, op_fft.op,
                       op_fft.DFTdim, op_fft.DFTsize, beta, op_fft.index_missing, op_fft.fft_timer,
                       op_fft.mapping_timer; rdft=op_fft.rdft)
end

function M_perpt_M_perp_vec(op_fft::FFTOperator, vec)
    tmp = M_perp_beta(op_fft, vec)
    tmp = M_perpt_z(op_fft, tmp)
    return tmp
end

function M_perpt_z(buffer_real, buffer_complex1, buffer_complex2, op, DFTdim, DFTsize, z, fft_timer, mapping_timer; rdft::Bool=false)
    N = prod(DFTsize)

    t1 = time_ns()
    if rdft
        temp = mul!(buffer_complex1, op, z)  # op_rfft
    else
        buffer_complex2 .= z  # z should be store in a complex buffer for mul!
        temp = mul!(buffer_complex1, op, buffer_complex2)  # op_fft
    end
    temp ./= sqrt(N)
    t2 = time_ns()
    fft_timer[] = fft_timer[] + (t2 - t1) / 1e9

    t3 = time_ns()
    beta = vec(buffer_real)
    DFT_to_beta!(beta, DFTdim, DFTsize, temp; rdft)
    t4 = time_ns()
    mapping_timer[] = mapping_timer[] + (t4 - t3) / 1e9
    return beta
end

function M_perp_beta(buffer_real, buffer_complex1, buffer_complex2, op, DFTdim, DFTsize, beta, index_missing, fft_timer, mapping_timer; rdft::Bool=false)
    N = prod(DFTsize)

    t3 = time_ns()
    v = buffer_complex2
    beta_to_DFT!(v, DFTdim, DFTsize, beta; rdft)
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

    buffer_real[index_missing] .= 0
    return buffer_real
end

function M_perpt_M_perp_vec(buffer_real, buffer_complex1, buffer_complex2, op, DFTdim, DFTsize, vec, index_missing, fft_timer, mapping_timer; rdft::Bool=false)
    tmp = M_perp_beta(buffer_real, buffer_complex1, buffer_complex2, op, DFTdim, DFTsize, vec, index_missing, fft_timer, mapping_timer; rdft)
    tmp = M_perpt_z(buffer_real, buffer_complex1, buffer_complex2, op, DFTdim, DFTsize, tmp, fft_timer, mapping_timer; rdft)
    return tmp
end

# mapping between DFT and real vector beta

# mapping DFT to beta
# @param DFTdim The DFTdimension of the problem (DFTdim = 1, 2, 3)
# @param size The size of each DFTdimension of the problem
#(we only consider the cases when the sizes are even for all the DFTdimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))
# @param v DFT

# @details This fucnction maps DFT to beta

# @return A 1-DFTdimensional real vector beta whose length is the product of size
# @example
# >DFTdim = 2;
# >size1 = (6, 8)
# >x = randn(6, 8)
# >v = fft(x)/sqrt(prod(size1))
# >beta = DFT_to_beta(DFTdim, size1, v)

function DFT_to_beta!(beta, DFTdim::Int, size, v; rdft::Bool=false)
    if (DFTdim == 1)
        DFT_to_beta_1d!(beta, v, size; rdft)
    elseif (DFTdim == 2)
        DFT_to_beta_2d!(beta, v, size; rdft)
    else
        DFT_to_beta_3d!(beta, v, size; rdft)
    end
    return beta
end

function DFT_to_beta(DFTdim::Int, size, v::Array{ComplexF64}; rdft::Bool=false)
    N = prod(size)
    beta = Vector{Float64}(undef, N)
    DFT_to_beta!(beta, DFTdim, size, v; rdft)
    return beta
end

function DFT_to_beta(DFTdim::Int, size, v::CuArray{ComplexF64}; rdft::Bool=false)
    N = prod(size)
    beta = CuVector{Float64}(undef, N)
    DFT_to_beta!(beta, DFTdim, size, v; rdft)
    return beta
end

function DFT_to_beta(DFTdim::Int, size, v::ROCArray{ComplexF64}; rdft::Bool=false)
    N = prod(size)
    beta = ROCVector{Float64}(undef, N)
    DFT_to_beta!(beta, DFTdim, size, v; rdft)
    return beta
end

# mapping beta to DFT
# @param DFTdim The DFTdimension of the problem (DFTdim = 1, 2, 3)
# @param size The size of each DFTdimension of the problem
#(we only consider the cases when the sizes are even for all the DFTdimenstions)
#(size is a tuple, e.g. size = (10, 20, 30))
# @param beta A 1-DFTdimensional real vector with length equal to the product of size

# @details This fucnction maps beta to DFT

# @return DFT DFT shares the same size as param sizes

# @example
# >DFTdim = 2;
# >size1 = (6, 8)
# >x = randn(6, 8)
# >v = fft(x)/sqrt(prod(size1))
# >beta = DFT_to_beta(DFTdim, size1, v)
# >w = beta_to_DFT(DFTdim, size1, beta) (w should be equal to v)

function beta_to_DFT!(v, DFTdim::Int, size, beta; rdft::Bool=false)
    if (DFTdim == 1)
        v = beta_to_DFT_1d!(v, beta, size; rdft)
    elseif (DFTdim == 2)
        v = beta_to_DFT_2d!(v, beta, size; rdft)
    else
        v = beta_to_DFT_3d!(v, beta, size; rdft)
    end
    return v
end

function beta_to_DFT(DFTdim::Int, size, beta::StridedVector{Float64}; rdft::Bool=false)
    v = Array{ComplexF64}(undef, size)
    beta_to_DFT!(v, DFTdim, size, beta; rdft)
    return v
end

function beta_to_DFT(DFTdim::Int, size, beta::StridedCuVector{Float64}; rdft::Bool=false)
    v = CuArray{ComplexF64}(undef, size)
    beta_to_DFT!(v, DFTdim, size, beta; rdft)
    return v
end

function beta_to_DFT(DFTdim::Int, size, beta::AMDGPU.StridedROCVector{Float64}; rdft::Bool=false)
    v = ROCArray{ComplexF64}(undef, size)
    beta_to_DFT!(v, DFTdim, size, beta; rdft)
    return v
end
