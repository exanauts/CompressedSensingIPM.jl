function MperptMRho(y_op, v_ip, fft_operator, rho)
    # return (M_perp^t * M_perp + rho * I) * x
    y_op .= M_perpt_M_perp_vec(fft_operator, v_ip)
    y_op .= y_op .+ rho .* v_ip
    return y_op
end

function CompressedSensingADMM(fft_parameter, fft_operator; rho=1, maxt = 1000, tol = 1e-6)
    DFTdim = fft_parameter.DFTdim
    DFTsize = fft_parameter.DFTsize
    VT = typeof(vec(fft_parameter.z0))

    tmp = M_perpt_z(fft_operator, fft_parameter.z0)
    M_perpt_z0 = copy(tmp)
    lambda = fft_parameter.lambda
    index_missing = fft_parameter.index_missing
    n = prod(DFTsize)
    x0 = VT(undef, n)
    fill!(x0, 1.0)
    ztemp = VT(undef, n)
    fill!(ztemp, 1.0)
    y0 = VT(undef, n)
    fill!(y0, 0.0)
    y1 = VT(undef, n)
    z1 = VT(undef, n)
    t = 0
    err = 1
    buffer = VT(undef, n)
    workspace = CgWorkspace(n, n, VT)
    op_fft = LinearOperator(Float64, n, n, true, true, (y_op, v_ip) -> MperptMRho(y_op, v_ip, fft_operator, rho))

    while (t < maxt) && (err > tol)
        b = M_perpt_z0 .+ (rho .* ztemp) .- y0

        # update x
        cg!(workspace, op_fft, b)
        x1 = workspace.x

        # update z
        buffer .= x1 .+ y0 ./ rho
        z1 = softthreshold!(z1, buffer, lambda/rho)

        # update y
        y1 .= y0 .+ rho .* (x1 .- z1)

        # check the convergence
        buffer .= x1 .- x0
        err1 = norm(buffer, 2)
        buffer .= y1 .- y0
        err2 = norm(buffer, 2)
        buffer .= z1.- ztemp
        err3 = norm(buffer,2)
        err = max(err1, err2, err3)

        x0 .= x1
        ztemp .= z1
        y0 .= y1
        t = t + 1
    end
    return ztemp
end

function softthreshold_scalar(x, thre)
    if x > thre
        return x - thre
    elseif x < -thre
        return x + thre
    else
        return 0.0
    end
end

function softthreshold!(z1, x, thre)
    map!(xi -> softthreshold_scalar(xi, thre), z1, x)
    return z1
end
