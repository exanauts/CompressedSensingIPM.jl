using Krylov, LinearOperators
function CompressedSensingADMM(fft_parameter, fft_operator;rho=1, maxt = 1000, tol = 1e-6)
    DFTdim = fft_parameter.DFTdim
    DFTsize = fft_parameter.DFTsize
    z0 = fft_parameter.z0
    lambda = fft_parameter.lambda
    index_missing = fft_parameter.index_missing
    n = prod(DFTsize)
    x0 = ones(n)
    ztemp = ones(n)
    y0 = zeros(n);
    t = 0;
    err = 1;
    workspace = CgWorkspace(Vector{Float64}, n, n)
    op_fft = LinearOperator(Float64, n, n, true, true, (y_op, v_ip)->MperptMRho(y_op, v_ip, fft_operator, rho))
    
    # subgrad_vec = subgrad(paramf, z0);
    time_vec =[0];
    cg_vec=[];

    while((t<maxt) & (err>tol))
        b = z0.+(rho.*ztemp).-y0;
        # update x
        # x1,cgiter = cg(rho, b, index_missing, DFTdim, DFTsize);
        cg!(workspace, op_fft, b)
        # @show cgiter
        x1 = workspace.x
        # update z
        z1 = softthreshold.(x1 + y0/rho, lambda/rho);
        # update y
        y1 = y0 + rho*(x1 - z1);
        # check the convergence
        err = max(norm(x1 - x0, 2), norm(y1 - y0, 2), norm(z1 - z0, 2));
        x0 = x1;
        z0 = z1;
        y0 = y1;
        t = t + 1;

        # subgrad_vec = [subgrad_vec; subgrad(paramf, z0)];
        # time_vec = [time_vec; time()-Time];
        # cg_vec =[cg_vec; cgiter]
        
    end
end





function MperptMRho(y_op, v_ip, fft_operator,rho)
    # return (M_perp^t*M_perp+rho*I)*x
    y_op = M_perpt_M_perp_vec(fft_operator, v_ip)
    y_op = y_op .+ rho.*v_ip;
    return y_op
end
