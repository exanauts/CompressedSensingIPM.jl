function punching(DFTdim, DFTsize, centers, radius, data)
    if DFTdim == 1
        index_missing = punching1D(DFTsize, centers, radius);
        data[index_missing].=0;
    elseif DFTdim == 2
        index_missing = punching2D(DFTsize, centers, radius);
        data[index_missing].=0;
    else
        index_missing = punching3D(DFTsize, centers, radius);
        data[index_missing].=0;
    end
    #radius_x, radius_y = (typeof(radius) <: Tuple) ? radius :
                                                #(radius, radius)
    return index_missing, data
end

function punching1D_onepoint(DFTsize, center, radius)
    inds = filter(i -> (norm(i-center)  <= radius), collect(1:DFTsize[1]));
    return inds
end

function punching1D(DFTsize, centers, radius)
    index_missing = [];
    for center in centers
        absolute_indices1 = punching1D_onepoint(DFTsize, center, radius);
        index_missing = [index_missing; absolute_indices1];
    end
    return Int.(index_missing)
end



function punching2D_onepoint(DFTsize, center, radius)
    inds = filter(i -> (((center[1]-i[1])^2+(center[2]-i[2])^2)  <= radius^2), CartesianIndices((1:DFTsize[1], 1:DFTsize[2])));
    return inds
end

function punching2D(DFTsize, centers, radius)
    index_missing = CartesianIndex{2}[];
    for center in centers
        absolute_indices1 = punching2D_onepoint(DFTsize, center, radius);
        append!(index_missing, absolute_indices1);
    end
    return index_missing
end


function punching3D_onepoint(DFTsize, center, radius)
    inds = filter(i -> (((center[1]-i[1])^2+(center[2]-i[2])^2+(center[3]-i[3])^2)  <= radius^2), CartesianIndices((1:DFTsize[1], 1:DFTsize[2], 1:DFTsize[3])));
    return inds
end

function punching3D(DFTsize, centers, radius)
    index_missing = CartesianIndex{3}[];
    for center in centers
        absolute_indices1 = punching3D_onepoint(DFTsize, center, radius);
        append!(index_missing, absolute_indices1);
    end
    return index_missing
end


function center_1d(DFTsize, missing_prob)
    N = prod(DFTsize);
    n = N*missing_prob/3;
    stepsize = ceil(N/n);
    centers = collect(1:stepsize:N);
    return centers
end

function center_2d(DFTsize, missing_prob)
    N = prod(DFTsize);
    Nt = DFTsize[1];
    Ns = DFTsize[2];
    n = round((N*missing_prob/5)^(1/2));
    stepsize1 = Int.(ceil(Nt/n));
    stepsize2 = Int.(ceil(Ns/n));
    centers = CartesianIndices((1:stepsize1:Nt, 1:stepsize2:Ns));
    return centers
end

function center_3d(DFTsize, missing_prob)
    N = prod(DFTsize);
    N1 = DFTsize[1];
    N2 = DFTsize[2];
    N3 = DFTsize[3];
    n = round((N*missing_prob/7)^(1/3));
    stepsize1 = Int.(ceil(N1/n));
    stepsize2 = Int.(ceil(N2/n));
    stepsize3 = Int.(ceil(N3/n));
    centers = CartesianIndices((1:stepsize1:N1, 1:stepsize2:N2, 1:stepsize3:N3));
    return centers
end

function centering(DFTdim, DFTsize, missing_prob)
    if DFTdim == 1
        return center_1d(DFTsize, missing_prob)
    elseif DFTdim == 2
        return center_2d(DFTsize, missing_prob)
    else
        return center_3d(DFTsize, missing_prob)
    end
end
