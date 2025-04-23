## punching
function punching(DFTdim, DFTsize, centers, radius, data)
    if radius == 1
        return punching_optimized(DFTdim, DFTsize, centers, data)
    end
    if DFTdim == 1
        index_missing = punching1D(DFTsize, centers, radius)
        data[index_missing] .= 0
    elseif DFTdim == 2
        index_missing = punching2D(DFTsize, centers, radius)
        data[index_missing] .= 0
    else
        index_missing = punching3D(DFTsize, centers, radius)
        data[index_missing] .= 0
    end
    return index_missing, data
end

function punching1D(DFTsize, centers, radius)
    N = prod(DFTsize)
    index_missing = Vector{CartesianIndex{1}}(undef, N)
    pos = 0
    for i = 1:DFTsize[1]
        for center in centers
            if abs(i - center[1]) <= radius
                pos = pos + 1
                index_missing[pos] = CartesianIndex{1}(i)
            end
        end
    end
    resize!(index_missing, pos)
    return index_missing
end

function punching2D(DFTsize, centers, radius)
    N = prod(DFTsize)
    index_missing = Vector{CartesianIndex{2}}(undef, N)
    pos = 0
    for i = 1:DFTsize[1]
        for j = 1:DFTsize[2]
            for center in centers
                if (center[1] - i)^2 + (center[2] - j)^2 <= radius^2
                    pos = pos + 1
                    index_missing[pos] = CartesianIndex{2}(i, j)
                end
            end
        end
    end
    resize!(index_missing, pos)
    return index_missing
end

function punching3D(DFTsize, centers, radius)
    N = prod(DFTsize)
    index_missing = Vector{CartesianIndex{3}}(undef, N)
    pos = 0
    for i = 1:DFTsize[1]
        for j = 1:DFTsize[2]
            for k = 1:DFTsize[3]
                for center in centers
                    if (center[1] - i)^2 + (center[2] - j)^2 + (center[3] - k)^2 <= radius^2
                        pos = pos + 1
                        index_missing[pos] = CartesianIndex{3}(i, j, k)
                    end
                end
            end
        end
    end
    resize!(index_missing, pos)
    return index_missing
end

## centering
function centering(DFTdim, DFTsize, missing_prob)
    if DFTdim == 1
        return center_1d(DFTsize, missing_prob)
    elseif DFTdim == 2
        return center_2d(DFTsize, missing_prob)
    else
        return center_3d(DFTsize, missing_prob)
    end
end

function center_1d(DFTsize, missing_prob)
    N = prod(DFTsize)
    n = N*missing_prob/3
    stepsize = ceil(N/n) |> Int
    centers = CartesianIndices((1:stepsize:N))
    return centers
end

function center_2d(DFTsize, missing_prob)
    N = prod(DFTsize)
    Nt = DFTsize[1]
    Ns = DFTsize[2]
    n = (N*missing_prob/5)^(1/2)
    stepsize1 = ceil(Nt/n) |> Int
    stepsize2 = ceil(Ns/n) |> Int
    centers = CartesianIndices((1:stepsize1:Nt, 1:stepsize2:Ns))
    return centers
end

function center_3d(DFTsize, missing_prob)
    N = prod(DFTsize)
    N1 = DFTsize[1]
    N2 = DFTsize[2]
    N3 = DFTsize[3]
    n = (N*missing_prob/7)^(1/3)
    stepsize1 = ceil(N1/n) |> Int
    stepsize2 = ceil(N2/n) |> Int
    stepsize3 = ceil(N3/n) |> Int
    centers = CartesianIndices((1:stepsize1:N1, 1:stepsize2:N2, 1:stepsize3:N3))
    return centers
end

## punching_optimized
function punching_optimized(DFTdim, DFTsize, centers, data)
    if DFTdim == 1
        index_missing = punching_optimized_1D(DFTsize, centers)
        data[index_missing] .= 0
    elseif DFTdim == 2
        index_missing = punching_optimized_2D(DFTsize, centers)
        data[index_missing] .= 0
    else
        index_missing = punching_optimized_3D(DFTsize, centers)
        data[index_missing] .= 0
    end
    return index_missing, data
end

function punching_optimized_1D(DFTsize, centers)
    ncenters = prod(centers |> size)
    index_missing = Vector{CartesianIndex{1}}(undef, 3*ncenters)
    Nx = DFTsize[1]
    pos = 0
    for center in centers
        for i in center[1]
            for i2 = i-1:i+1
                if 1 <= i2 <= Nx
                    pos = pos + 1
                    index_missing[pos] = CartesianIndex{1}(i2)
                end
            end
        end
    end
    resize!(index_missing, pos)
    return index_missing
end

function punching_optimized_2D(DFTsize, centers)
    ncenters = prod(centers |> size)
    index_missing = Vector{CartesianIndex{2}}(undef, 5*ncenters)
    Nx = DFTsize[1]
    Ny = DFTsize[2]
    pos = 0
    for center in centers
        for i in center[1]
            for j in center[2]
                for i2 = i-1:i+1
                    for j2 = j-1:j+1
                        if (1 <= i2 <= Nx) && (1 <= j2 <= Ny) && (abs(i2 - i) + abs(j2 - j) <= 1)
                            pos = pos + 1
                            index_missing[pos] = CartesianIndex{2}(i2, j2)
                        end
                    end
                end
            end
        end
    end
    resize!(index_missing, pos)
    return index_missing
end

function punching_optimized_3D(DFTsize, centers)
    ncenters = prod(centers |> size)
    index_missing = Vector{CartesianIndex{3}}(undef, 7*ncenters)
    Nx = DFTsize[1]
    Ny = DFTsize[2]
    Nz = DFTsize[3]
    pos = 0
    for center in centers
        for i in center[1]
            for j in center[2]
                for k in center[3]
                    for i2 = i-1:i+1
                        for j2 = j-1:j+1
                            for k2 = k-1:k+1
                                if (1 <= i2 <= Nx) && (1 <= j2 <= Ny) && (1 <= k2 <= Nz) && (abs(i2 - i) + abs(j2 - j) + abs(k2 - k) <= 1)
                                    pos = pos + 1
                                    index_missing[pos] = CartesianIndex{3}(i2, j2, k2)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    resize!(index_missing, pos)
    return index_missing
end
