function punching(DFTdim, DFTsize, centers, radius, data)
    index_missing = Int[]
    if DFTdim == 1
        punching1D(DFTsize, centers, radius, index_missing)
    elseif DFTdim == 2
        punching2D(DFTsize, centers, radius, index_missing)
    else
        punching3D(DFTsize, centers, radius, index_missing)
    end
    data |> typeof |> println
    vec(data)[index_missing] .= 0
    return index_missing, data
end

function punching1D(DFTsize, centers, radius, index_missing)
    Nx = DFTsize[1]
    for center in centers
        for x in 1:Nx
            if abs(x - center) ≤ radius
                push!(index_missing, x)
            end
        end
    end
    return index_missing
end

function punching2D(DFTsize, centers, radius, index_missing)
    Nx = DFTsize[1]
    Ny = DFTsize[2]
    for center in centers
        for x in 1:Nx
            for y in 1:Ny
                if (center[1] - x[1])^2 + (center[2] - x[2])^2 ≤ radius^2
                    push!(index_missing, x + Nx*y)
                end
            end
        end
    end
    return index_missing
end

function punching3D(DFTsize, centers, radius, index_missing)
    Nx = DFTsize[1]
    Ny = DFTsize[2]
    Nz = DFTsize[3]
    for center in centers
        for x in 1:Nx
            for y in 1:Ny
                for z in 1:Nz
                    if (center[1] - x[1])^2 + (center[2] - x[2])^2 + (center[3] - x[3])^2 ≤ radius^2
                        push!(index_missing, x + Nx*y + Nx*Ny*z)
                    end
                end
            end
        end
    end
    return index_missing
end

function center_1d(DFTsize, missing_prob)
    N = prod(DFTsize)
    Nx = DFTsize[1]
    n = N * missing_prob / 3
    stepsize = ceil(Nx / n)
    centers = 1:stepsize:N
    return centers
end

function center_2d(DFTsize, missing_prob)
    N = prod(DFTsize)
    Nx = DFTsize[1]
    Ny = DFTsize[2]
    n = round((N * missing_prob / 5) |> sqrt)
    stepsize1 = ceil(Nx / n) |> Int
    stepsize2 = ceil(Ny / n) |> Int
    centers = (1:stepsize1:Nx, 1:stepsize2:Ny) |> CartesianIndices
    return centers
end

function center_3d(DFTsize, missing_prob)
    N = prod(DFTsize)
    Nx = DFTsize[1]
    Ny = DFTsize[2]
    Nz = DFTsize[3]
    n = round((N * missing_prob / 7) |> cbrt)
    stepsize1 = ceil(N1/n) |> Int
    stepsize2 = ceil(N2/n) |> Int
    stepsize3 = ceil(N3/n) |> Int
    centers = (1:stepsize1:Nx, 1:stepsize2:Ny, 1:stepsize3:Nz) |> CartesianIndices
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
