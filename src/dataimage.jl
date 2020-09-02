function euclideanDistances(dataMatrix::Array{Float64,2})::Array{Float64,2}
    n, p = size(dataMatrix)
    d = zeros(Float64, n, n)
    for i in 1:n
        for j in i:n
            if i != j 
                d[i, j] = sqrt(sum((dataMatrix[i,:] .- dataMatrix[j,:]).^2.0))
                d[j, i] = d[i, j]
            end
        end
    end
    return d
end

function mahalanobisBetweenPairs(dataMatrix::Array{Float64,2})::Array{Float64,2}
    n, p = size(dataMatrix)
    d = zeros(Float64, n, n)
    covmat = cov(dataMatrix)
    if det(covmat) == 0.0
        @warn "Covariance matrix is singular, mahalanobis distances can not be calculated."
    end
    covinv = inv(covmat)
    for i in 1:n
        for j in i:n
            if i != j 
                d[i, j] = sqrt((dataMatrix[i,:] .- dataMatrix[j,:]) * covinv * (dataMatrix[i,:] .- dataMatrix[j,:])')
                d[j, i] = d[i, j]
            end
        end
    end
    return d
end

function dataimage(dataMatrix::Array{Float64,2}; distance=:mahalanobis)
    d = nothing
    if distance == :mahalanobis
        d = mahalanobisSquaredBetweenPairs(dataMatrix)
    elseif distance == :euclidean
        d = euclideanDistances(dataMatrix)
    else
        @error "Distance function unknown: " distance 
        @error "Using mahalanobis instead"
        d = mahalanobisSquaredBetweenPairs(dataMatrix)
    end
    colours = 1.0 .- d / maximum(d)
    n, _ = size(d)
    colormatrix = Array{RGB{Float64}}(undef, n, n)
    for i in 1:n
        for j in 1:n
            colormatrix[i, j] = RGB(colours[i, j])
        end
    end
    plot(colormatrix)
end
