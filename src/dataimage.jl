"""

    euclideanDistances(dataMatrix)

Calculate Euclidean distances between pairs. 

# Arguments
- `dataMatrix::Array{Float64, 1}`: Data matrix with dimensions n x p, where n is the number of observations and p is the number of variables.

# Notes
    This is the helper function for the dataimage() function defined in Marchette & Solka (2003).
    
# References
Marchette, David J., and Jeffrey L. Solka. "Using data images for outlier detection." 
Computational Statistics & Data Analysis 43.4 (2003): 541-552.
"""
function euclideanDistances(dataMatrix::Array{Float64,2})::Array{Float64,2}
    n, p = size(dataMatrix)
    d = zeros(Float64, n, n)
    for i in 1:n
        for j in i:n
            if i != j 
                @inbounds d[i, j] = sqrt(sum((dataMatrix[i,:] .- dataMatrix[j,:]).^2.0))
                @inbounds d[j, i] = d[i, j]
            end
        end
    end
    return d
end


"""

    mahalanobisBetweenPairs(dataMatrix)

Calculate Mahalanobis distances between pairs. 

# Arguments
- `dataMatrix::Array{Float64, 1}`: Data matrix with dimensions n x p, where n is the number of observations and p is the number of variables.

# Notes
    Differently from Mahalabonis distances, this function calculates Mahalanobis distances between 
    pairs, rather than the distances to center of the data. This is the helper function for the 
    dataimage() function defined in Marchette & Solka (2003).
    
# References
Marchette, David J., and Jeffrey L. Solka. "Using data images for outlier detection." 
Computational Statistics & Data Analysis 43.4 (2003): 541-552.
"""
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
                @inbounds d[i, j] = sqrt((dataMatrix[i,:] .- dataMatrix[j,:]) * covinv * (dataMatrix[i,:] .- dataMatrix[j,:])')
                @inbounds d[j, i] = d[i, j]
            end
        end
    end
    return d
end



"""

    dataimage(dataMatrix; distance = :mahalanobis)

Generate the Marchette & Solka (2003) data image for a given data matrix. 

# Arguments
- `dataMatrix::Array{Float64, 1}`: Data matrix with dimensions n x p, where n is the number of observations and p is the number of variables.
- `distance::Symbol`: Optional argument for the distance function.

# Notes
    distance is :mahalanobis by default, for the Mahalanobis distances. 
    use 

        dataimage(mat, distance = :euclidean)
    
    to use Euclidean distances.
    
# Examples
```julia-repl
julia> x1 = hbk[:,"x1"];
julia> x2 = hbk[:,"x2"];
julia> x3 = hbk[:,"x3"];
julia> mat = hcat(x1, x2, x3);
julia> dataimage(mat)
```

# References
Marchette, David J., and Jeffrey L. Solka. "Using data images for outlier detection." 
Computational Statistics & Data Analysis 43.4 (2003): 541-552.
"""
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
            @inbounds colormatrix[i, j] = RGB(colours[i, j])
        end
    end
    plot(colormatrix)
end
