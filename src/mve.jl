function enlargesubset(initialsubset, data::DataFrame, dataMatrix::Matrix, h::Int)
    n, p = size(dataMatrix)
    indices = collect(1:n)
    basicsubset = copy(initialsubset)
    while length(basicsubset) < h
        meanvector = applyColumns(mean, data[basicsubset,:])
        covmatrix = cov(dataMatrix[basicsubset, :])
        md2mat = mahalanobisSquaredMatrix(data, meanvector=meanvector, covmatrix=covmatrix)
        md2 = diag(md2mat)
        md2sortedindex = sortperm(md2)
        basicsubset = md2sortedindex[1:(length(basicsubset) + 1)]
    end
    return basicsubset
end


function robcov(data::DataFrame; alpha=0.01, estimator=:mve)
    dataMatrix = convert(Matrix, data)
    n, p = size(dataMatrix)
    chisquared = Chisq(p)
    chisqcrit = quantile(chisquared, 1.0 - alpha)
    c = sqrt(chisqcrit)
    h = Int(floor((n + p + 1.0) / 2.0)) 
    indices = collect(1:n)
    k = p + 1
    mingoal = Inf
    bestinitialsubset = []
    besthsubset = []
    maxiter = minimum([p * 500, 3000])
    initialsubset = []
    hsubset = []
    for iter in 1:maxiter
        goal = Inf
        try
            initialsubset = sample(indices, k, replace=false)
            hsubset = enlargesubset(initialsubset, data, dataMatrix, h) 
            covmatrix = cov(dataMatrix[hsubset, :])
            if estimator == :mve
                meanvector = applyColumns(mean, data[hsubset, :])
                md2mat = mahalanobisSquaredMatrix(data, meanvector=meanvector, covmatrix=covmatrix)
                DJ = sqrt(sort(diag(md2mat))[h])
                goal =  (DJ / c)^p * det(covmatrix)
            else
                goal = det(covmatrix)
            end
        catch e
            # Possibly singularity
        end
        if goal < mingoal
            mingoal = goal
            bestinitialsubset = initialsubset
            besthsubset = hsubset
        end
    end
    meanvector = applyColumns(mean, data[besthsubset, :])
    covariancematrix = cov(dataMatrix[besthsubset, :])
    md2 = diag(mahalanobisSquaredMatrix(data, meanvector=meanvector, covmatrix=covariancematrix))
    outlierset = filter(x -> md2[x] > chisqcrit, 1:n)
    result = Dict()
    result["goal"] = mingoal
    result["best.subset"] = sort(besthsubset)
    result["robust.location"] = meanvector
    result["robust.covariance"] = covariancematrix
    result["squared.mahalanobis"] = md2
    result["chisq.crit"] = chisqcrit 
    result["alpha"] = alpha
    result["outliers"] = outlierset
    return result 
end


"""
    mve(data; alpha = 0.01)

Performs the Minimum Volume Ellipsoid algorithm for a robust covariance matrix.

# Arguments
- `data::DataFrame`: Multivariate data.
- `alpha::Float64`: Probability for quantiles of Chi-Squared statistic.

# Description 
`mve` searches for a robust location vector and a robust scale matrix, e.g covariance matrix.
The method also reports a usable diagnostic measure, Mahalanobis distances, which are calculated using 
the robust counterparts instead of mean vector and usual covariance matrix. Mahalanobis distances 
are directly comparible with quantiles of a ChiSquare Distribution with `p` degrees of freedom.
See also: [`mcd`](@ref)  

# Output
- `["goal"]`: Objective value
- `["best.subset"]`: Indices of best h-subset of observations
- `["robust.location"]`: Vector of robust location measures
- `["robust.covariance"]`: Robust covariance matrix
- `["squared.mahalanobis"]`: Array of Mahalanobis distances calculated using robust location and scale measures.
- `["chisq.crit"]`: Chisquare quantile used in threshold
- `["alpha"]`: Probability used in calculating the Chisquare quantile, e.g `chisq.crit`
- `["outliers"]`: Array of indices of outliers.


# References
Van Aelst, Stefan, and Peter Rousseeuw. "Minimum volume ellipsoid." Wiley 
Interdisciplinary Reviews: Computational Statistics 1.1 (2009): 71-82.
"""
function mve(data::DataFrame; alpha=0.01)
    robcov(data, alpha=alpha, estimator=:mve) 
end

function mve(data::Array{Float64,2}; alpha=0.01)
    return mve(DataFrame(data), alpha=alpha)
end


"""
    mcd(data; alpha = 0.01)

Performs the Minimum Covariance Determinant algorithm for a robust covariance matrix.

# Arguments
- `data::DataFrame`: Multivariate data.
- `alpha::Float64`: Probability for quantiles of Chi-Squared statistic.

# Description 
`mcd` searches for a robust location vector and a robust scale matrix, e.g covariance matrix.
The method also reports a usable diagnostic measure, Mahalanobis distances, which are calculated using 
the robust counterparts instead of mean vector and usual covariance matrix. Mahalanobis distances 
are directly comparible with quantiles of a ChiSquare Distribution with `p` degrees of freedom.  
See also: [`mve`](@ref)

# Output
- `["goal"]`: Objective value
- `["best.subset"]`: Indices of best h-subset of observations
- `["robust.location"]`: Vector of robust location measures
- `["robust.covariance"]`: Robust covariance matrix
- `["squared.mahalanobis"]`: Array of Mahalanobis distances calculated using robust location and scale measures.
- `["chisq.crit"]`: Chisquare quantile used in threshold
- `["alpha"]`: Probability used in calculating the Chisquare quantile, e.g `chisq.crit`
- `["outliers"]`: Array of indices of outliers.


# Notes
Algorithm is implemented using concentration steps as described in the reference paper.
However, details about number of iterations are slightly different.

# References
Rousseeuw, Peter J., and Katrien Van Driessen. "A fast algorithm for the minimum covariance 
determinant estimator." Technometrics 41.3 (1999): 212-223.
"""
function mcd(data::DataFrame; alpha=0.01)
    robcov(data, alpha=alpha, estimator=:mcd)
end

function mcd(data::Array{Float64,2}; alpha=0.01)
    return mcd(DataFrame(data), alpha=alpha)
end
