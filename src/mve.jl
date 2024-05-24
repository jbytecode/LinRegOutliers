module MVE

export mve


import DataFrames: DataFrame
import LinearAlgebra: diag, det
import Distributions: median, cov, mean, quantile, sample, Chisq

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..Diagnostics: mahalanobisSquaredMatrix


function enlargesubset(initialsubset, data::AbstractMatrix, h::Int)
    n, p = size(data)

    basicsubset = copy(initialsubset)
    
    meanvector = Array{Float64}(undef, p)
    covmatrix = Matrix{Float64}(undef, p, p)
    md2mat = Matrix{Float64}(undef, n, n)
    md2 = Array{Float64}(undef, n)
    md2sortedindex = Array{Int}(undef, n)

    while length(basicsubset) < h
        meanvector .= applyColumns(mean, data[basicsubset, :])
        covmatrix .= cov(data[basicsubset, :])
        md2mat .=
            mahalanobisSquaredMatrix(data, meanvector = meanvector, covmatrix = covmatrix)
        md2 .= diag(md2mat)
        md2sortedindex .= sortperm(md2)
        basicsubset = md2sortedindex[1:(length(basicsubset)+1)]
    end
    return basicsubset
end


function robcov(data::Matrix; alpha = 0.01, estimator = :mve)
    
    n, p = size(data)
    chisquared = Chisq(p)
    chisqcrit = quantile(chisquared, 1.0 - alpha)
    c = sqrt(chisqcrit)
    h = Int(floor((n + p + 1.0) / 2.0))
    indices = collect(1:n)
    k = p + 1
    mingoal = Inf

    maxiter = minimum([p * 500, 3000])
    
    initialsubset = Array{Int}(undef, k)
    bestinitialsubset = Array{Int}(undef, k)
    
    hsubset = Array{Int}(undef, h)
    besthsubset = Array{Int}(undef, h)

    covmatrix = Matrix{Float64}(undef, p, p)
    meanvector = Array{Float64}(undef, p)
    md2mat = Matrix{Float64}(undef, n, n)

    md2 = Array{Float64}(undef, n)

    for iter = 1:maxiter
        goal = Inf
        try
            initialsubset .= sample(indices, k, replace = false)
            hsubset .= enlargesubset(initialsubset, data, h)
            covmatrix .= cov(data[hsubset, :])
            if estimator == :mve
                meanvector .= applyColumns(mean, data[hsubset, :])
                md2mat .= mahalanobisSquaredMatrix(
                    data,
                    meanvector = meanvector,
                    covmatrix = covmatrix,
                )
                DJ = sqrt(sort(diag(md2mat))[h])
                goal = (DJ / c)^p * det(covmatrix)
            else
                goal = det(covmatrix)
            end
        catch e
            # Possibly singularity
        end
        if goal < mingoal
            mingoal = goal
            bestinitialsubset .= initialsubset
            besthsubset .= hsubset
        end
    end
   

    meanvector .= applyColumns(mean, data[besthsubset, :])
    covmatrix .= cov(data[besthsubset, :])
    md2 .= diag(
        mahalanobisSquaredMatrix(
            data,
            meanvector = meanvector,
            covmatrix = covmatrix,
        ),
    )
    outlierset = filter(x -> md2[x] > chisqcrit, 1:n)
    result = Dict{String, Any}()
    result["goal"] = mingoal
    result["best.subset"] = sort(besthsubset)
    result["robust.location"] = meanvector
    result["robust.covariance"] = covmatrix
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
function mve(data::DataFrame; alpha = 0.01)
    robcov(Matrix(data), alpha = alpha, estimator = :mve)
end

function mve(data::AbstractMatrix{Float64}; alpha = 0.01)
    robcov(data, alpha = alpha, estimator = :mve)
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
function mcd(data::DataFrame; alpha = 0.01)
    robcov(Matrix(data), alpha = alpha, estimator = :mcd)
end

function mcd(data::AbstractMatrix{Float64}; alpha = 0.01)
    robcov(data, alpha = alpha, estimator = :mcd)
end


end # end of module MVE 
