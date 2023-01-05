module Satman2013


export satman2013


import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..LTS: iterateCSteps
import ..OrdinaryLeastSquares: ols, coef, wls, residuals
import ..Diagnostics: mahalanobisSquaredMatrix
import Distributions: median
import LinearAlgebra: diag


"""

    satman2013(setting)

Perform Satman (2013) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description
The algorithm constructs a fast and robust covariance matrix to calculate robust mahalanobis
distances. These distances are then used to construct weights for later use in a weighted least 
squares estimation. In the last stage, C-steps are iterated on the basic subset found in previous
stages. 

# Output
- `["outliers"]`: Array of indices of outliers.
- `["betas"]`: Array of estimated regression coefficients.
- `["residuals"]`: Array of residuals.

# Examples
```julia-repl
julia> eg0001 = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk);
julia> satman2013(reg0001)
Dict{Any,Any} with 1 entry:
  "outliers" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 47]
  "betas" => ...
  "residuals" => ...
```

# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.
"""
function satman2013(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return satman2013(X, y)
end


function satman2013(X::Array{Float64,2}, y::Array{Float64,1})
    # Sample size and the number of regression parameters
    n, p = size(X)

    # A lower limit for the number of clean observations 
    h = Int(floor((n + p + 1.0) / 2.0))

    # If the intercept is included, remove it from the data
    X0 = X
    p0 = p
    if X0[:, 1] == ones(n)
        X0 = X[:, 2:end]
        p0 = p - 1
    end

    allindices = collect(1:n)

    # Initial covariance matrix 
    covmat = zeros(p0, p0)

    # Construct an estimation of the covariance matrix
    for i = 1:p0
        for j = 1:p0
            if i == j
                @inbounds covmat[i, j] = median(abs.(X0[:, i] .- median(X0[:, i])))
            else
                @inbounds covmat[i, j] =
                    median((X0[:, i] .- median(X0[:, i])) .* (X0[:, j] .- median(X0[:, j])))
            end
        end
    end

    medians = applyColumns(median, X0)
    mhs = mahalanobisSquaredMatrix(X0, meanvector = medians, covmatrix = covmat)
    if mhs isa Nothing
        md2 = zeros(Float64, n)
    else
        md2 = diag(mhs)
    end
    md = sqrt.(md2)

    # Perform Weighted Least Squares using the weights based on Mahalanobis distances
    wlsreg = wls(X, y, 1.0 ./ md)
    wlsresiduals = residuals(wlsreg)

    # Find best h indices using the residuals obtained from WLS
    sorted_indices = sortperm(abs.(wlsresiduals))
    best_h_indices = sorted_indices[1:h]

    # Iterate C-steps
    _, bestset = iterateCSteps(X, y, best_h_indices, h)

    # Estimate the final regression parameters
    olsreg = ols(X[bestset, :], y[bestset])
    betas = coef(olsreg)
    resids = y .- (X * betas)
    med_res = median(resids)
    standardized_resids = (resids .- med_res) / median(abs.(resids .- med_res))

    outlierset = filter(i -> abs(standardized_resids[i]) > 2.5, allindices)

    result = Dict()
    result["outliers"] = outlierset
    result["betas"] = betas 
    result["residuals"] = resids

    return result
end


end # end of module Satman2013
