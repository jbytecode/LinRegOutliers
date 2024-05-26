module Hadi94


export hadi1994


import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..Diagnostics: mahalanobisSquaredMatrix, coordinatwisemedians

import Distributions: mean, cov, quantile
import LinearAlgebra: det, diag
import Distributions: Chisq



"""
    hadi1994(multivariateData)

Perform Hadi (1994) algorithm for a given multivariate data.

# Arguments
- `multivariateData::AbstractMatrix{Float64}`: Multivariate data.

# Description
Algorithm starts with an initial subset and enlarges the subset to 
obtain robust covariance matrix and location estimates. This algorithm 
is an extension of `hadi1992`.

# Output
- `["outliers"]`: Array of indices of outliers
- `["critical.chi.squared"]`: Threshold value for determining being an outlier
- `["rth.robust.distance"]`: rth robust distance, where (r+1)th robust distance is the first one that exceeds the threshold.


# Examples
```julia-repl
julia> multidata = hcat(hbk.x1, hbk.x2, hbk.x3);

julia> hadi1994(multidata)
Dict{Any,Any} with 3 entries:
  "outliers"              => [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  "critical.chi.squared" => 7.81473
  "rth.robust.distance"   => 5.04541
```

# Reference
Hadi, Ali S. "A modification of a method for the dedection of outliers in multivariate samples"
Journal of the Royal Statistical Society: Series B (Methodological) 56.2 (1994): 393-396.
"""
function hadi1994(multivariateData::AbstractMatrix{Float64}; alpha = 0.05)
    n, p = size(multivariateData)
    h = Int(round((n + p + 1.0) / 2.0))
    cnp = (1 + ((2) / (n - 1 - 3p)) + ((p + 1) / (n - p)))^2
    chi_50_quantile = quantile(Chisq(p), 0.50)
    critical_quantile = quantile(Chisq(p), 1 - alpha)
    allindices = collect(1:n)

    meds = coordinatwisemedians(multivariateData)
    Sm = (1.0 / (n - 1.0)) * (multivariateData .- meds')' * (multivariateData .- meds')
    
    msm1 = mahalanobisSquaredMatrix(multivariateData, meanvector = meds, covmatrix = Sm)

    if isnothing(msm1)
            throw(ErrorException("Mahalanobis distances are not calculated"))
        end
    
    mah0 = diag(msm1)

    ordering_indices_mah0 = sortperm(mah0)
    best_indices_mah0 = ordering_indices_mah0[1:h]
    starting_data = multivariateData[best_indices_mah0, :]

    Cv = coordinatwisemedians(starting_data)
    Sv = (1.0 / (h - 1.0)) * (starting_data .- Cv')' * (starting_data .- Cv')
    msm2 = mahalanobisSquaredMatrix(multivariateData, meanvector = Cv, covmatrix = Sv)

    if isnothing(msm2)
        throw(ErrorException("Mahalanobis distances are not calculated"))
    end
    
    mah1 = diag(msm2)
    ordering_indices_mah1 = sortperm(mah1)

    r = p + 1

    basic_subset_indices = Int[]
    
    basic_subset = []

    sorted_mah1 = []
    
    Cb = zeros(Float64, p)
    Sb = zeros(Float64, p, p)
    msm3 = zeros(Float64, n, n)
    sorted_mah1 = zeros(Float64, n)

    cfactor = 0
    isFullRank = false

    while r < n

        isFullRank = false

        while !isFullRank
            basic_subset_indices = ordering_indices_mah1[1:r]
            basic_subset = multivariateData[basic_subset_indices, :]
            Cb .= applyColumns(mean, basic_subset)
            Sb .= cov(basic_subset)
            cfactor = cnp * sqrt(sort(mah1)[h]) / chi_50_quantile
            r += 1

            if (det(cfactor * Sb) == 0 && r < n)
                isFullRank = false
            else
                isFullRank = true
            end
        end

        msm3 .= mahalanobisSquaredMatrix(
            multivariateData,
            meanvector = Cb,
            covmatrix = (cfactor * Sb),
        )

        if isnothing(msm3)
            throw(ErrorException("Mahalanobis distances are not calculated"))
        end

        mah1 = diag(msm3)

        ordering_indices_mah1 = sortperm(mah1)
        basic_subset_indices = ordering_indices_mah1[1:r]

        sorted_mah1 .= sort(mah1)

        if sorted_mah1[r] >= critical_quantile
            break
        end
    end

    outlierset = setdiff(allindices, basic_subset_indices)

    result = Dict{String, Any}()
    result["outliers"] = sort(outlierset)
    result["critical.chi.squared"] = critical_quantile
    result["rth.robust.distance"] = sorted_mah1[r-1]
    return result
end


end # end of module Hadi94 
