module ASM2000

export asm2000

import Distributions: quantile, mean, sample, cov
import LinearAlgebra: det
import Clustering: Hclust, hclust, cutree


import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, zstandardize
import ..Diagnostics: mahalanobisSquaredBetweenPairs
import ..LTS: lts
import ..SMR98: majona
import ..OrdinaryLeastSquares: ols, coef



"""

    asm2000(setting)

Perform the Setan, Halim and Mohd (2000) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description
The algorithm performs a Least Trimmed Squares (LTS) estimate and yields standardized 
residual - fitted response pairs. A single linkage clustering algorithm is performed on these
pairs. Like `smr98`, the cluster tree is cut using the Majona criterion. Subtrees with 
relatively small number of observations are declared to be outliers.

# Output
- `["outliers"]`: Vector of indices of outliers.
- `["betas"]`: Vector of regression coefficients.


# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> asm2000(reg0001)
Dict{Any, Any} with 2 entries:
  "betas"    => [-63.4816, 1.30406]
  "outliers" => [15, 16, 17, 18, 19, 20]
```

# References
Robiah Adnan, Mohd Nor Mohamad, & Halim Setan (2001). 
Identifying multiple outliers in linear regression: robust fit and clustering approach. 
Proceedings of the Malaysian Science and Technology Congress 2000: Symposium C, Vol VI, (p. 400). 
Malaysia: Confederation of Scientific and Technological Associations in Malaysia COSTAM.
"""
function asm2000(setting::RegressionSetting)::Dict
    X, y = @extractRegressionSetting setting
    return asm2000(X, y)
end


function asm2000(X::AbstractMatrix{Float64}, y::AbstractVector{Float64})::Dict
    n, p = size(X)
    h = floor((n + p - 1) / 2)
    ltsreg = lts(X, y)

    betas = ltsreg["betas"]
    hsubset = ltsreg["hsubset"]

    predicteds = [sum(X[i, :] .* betas) for i = 1:n]
    resids = y .- predicteds
 
    stdres = zstandardize(resids)
    stdfit = zstandardize(predicteds)

    pairs = hcat(stdfit, stdres)

    pairs = hcat(resids, predicteds)

    covmatrix = cov(pairs[hsubset, :])
    mahdist = mahalanobisSquaredBetweenPairs(pairs, covmatrix = covmatrix)

    @assert !isnothing(mahdist)

    outlierset = Array{Int,1}(undef, 0)

    hcl = hclust(mahdist, linkage = :single)
    majonacrit = majona(hcl)
    clustermappings = cutree(hcl, h = majonacrit)
    uniquemappings = unique(clustermappings)
    for clustid in uniquemappings
        cnt = count(x -> x == clustid, clustermappings)
        if cnt >= h
            outlierset = filter(i -> clustermappings[i] != clustid, 1:n)
        end
    end

    inlierset = setdiff(1:n, outlierset)
    cleanols = ols(X[inlierset, :], y[inlierset])
    cleanbetas = coef(cleanols)

    result = Dict()
    result["outliers"] = outlierset
    result["betas"] = cleanbetas
    return result
end

end # End of the module ASM2000
