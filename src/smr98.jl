module SMR98


export majona, smr98


import Clustering: Hclust, hclust, cutree
import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, zstandardize
import Distributions: mean, std
import ..OrdinaryLeastSquares: ols, residuals, predict, coef

function distances(resids::AbstractVector{Float64}, fitteds::Vector{Float64})::AbstractMatrix{Float64}
    n = length(resids)
    d = zeros(Float64, n, n)
    for i = 1:n
        for j = i:n
            if i != j
                p1 = [resids[i], fitteds[i]]
                p2 = [resids[j], fitteds[j]]
                d[i, j] = sqrt(sum((p1 .- p2) .^ 2.0))
                d[j, i] = d[i, j]
            end
        end
    end
    return d
end

function majona(cluster::Hclust)::Float64
    heights = cluster.heights
    return mean(heights) + 1.25 * std(heights)
end

"""

    smr98(setting)

Perform the Sebert, Monthomery and Rollier (1998) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description 
The algorithm starts with an ordinary least squares 
estimation for a given model and data. Residuals and fitted responses are calculated 
using the estimated model. A hierarchical clustering analysis is applied using standardized
residuals and standardized fitted responses. The tree structure of clusters are cut using
a threshold, e.g Majona criterion, as suggested by the authors. It is expected that 
the subtrees with relatively small number of observations are declared to be clusters of outliers.

# Output
- `["outliers"]`: Array of indices of outliers.
- `["betas"]`: Vector of regression coefficients.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> smr98(reg0001)
Dict{String, Vector} with 2 entries:
  "betas"    => [-55.4519, 1.15692]
  "outliers" => [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
```

# References
Sebert, David M., Douglas C. Montgomery, and Dwayne A. Rollier. "A clustering algorithm for 
identifying multiple outliers in linear regression." Computational statistics & data analysis 
27.4 (1998): 461-484.
"""
function smr98(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return smr98(X, y)
end


"""

    smr98(X, y)

Perform the Sebert, Monthomery and Rollier (1998) algorithm for the given regression setting.

# Arguments
- `X::AbstractMatrix{Float64}`: Design matrix of the linear regression model.
- `y::AbstractVector{Float64}`: Response vector of the linear regression model.


# References
Sebert, David M., Douglas C. Montgomery, and Dwayne A. Rollier. "A clustering algorithm for 
identifying multiple outliers in linear regression." Computational statistics & data analysis 
27.4 (1998): 461-484.
"""

function smr98(X::AbstractMatrix{Float64}, y::AbstractVector{Float64})
    olsreg = ols(X, y)
    #stdres = standardize(ZScoreTransform, residuals(olsreg), dims = 1)
    #stdfit = standardize(ZScoreTransform, predict(olsreg), dims = 1)
    stdres = zstandardize(residuals(olsreg))
    stdfit = zstandardize(predict(olsreg))
    n, p = size(X)
    d = distances(stdres, stdfit)
    h = floor((n + p - 1) / 2)
    hcl = hclust(d, linkage = :single)
    majonacrit = majona(hcl)
    clustermappings = cutree(hcl, h = majonacrit)
    uniquemappings = unique(clustermappings)
    for clustid in uniquemappings
        cnt = count(x -> x == clustid, clustermappings)
        if cnt >= h
            outlierset = filter(i -> clustermappings[i] != clustid, 1:n)
            inlierset = setdiff(1:n, outlierset)
            cleanols = ols(X[inlierset, :], y[inlierset])
            cleanbetas = coef(cleanols)
            return Dict("outliers" => outlierset, "betas" => cleanbetas)
        end
    end

    return Dict("outliers" => [], "betas" => coef(olsreg))
end

end # end of module SMR98
