module HS93


export hs93


import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns

import ..Diagnostics: dffits

import Distributions: TDist, quantile

import LinearAlgebra: det

"""

    hs93initialset(setting)

Perform the Hadi & Simonoff (1993) algorithm's first part for a given regression setting.
The returned array of indices are indices of clean subset length of p + 1
where p is the number of regression parameters.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> hs93initialset(reg0001)
3-element Array{Int64,1}:
 4
 3
 5
```

# References
Hadi, Ali S., and Jeffrey S. Simonoff. "Procedures for the identification of 
multiple outliers in linear models." Journal of the American Statistical 
Association 88.424 (1993): 1264-1272.
"""
function hs93initialset(setting::RegressionSetting)::Array{Int,1}
    X, y = @extractRegressionSetting setting
    return hs93initialset(X, y)
end

function hs93initialset(X::AbstractMatrix{Float64}, y::AbstractVector{Float64})::Array{Int,1}
    p = size(X, 2)
    s = p + 1
    dfs = abs.(dffits(X, y))
    sortedindices = sortperm(dfs)
    basicsetindices = sortedindices[1:s]
    return basicsetindices
end


"""

    hs93basicsubset(setting, initialindices)

Perform the Hadi & Simonoff (1993) algorithm's second part for a given regression setting.
The returned array of indices are indices of clean subset of length h
where h is at least the half of the number of observations. h is set to 
integer part of (n + p - 1) / 2.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `initialindices::Array{Int, 1}`: (p + 1) subset of clean observations. 

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> initialsetindices = hs93initialset(reg0001)
3-element Array{Int64,1}:
 4
 3
 5
 julia> hs93basicsubset(reg0001, initialsetindices)
12-element Array{Int64,1}:
  5
  9
 10
  3
  6
  4
  7
 22
 11
  8
 12
 13
```

# References
Hadi, Ali S., and Jeffrey S. Simonoff. "Procedures for the identification of 
multiple outliers in linear models." Journal of the American Statistical 
Association 88.424 (1993): 1264-1272.
"""
function hs93basicsubset(
    setting::RegressionSetting,
    initialindices::Array{Int,1},
)::Array{Int,1}
    X = designMatrix(setting)
    y = responseVector(setting)
    return hs93basicsubset(X, y, initialindices)
end


function hs93basicsubset(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    initialindices::Array{Int,1},
)::Array{Int,1}
    n, p = size(X)
    h = floor((n + p - 1) / 2)
    s = length(initialindices)
    indices = initialindices
    betas = zeros(Float64, p)
    d = zeros(Float64, n)
    orderingd = zeros(Int, n)

    for i in range(s + 1, stop = h)
        betas = X[indices, :] \ y[indices]    
        XM = X[indices, :]
        for j = 1:n
            if det(XM'XM) > 0
                xxxx = X[j, :]' * inv(XM'XM) * X[j, :]
                if j in indices
                    d[j] = abs.(y[j] - sum(X[j, :] .* betas)) / sqrt(abs(1 - xxxx))
                else
                    d[j] = abs.(y[j] - sum(X[j, :] .* betas)) / sqrt(abs(1 + xxxx))
                end
            else
                # When XM'XM is singular, the corresponding d[j] is set to the maximum of y.
                d[j] = maximum(y)
            end
        end
        orderingd .= sortperm(abs.(d))
        indices = orderingd[1:Int(i)]
    end
    return indices
end


"""

    hs93(setting; alpha = 0.05, basicsubsetindices = nothing)

Perform the Hadi & Simonoff (1993) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `alpha::Float64`: Optional argument of the probability of rejecting the null hypothesis.
- `basicsubsetindices::Array{Int, 1}`: Initial basic subset, by default, the algorithm creates an initial set of clean observations.

# Description
Performs a forward search by selecting and enlarging an initial clean subset of observations and 
iterates until scaled residuals exceeds a threshold.
 
# Output
- `["outliers"]`: Array of indices of outliers
- `["t"]`: Threshold, specifically, calculated quantile of a Student-T distribution
- `["d"]`: Internal and external scaled residuals. 
- `["betas"]: Vector of estimated regression coefficients.
- `["converged"]: Boolean value indicating whether the algorithm converged or not.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> hs93(reg0001)
Dict{Any,Any} with 3 entries:
  "outliers" => [14, 15, 16, 17, 18, 19, 20, 21]
  "t"        => -3.59263
  "d"        => [2.04474, 1.14495, -0.0633255, 0.0632934, -0.354349, -0.766818, -1.06862, -1.47638, -0.7…
  "converged"=> true
```

# References
Hadi, Ali S., and Jeffrey S. Simonoff. "Procedures for the identification of 
multiple outliers in linear models." Journal of the American Statistical 
Association 88.424 (1993): 1264-1272.
"""
function hs93(setting::RegressionSetting; alpha = 0.05, basicsubsetindices = nothing)
    X = designMatrix(setting)
    y = responseVector(setting)
    return hs93(X, y, alpha = alpha, basicsubsetindices = basicsubsetindices)
end


function hs93(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64};
    alpha = 0.05,
    basicsubsetindices = nothing,
)


    if isnothing(basicsubsetindices)
        initialsetindices = hs93initialset(X, y)
        basicsubsetindices = hs93basicsubset(X, y, initialsetindices)
    end

    indices = basicsubsetindices
    n, p = size(X)
    s = length(indices)
    betas = zeros(Float64, p)
    d = zeros(Float64, n)
    orderingd = zeros(Int, n)
    
    while s < n
        betas .= X[indices, :] \ y[indices]
        resids = y[indices] - X[indices,:] * betas
        sigma = sqrt(sum(resids .^ 2.0) / (length(resids) - p))
        
        XM = X[indices, :]

        if det(XM'XM) <= 0
            return Dict(
                "d" => [],
                "t" => [],
                "outliers" => [],
                "betas" => betas,
                "converged" => false,
            )
        end

        iXmXm = inv(XM'XM)
        for j = 1:n
            xMMx = X[j, :]' * iXmXm * X[j, :]
            if j in indices
                d[j] = (y[j] - sum(X[j, :] .* betas)) / (sigma * sqrt(abs(1.0 - xMMx)))
            else
                d[j] = (y[j] - sum(X[j, :] .* betas)) / (sigma * sqrt(abs(1.0 + xMMx)))
            end
        end
        orderingd .= sortperm(abs.(d))
        tdist = TDist(s - p)
        tcalc = quantile(tdist, alpha / (2 * (s + 1)))


        outlierset = filter(x -> abs(d[x]) > abs(tcalc), 1:n)
        inlierset = setdiff(1:n, outlierset)
        cleanbetas = X[inlierset, :] \ y[inlierset]
        if abs(d[orderingd][s+1]) > abs(tcalc)
            result = Dict(
                "d" => d,
                "t" => tcalc,
                "outliers" => outlierset,
                "betas" => cleanbetas,
                "converged" => true,
            )
            return result
        end
        s += 1
        indices = orderingd[1:s]
    end

    return Dict(
        "d" => [],
        "t" => [],
        "outliers" => [],
        "betas" => betas,
        "converged" => false,
    )
end


end # end of module HS93
