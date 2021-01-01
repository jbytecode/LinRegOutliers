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

function hs93initialset(X::Array{Float64,2}, y::Array{Float64,1})::Array{Int,1}
    n, p = size(X)
    s = p + 1
    dfs = abs.(dffit(X, y))
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
function hs93basicsubset(setting::RegressionSetting, initialindices::Array{Int,1})::Array{Int,1}
    X = designMatrix(setting)
    y = responseVector(setting)
    return hs93basicsubset(X, y, initialindices)
end


function hs93basicsubset(X::Array{Float64,2}, y::Array{Float64,1}, initialindices::Array{Int,1})::Array{Int,1}
    n, p = size(X)
    h = floor((n + p - 1) / 2)
    s = length(initialindices)
    indices = initialindices
    for i in range(s + 1, stop=h)
        olsreg = ols(X[indices,:], y[indices])
        betas = coef(olsreg)
        d = zeros(Float64, n)
        XM = X[indices,:]
        for j in 1:n
            xxxx = X[j,:]' * inv(XM'XM) * X[j,:]
            if j in indices
                @inbounds d[j] = abs.(y[j] - sum(X[j,:] .* betas)) / sqrt(1 - xxxx)
            else
                @inbounds d[j] = abs.(y[j] - sum(X[j,:] .* betas)) / sqrt(1 + xxxx)
            end
        end
        orderingd = sortperm(abs.(d))
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


# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> hs93(reg0001)
Dict{Any,Any} with 3 entries:
  "outliers" => [14, 15, 16, 17, 18, 19, 20, 21]
  "t"        => -3.59263
  "d"        => [2.04474, 1.14495, -0.0633255, 0.0632934, -0.354349, -0.766818, -1.06862, -1.47638, -0.7…
```

# References
Hadi, Ali S., and Jeffrey S. Simonoff. "Procedures for the identification of 
multiple outliers in linear models." Journal of the American Statistical 
Association 88.424 (1993): 1264-1272.
"""
function hs93(setting::RegressionSetting; alpha=0.05, basicsubsetindices=nothing)
    X = designMatrix(setting)
    y = responseVector(setting)
    return hs93(X, y, alpha=alpha, basicsubsetindices=basicsubsetindices)
end


function hs93(X::Array{Float64,2}, y::Array{Float64,1}; alpha=0.05, basicsubsetindices=nothing)
    if basicsubsetindices === nothing
        initialsetindices = hs93initialset(X, y)
        basicsubsetindices = hs93basicsubset(X, y, initialsetindices)
    end
    indices = basicsubsetindices
    n, p = size(X)
    s = length(indices)
    while s < n
        olsreg = ols(X[indices, :], y[indices])
        betas = coef(olsreg)
        resids = residuals(olsreg)
        sigma = sqrt(sum(resids.^2.0) / (length(resids) - p))
        d = zeros(Float64, n)
        XM = @inbounds X[indices,:]
        iXmXm = inv(XM'XM)
        for j in 1:n
            @inbounds xMMx = X[j,:]' * iXmXm * X[j,:]
            if j in indices
                @inbounds d[j] = (y[j] - sum(X[j,:] .* betas)) / (sigma * sqrt(1.0 - xMMx))
            else
                @inbounds d[j] = (y[j] - sum(X[j,:] .* betas)) / (sigma * sqrt(1.0 + xMMx))
            end
        end
        orderingd = sortperm(abs.(d))
        tdist = TDist(s - p)
        tcalc = quantile(tdist,  alpha / (2 * (s + 1)))
        if abs(d[orderingd][s + 1]) > abs(tcalc)
            result = Dict()
            result["d"] = d
            result["t"] = tcalc
            result["outliers"] = filter(x -> abs(d[x]) > abs(tcalc), 1:n)
            return result
        end
        s += 1
        indices = orderingd[1:s]
    end
    return []
end
