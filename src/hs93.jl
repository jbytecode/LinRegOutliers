module HS93


export hs93


import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns

import ..Diagnostics: dffits

import Distributions: TDist, quantile

import LinearAlgebra: ColumnNorm, PosDefException, Symmetric, cholesky, ldiv!, qr

@inline function _rowdot(
    X::AbstractMatrix{Float64},
    rowindex::Int,
    betas::AbstractVector{Float64},
)::Float64
    s = 0.0
    @inbounds @simd for j in eachindex(betas)
        s += X[rowindex, j] * betas[j]
    end
    return s
end

function _set_membership!(
    membership::AbstractVector{Bool},
    indices::AbstractVector{Int},
)::Nothing
    fill!(membership, false)
    @inbounds for idx in indices
        membership[idx] = true
    end
    return nothing
end

function _gramian!(
    target::AbstractMatrix{Float64},
    X::AbstractMatrix{Float64},
    indices::AbstractVector{Int},
)::AbstractMatrix{Float64}
    fill!(target, 0.0)
    p = size(X, 2)
    @inbounds for idx in indices
        for col = 1:p
            xcol = X[idx, col]
            for row = 1:col
                target[row, col] += X[idx, row] * xcol
            end
        end
    end
    @inbounds for col = 1:p
        for row = (col + 1):p
            target[row, col] = target[col, row]
        end
    end
    return target
end

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
    h = fld(n + p - 1, 2)
    s = length(initialindices)
    indices = copy(initialindices)
    if h > s
        resize!(indices, h)
    end
    betas = Array{Float64, 1}(undef, p)
    d = Array{Float64, 1}(undef, n)
    orderingd = Array{Int}(undef, n)
    xtx = Matrix{Float64}(undef, p, p)
    insubset = falses(n)
    solvework = Vector{Float64}(undef, p)
    ymax = maximum(y)

    for i in (s + 1):h
        activeindices = view(indices, 1:(i - 1))
        betas .= qr(view(X, activeindices, :), ColumnNorm()) \ view(y, activeindices)
        _set_membership!(insubset, activeindices)
        _gramian!(xtx, X, activeindices)

        chol = try
            cholesky(Symmetric(xtx))
        catch err
            if err isa PosDefException
                nothing
            else
                rethrow(err)
            end
        end

        for j = 1:n
            if !isnothing(chol)
                @inbounds for k = 1:p
                    solvework[k] = X[j, k]
                end
                ldiv!(chol, solvework)
                xxxx = _rowdot(X, j, solvework)
                resid = abs(y[j] - _rowdot(X, j, betas))
                if insubset[j]
                    d[j] = resid / sqrt(abs(1 - xxxx))
                else
                    d[j] = resid / sqrt(abs(1 + xxxx))
                end
            else
                d[j] = ymax
            end
        end
        sortperm!(orderingd, d, by = abs)
        copyto!(indices, 1, orderingd, 1, i)
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
    betas = Array{Float64, 1}(undef, p)
    d = Array{Float64, 1}(undef, n)
    orderingd = Array{Int, 1}(undef, n)
    if s < n
        workingsubset = copy(indices)
        resize!(workingsubset, n)
    else
        workingsubset = copy(indices)
    end
    indices = workingsubset
    xtx = Matrix{Float64}(undef, p, p)
    insubset = falses(n)
    solvework = Vector{Float64}(undef, p)
    
    while s < n
        activeindices = view(indices, 1:s)
        betas .= qr(view(X, activeindices, :), ColumnNorm()) \ view(y, activeindices)

        rss = 0.0
        @inbounds for idx in activeindices
            resid = y[idx] - _rowdot(X, idx, betas)
            rss += resid * resid
        end
        sigma = sqrt(rss / (s - p))

        _gramian!(xtx, X, activeindices)
        chol = try
            cholesky(Symmetric(xtx))
        catch err
            if err isa PosDefException
                nothing
            else
                rethrow(err)
            end
        end

        if isnothing(chol)
            return Dict(
                "d" => [],
                "t" => [],
                "outliers" => [],
                "betas" => betas,
                "converged" => false,
            )
        end
        _set_membership!(insubset, activeindices)
        for j = 1:n
            @inbounds for k = 1:p
                solvework[k] = X[j, k]
            end
            ldiv!(chol, solvework)
            xMMx = _rowdot(X, j, solvework)
            resid = y[j] - _rowdot(X, j, betas)
            if insubset[j]
                d[j] = resid / (sigma * sqrt(abs(1.0 - xMMx)))
            else
                d[j] = resid / (sigma * sqrt(abs(1.0 + xMMx)))
            end
        end
        sortperm!(orderingd, d, by = abs)
        tdist = TDist(s - p)
        tcalc = quantile(tdist, alpha / (2 * (s + 1)))

        if abs(d[orderingd[s + 1]]) > abs(tcalc)
            outliercount = 0
            @inbounds for j = 1:n
                if abs(d[j]) > abs(tcalc)
                    outliercount += 1
                end
            end
            outlierset = Vector{Int}(undef, outliercount)
            inlierset = Vector{Int}(undef, n - outliercount)
            outlierpos = 1
            inlierpos = 1
            @inbounds for j = 1:n
                if abs(d[j]) > abs(tcalc)
                    outlierset[outlierpos] = j
                    outlierpos += 1
                else
                    inlierset[inlierpos] = j
                    inlierpos += 1
                end
            end
            cleanbetas = qr(view(X, inlierset, :), ColumnNorm()) \ view(y, inlierset)
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
        copyto!(indices, 1, orderingd, 1, s)
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
