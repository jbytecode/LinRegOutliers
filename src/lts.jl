module LTS

export lts
export iterateCSteps

import ..Basis: RegressionSetting, @extractRegressionSetting, designMatrix, responseVector
import ..OrdinaryLeastSquares: residuals, coef, olsf!, olsf

import Distributions: sample!
import LinearAlgebra: mul!

"""

    iterateCSteps(setting, subsetindices, h)

Perform a concentration step for a given subset of a regression setting. 

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `subsetindices::Array{Int, 1}`: Indices of observations in the initial subset.
- `h::Int`: A constant at least half of the number of observations.
- `eps::Float64`: A small number, default is 0.01. If difference of last two objectives is less than eps, function terminates.
- `maxiter::Int`: Maximum number of iteration. Default is 10000.

# Notes
    This function is a helper for the lts function. A concentration step starts with a 
    initial subset. The size of the subset is enlarged to h, a constant at least half of the 
    observations. Please refer to the citation given below.

# References
Rousseeuw, Peter J., and Katrien Van Driessen. "An algorithm for positive-breakdown 
regression based on concentration steps." Data Analysis. 
Springer, Berlin, Heidelberg, 2000. 335-346.
"""
function iterateCSteps(setting::RegressionSetting, 
    subsetindices::Array{Int,1}, 
    h::Int; eps::Float64 = 0.01, maxiter::Int = 10000)
    X, y = @extractRegressionSetting setting
    return iterateCSteps(X, y, subsetindices, h, eps = eps, maxiter = maxiter)
end


function iterateCSteps(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    subsetindices::AbstractVector{Int},
    h::Int; eps::Float64 = 0.01, maxiter::Int = 10000
)
    n, p = size(X)
    oldobjective::Float64 = Inf64
    objective::Float64 = Inf64
    iter::Int = 0
    sortedresindices = Array{Int}(undef, n)
    tempbetas = Vector{Float64}(undef, p)
    fitted = Vector{Float64}(undef, n)
    absres = Vector{Float64}(undef, n)
    workingsubset = copy(subsetindices)
    if h > length(workingsubset)
        resize!(workingsubset, h)
    end
    workingsubsetlen = length(subsetindices)
    while iter < maxiter
        activeindices = view(workingsubset, 1:workingsubsetlen)
        olsf!(view(X, activeindices, :), view(y, activeindices), tempbetas)
        mul!(fitted, X, tempbetas)
        @inbounds for i in eachindex(y)
            absres[i] = abs(y[i] - fitted[i])
        end
        sortperm!(sortedresindices, absres)
        objective = 0.0
        @inbounds for i = 1:h
            idx = sortedresindices[i]
            workingsubset[i] = idx
            objective += abs2(absres[idx])
        end
        workingsubsetlen = h
        if abs(oldobjective - objective) < eps
            break
        end
        oldobjective = objective
        iter += 1
    end
    resize!(workingsubset, workingsubsetlen)
    return (objective, workingsubset)
end


function iterateCSteps(setting::RegressionSetting, 
    initialBetas::AbstractVector{Float64}, 
    h::Int; eps::Float64 = 0.01, maxiter::Int = 10000)
    X = designMatrix(setting)
    y = responseVector(setting)
    return iterateCSteps(X, y, initialBetas, h, eps = eps, maxiter = maxiter)
end

function iterateCSteps(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    initialBetas::AbstractVector{Float64},
    h::Int; eps::Float64 = 0.01, maxiter::Int = 10000
)
    p = size(X, 2)
    res = Vector{Float64}(undef, length(y))
    mul!(res, X, initialBetas)
    @inbounds for i in eachindex(y)
        res[i] = abs(y[i] - res[i])
    end
    sortedresindices = Array{Int}(undef, length(y))
    sortperm!(sortedresindices, res)
    subsetindices = Vector{Int}(undef, p)
    copyto!(subsetindices, 1, sortedresindices, 1, p)
    return iterateCSteps(X, y, subsetindices, h, eps = eps, maxiter = maxiter)
end



"""

    lts(setting; iters = nothing, crit = 2.5, earlystop = true)

Perform the Fast-LTS (Least Trimmed Squares) algorithm for a given regression setting. 

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `iters::Int`: Number of iterations.
- `crit::Float64`: Critical value.
- `earlystop::Bool`: Early stop if the best objective does not change in iters / 2 iterations.

# Description 
The algorithm searches for estimations of regression parameters which minimize the sum of first h 
ordered squared residuals where h is Int(floor((n + p + 1.0) / 2.0)). Specifically, our implementation, 
uses the algorithm Fast-LTS in which concentration steps are used for enlarging a basic 
subset to subset of clean observation of size h.  


# Output
- `["betas"]`: Estimated regression coefficients
- `["S"]`: Standard error of regression
- `["hsubset"]`: Best subset of clean observation of size h.
- `["outliers"]`: Array of indices of outliers
- `["scaled.residuals"]`: Array of scaled residuals
- `["objective"]`: LTS objective value.


# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> lts(reg)
Dict{Any,Any} with 6 entries:
  "betas"            => [-56.5219, 1.16488]
  "S"                => 1.10918
  "hsubset"          => [11, 10, 5, 6, 23, 12, 13, 9, 24, 7, 3, 4, 8]
  "outliers"         => [14, 15, 16, 17, 18, 19, 20, 21]
  "scaled.residuals" => [2.41447, 1.63472, 0.584504, 0.61617, 0.197052, -0.222066, -0.551027, -0.970146, -0.397538, -0.185558  …  …
  "objective"        => 3.43133
```

# References
Rousseeuw, Peter J., and Katrien Van Driessen. "An algorithm for positive-breakdown 
regression based on concentration steps." Data Analysis. 
Springer, Berlin, Heidelberg, 2000. 335-346.
"""
function lts(setting::RegressionSetting; iters=nothing, crit=2.5, earlystop = true)
    X = designMatrix(setting)
    y = responseVector(setting)
    return lts(X, y, iters=iters, crit=crit, earlystop = earlystop)
end

function lts(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; iters=nothing, crit=2.5, earlystop = true)

    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    if isnothing(iters)
        iters = min(5 * p, 3000)
    end

    allindices = collect(1:n)
    bestobjective = Inf
    besthsubset = Array{Int}(undef, h)
    subsetindices = Array{Int}(undef, p)

    bestobjectiveunchanged = 0

    for _ = 1:iters
        sample!(allindices, subsetindices, replace=false)
        objective, hsubsetindices = iterateCSteps(X, y, subsetindices, h)
        if objective < bestobjective
            bestobjective = objective
            besthsubset .= hsubsetindices
            bestobjectiveunchanged = 0
        else
            bestobjectiveunchanged += 1
            if earlystop && bestobjectiveunchanged >= 100
                break
            end
        end
    end

    ltsbetas = olsf(view(X, besthsubset, :), view(y, besthsubset))
    ltsres = Vector{Float64}(undef, n)
    mul!(ltsres, X, ltsbetas)
    @inbounds for i in eachindex(y)
        ltsres[i] = y[i] - ltsres[i]
    end

    ltsSsum = 0.0
    @inbounds for i = 1:h
        ltsSsum += abs2(ltsres[i])
    end
    ltsS = sqrt(ltsSsum / (h - p))

    ltsresmean = 0.0
    @inbounds for idx in besthsubset
        ltsresmean += ltsres[idx]
    end
    ltsresmean /= h

    ltsScaledRes = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(ltsres)
        ltsScaledRes[i] = (ltsres[i] - ltsresmean) / ltsS
    end

    outlierindices = Int[]
    sizehint!(outlierindices, n)
    @inbounds for i = 1:n
        if abs(ltsScaledRes[i]) > crit
            push!(outlierindices, i)
        end
    end

    result = Dict(
        "objective" => bestobjective,
        "hsubset" => besthsubset,
        "betas" => ltsbetas,
        "S" => ltsS,
        "outliers" => outlierindices,
        "scaled.residuals" => ltsScaledRes
    )
    return result
end

end # End of module LTS 
