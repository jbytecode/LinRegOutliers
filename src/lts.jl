"""

    iterateCSteps(setting, subsetindices, h)

Perform a concentration step for a given subset of a regression setting. 

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `subsetindices::Array{Int, 1}`: Indicies of observations in the initial subset.
- `h::Int`: A constant at least half of the number of observations.

# Notes
    This function is a helper for the lts function. A concentration step starts with a 
    initial subset. The size of the subset is enlarged to h, a constant at least half of the 
    observations. Please refer to the citation given below.

# References
Rousseeuw, Peter J., and Katrien Van Driessen. "An algorithm for positive-breakdown 
regression based on concentration steps." Data Analysis. 
Springer, Berlin, Heidelberg, 2000. 335-346.
"""
function iterateCSteps(setting::RegressionSetting, subsetindices::Array{Int,1}, h::Int)
    X, y = @extractRegressionSetting setting
    return iterateCSteps(X, y, subsetindices, h)
end


function iterateCSteps(X::Array{Float64,2}, y::Array{Float64,1}, subsetindices::Array{Int,1}, h::Int)
    starterset = copy(subsetindices)
    oldobjective = Inf
    objective = Inf
    iter = 0
    maxiter = 10000
    n, p = size(X)
    while iter < maxiter
        try
            Xsub = X[subsetindices, :]
            ysub = y[subsetindices]
            olsreg = ols(Xsub, ysub)
            betas = coef(olsreg)
            res = [y[i] - sum(X[i,:] .* betas) for i in 1:n]
            sortedresindices = sortperm(abs.(res))
            subsetindices = sortedresindices[1:h]
            objective = sum(sort(res.^2.0)[1:h])
            if oldobjective == objective 
                break
            end
            oldobjective = objective
            iter += 1
        catch er
            @warn er
            return (objective, subsetindices)
        end
    end
    if iter >= maxiter
        @warn "in c-step stage of LTS, a h-subset is not converged for starting indices " starterset
    end
    return (objective, subsetindices)
end


function iterateCSteps(setting::RegressionSetting, initialBetas::Array{Float64,1}, h::Int)
    X = designMatrix(setting)
    y = responseVector(setting)
    return iterateCSteps(X, y, initialBetas, h)  
end

function iterateCSteps(X::Array{Float64,2}, y::Array{Float64,1}, initialBetas::Array{Float64,1}, h::Int)
    n, p = size(X)
    res = [y[i] - sum(X[i,:] .* initialBetas) for i in 1:n]
    sortedresindices = sortperm(abs.(res))
    subsetindices = sortedresindices[1:p]
    return iterateCSteps(X, y, subsetindices, h)    
end



"""

    lts(setting; iters, crit)

Perform the Fast-LTS (Least Trimmed Squares) algorithm for a given regression setting. 

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `iters::Int`: Number of iterations.
- `crit::Float64`: Critical value.

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
function lts(setting::RegressionSetting; iters=nothing, crit=2.5)
    X = designMatrix(setting)
    y = responseVector(setting)
    return lts(X, y, iters=iters, crit=crit)
end

function lts(X::Array{Float64,2}, y::Array{Float64,1}; iters=nothing, crit=2.5)
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))
    if iters === nothing
        iters = minimum([500 * p, 3000])
    end
    allindices = collect(1:n)
    bestobjective = Inf
    besthsubset = []
    for iter in 1:iters
        subsetindices = sample(allindices, p, replace=false)
        objective, hsubsetindices = iterateCSteps(X, y, subsetindices, h)
        if objective < bestobjective
            bestobjective = objective 
            besthsubset = hsubsetindices
        end
    end
    ltsreg = ols(X[besthsubset, :], y[besthsubset])
    ltsbetas = coef(ltsreg)
    ltsres = [y[i] - sum(X[i,:] .* ltsbetas) for i in 1:n]
    ltsS = sqrt(sum((ltsres.^2.0)[1:h]) / (h - p))
    ltsresmean = mean(ltsres[besthsubset])
    ltsScaledRes = (ltsres .- ltsresmean) / ltsS
    outlierindices = filter(i -> abs(ltsScaledRes[i]) > crit, 1:n)
    result = Dict()
    result["objective"] = bestobjective
    result["hsubset"] = besthsubset
    result["betas"] = ltsbetas
    result["S"] = ltsS
    result["outliers"] = outlierindices
    result["scaled.residuals"] = ltsScaledRes
    return result
end

