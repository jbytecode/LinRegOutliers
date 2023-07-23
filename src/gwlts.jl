module GALTS


export galts

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..OrdinaryLeastSquares: ols, predict, residuals, coef

import ..LTS: iterateCSteps
import ..GA: ga




"""
    galts(setting)

Perform Satman(2012) algorithm for estimating LTS coefficients.

# Arguments
- `setting`: A regression setting object.

# Description 
The algorithm performs a genetic search for estimating LTS coefficients using C-Steps. 

# Output
- `["betas"]`: Robust regression coefficients
- `["best.subset"]`: Clean subset of h observations, where h is an integer greater than n / 2. The default value of h is `Int(floor((n + p + 1.0) / 2.0))`.
- `["objective"]`: Objective value


# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> galts(reg)
Dict{Any,Any} with 3 entries:
  "betas"       => [-56.5219, 1.16488]
  "best.subset" => [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24]
  "objective"   => 3.43133
```

# References
Satman, M. Hakan. "A genetic algorithm based modification on the lts algorithm for large data sets."
 Communications in Statistics-Simulation and Computation 41.5 (2012): 644-652.
"""
function galts(setting::RegressionSetting)
    X = designMatrix(setting)
    y = responseVector(setting)
    return galts(X, y)
end

function galts(X::Matrix{Float64}, y::Vector{Float64})
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    function fcost(genes::Vector{Float64})
        objective, _ = iterateCSteps(X, y, genes, h)
        return objective
    end

    mins = ones(Float64, p) * 10^6 * (-1.0)
    maxs = ones(Float64, p) * 10^6
    popsize = 30

    garesult = ga(popsize, p, fcost, mins, maxs, 0.90, 0.05, 1, 100)

    best = garesult[1]

    objective, subsetindices = iterateCSteps(X, y, best.genes, h)

    ltsreg = ols(X[subsetindices, :], y[subsetindices])
    betas = coef(ltsreg)

    result = Dict()
    result["betas"] = betas
    result["best.subset"] = sort(subsetindices)
    result["objective"] = objective
    return result
end



end # end of module GALTS 
