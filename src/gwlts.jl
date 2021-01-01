"""
This method is a modified version of the algorithm given below.
the modification will be available in a new paper.
Working is in progress.

Satman, M. Hakan. "A genetic algorithm based modification on the lts algorithm for large data sets."
 Communications in Statistics-Simulation and Computation 41.5 (2012): 644-652.
"""

function gwcga(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return gwcga(X, y)
end

function gwcga(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))
    minp =  p + 1
    all_indices = collect(1:n)

    full_reg = ols(X, y)
    full_reg_residuals = residuals(full_reg)
    sumres1 = sum(full_reg_residuals .* full_reg_residuals)

    function costfunction(bitstring::Array{Bool,1})::Float64
        number_of_ones = sum(bitstring)
        if number_of_ones < minp
            return Inf64
        end
        
        indices = filter(i -> bitstring[i] == 1, all_indices)
        objective, clean_subset = iterateCSteps(X, y, indices, h)
        ltsreg = ols(X[clean_subset, :], y[clean_subset])
        res2 = residuals(ltsreg)
        sumres2 = sum(res2 .* res2)
        

        cost = -(sumres1 - sumres2) 
        # @info sumres1 sumres2 cost
        return cost
    end

    result_cga = cga(chsize=n, costfunction=costfunction, popsize=10)
    initial_indices = filter(i -> result_cga[i] == 1, all_indices)
    # @info "Initial subset: " result_cga initial_indices

    objective, clean_subset = iterateCSteps(X, y, initial_indices, h)
    
    ltsreg = ols(X[clean_subset, :], y[clean_subset])
    betas = coef(ltsreg)

    result = Dict()
    result["betas"] = betas
    result["initial.subset"] = sort(initial_indices)
    result["clean.subset"] = sort(clean_subset)
    return result
end




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

function galts(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))
    
    function fcost(genes::Array{Float64,1})
        objective, indices = iterateCSteps(X, y, genes, h)
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


