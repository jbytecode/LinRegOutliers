"""
This method is a modified version of the algorithm given below.
the modification will be available in a new paper.
Working is in progress.

Satman, M. Hakan. "A genetic algorithm based modification on the lts algorithm for large data sets."
 Communications in Statistics-Simulation and Computation 41.5 (2012): 644-652.
"""

function gwcga(setting::RegressionSetting)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))
    minp =  p + 1
    all_indices = collect(1:n)

    full_reg = lm(setting.formula, setting.data)
    full_reg_residuals = residuals(full_reg)
    sumres1 = sum(full_reg_residuals .* full_reg_residuals)

    function costfunction(bitstring::Array{Bool, 1})::Float64
        number_of_ones = sum(bitstring)
        if number_of_ones < minp
            return Inf64
        end
        
        indices = filter(i -> bitstring[i] == 1, all_indices)
        objective, clean_subset = iterateCSteps(setting, indices, h)
        ltsreg = lm(setting.formula, setting.data[clean_subset, :])
        res2 = residuals(ltsreg)
        sumres2 = sum(res2 .* res2)
        

        cost = -(sumres1 - sumres2) 
        # @info sumres1 sumres2 cost
        return cost
    end

    result_cga = cga(chsize = n, costfunction = costfunction, popsize = 10)
    initial_indices = filter(i -> result_cga[i] == 1, all_indices)
    # @info "Initial subset: " result_cga initial_indices

    objective, clean_subset = iterateCSteps(setting, initial_indices, h)
    
    ltsreg = lm(setting.formula, setting.data[clean_subset, :])
    betas = coef(ltsreg)

    result = Dict()
    result["betas"] = betas
    result["initial.subset"] = sort(initial_indices)
    result["clean.subset"] = sort(clean_subset)
    return result
end