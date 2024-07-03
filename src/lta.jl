module LTA


export lta


import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns


import Combinatorics: combinations
import Distributions: sample

"""

    lta(setting; exact = false, earlystop = true)

Perform the Hawkins & Olive (1999) algorithm (Least Trimmed Absolute Deviations) 
for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `exact::Bool`: Consider all possible subsets of p or not where p is the number of regression parameters.
- `earlystop::Bool`: Early stop if the best objective does not change in number of remaining iters / 5 iterations.

# Description
`lta` is a trimmed version of `lad` in which the sum of first h absolute residuals is minimized
where h is Int(floor((n + p + 1.0) / 2.0)). 


# Output
- `["betas"]`: Estimated regression coefficients
- `["objective]`: Objective value


# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> lta(reg0001)
Dict{Any,Any} with 2 entries:
  "betas"     => [-55.5, 1.15]
  "objective" => 5.7

julia> lta(reg0001, exact = true)
Dict{Any,Any} with 2 entries:
  "betas"     => [-55.5, 1.15]
  "objective" => 5.7  
```

# References
Hawkins, Douglas M., and David Olive. "Applications and algorithms for least trimmed sum of 
absolute deviations regression." Computational Statistics & Data Analysis 32.2 (1999): 119-134.
"""
function lta(setting::RegressionSetting; exact = false, earlystop = true)
    X, y = @extractRegressionSetting setting
    return lta(X, y, exact = exact, earlystop = earlystop)
end



"""

    lta(X, y; exact = false)

Perform the Hawkins & Olive (1999) algorithm (Least Trimmed Absolute Deviations) 
for the given regression setting.

# Arguments
- `X::AbstractMatrix{Float64}`: Design matrix of linear regression model.
- `y::AbstractVector{Float64}`: Response vector of linear regression model.
- `exact::Bool`: Consider all possible subsets of p or not where p is the number of regression parameters.
- `earlystop::Bool`: Early stop if the best objective does not change in number of remaining iters / 5 iterations.



# References
Hawkins, Douglas M., and David Olive. "Applications and algorithms for least trimmed sum of 
absolute deviations regression." Computational Statistics & Data Analysis 32.2 (1999): 119-134.
"""
function lta(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; exact = false, earlystop = true)
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    if exact
        psubsets = collect(combinations(1:n, p))
    else
        iters = p * 3000
        psubsets = [sample(1:n, p, replace = false) for _ = 1:iters]
    end

    function lta_cost(subsetindices::Array{Int,1})::Tuple{Float64,Vector{Float64}}
        try
            betas = X[subsetindices, :] \ y[subsetindices]
            sorted_res_abs = sort!(abs.(y .- X * betas))
            cost = sum(sorted_res_abs[1:h])
            return (cost, betas)
        catch
            return (Inf64, [])
        end
    end

    L = length(psubsets)
    bestobjective = Inf64 
    bestbetas = X \ y
    numberofitersunchanged = 0
    for i in 1:L
        objective, betas = lta_cost(psubsets[i])
        if objective < bestobjective
            bestobjective = objective 
            bestbetas = betas 
            numberofitersunchanged = 0
        else
            numberofitersunchanged += 1
        end
        if !exact && (numberofitersunchanged > ((L - i) / 5))
            break
        end
    end 

    result = Dict(
        "betas" => bestbetas,
        "objective" => bestobjective)

    return result
end



end # end of module LTA 
