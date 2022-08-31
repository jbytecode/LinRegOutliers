module LAD

export lad

using JuMP
using GLPK

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns

"""

    lad(setting)

Perform Least Absolute Deviations regression for a given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description 
The LAD estimator searches for regression parameters estimates that minimizes the sum of absolute residuals.
The optimization problem is 

Min z = u1(-) + u1(+) + u2(-) + u2(+) + .... + un(-) + un(+)
Subject to:
    y_1 - beta0 - beta1 * x_2 + u1(-) - u1(+) = 0
    y_2 - beta0 - beta1 * x_2 + u2(-) - u2(+) = 0
    .
    .
    .
    y_n - beta0 - beta1 * x_n + un(-) - un(+) = 0
where 
    ui(-), ui(+) >= 0
    i = 1, 2, ..., n 
    beta0, beta1 in R 
    n : Number of observations 

# Output
- `["betas"]`: Estimated regression coefficients
- `["residuals"]`: Regression residuals
- `["model"]`: Linear Programming Model

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> lad(reg0001)
Dict{Any,Any} with 2 entries:
  "betas"     => [-57.3269, 1.19155]
  "residuals" => [2.14958, 1.25803, 0.0664872, 0.0749413, -0.416605, -0.90815, -1.2997, -1.79124,…

```

"""
function lad(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return lad(X, y)
end


"""

    lad(X, y)

Perform Least Absolute Deviations regression for a given regression setting.

# Arguments
- `X::Array{Float64, 2}`: Design matrix of the linear model.
- `y::Array{Float64, 1}`: Response vector of the linear model.
"""
function lad(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)

    m = JuMP.Model(GLPK.Optimizer)

    JuMP.@variable(m, d[1:(2n)])
    JuMP.@variable(m, beta[1:p])

    JuMP.@objective(m, Min, sum(d[i] for i = 1:(2n)))

    for i = 1:n
        c = JuMP.@constraint(m, y[i] - sum(X[i, :] .* beta) + d[i] - d[n+i] == 0)
    end

    for i = 1:(2n)
        JuMP.@constraint(m, d[i] >= 0)
    end

    JuMP.optimize!(m)

    betahats = JuMP.value.(beta)
    residuals = y .- X * betahats

    result = Dict()
    result["betas"] = betahats
    result["residuals"] = residuals
    result["model"] = m
    return result
end

end # end of module LAD
