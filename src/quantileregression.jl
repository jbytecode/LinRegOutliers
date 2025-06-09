module QuantileRegression

export quantileregression

using JuMP
using HiGHS

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns

"""

    quantileregression(setting; tau = 0.5)

Perform Quantile Regression for a given regression setting (multiple linear regression).

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `tau::Float64`: Quantile level. Default is 0.5.

# Description 
The Quantile Regression estimator searches for the regression parameter estimates that minimize the 
 

Min z = (1 - tau) (u1(-) + u2(-) + ... + un(-)) + tau (u1(+) + u2(+) + ... + un(+))
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
    model is the y = beta1 + beta2 * x + u 

# Output
- `["betas"]`: Estimated regression coefficients
- `["residuals"]`: Regression residuals
- `["model"]`: Linear Programming Model

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> quantileregression(reg0001)
```

"""
function quantileregression(setting::RegressionSetting; tau::Float64 = 0.5)
    X, y = @extractRegressionSetting setting
    return quantileregression(X, y, tau = tau)
end


"""

    quantileregression(X, y, tau = 0.5)

Estimates parameters of linear regression using Quantile Regression Estimator for a given regression setting.

# Arguments
- `X::AbstractMatrix{Float64}`: Design matrix of the linear model.
- `y::AbstractVector{Float64}`: Response vector of the linear model.
- `tau::Float64`: Quantile level. Default is 0.5.


# Examples
```julia-repl
julia> income = [420.157651, 541.411707, 901.157457, 639.080229, 750.875606];
julia> foodexp = [255.839425, 310.958667, 485.680014, 402.997356, 495.560775];

julia> n = length(income)
julia> X = hcat(ones(Float64, n), income)

julia> result = quantileregression(X, foodexp, tau = 0.25)
```


"""
function quantileregression(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; tau::Float64 = 0.5)
    n, p = size(X)

    m = JuMP.Model(HiGHS.Optimizer)
    set_silent(m)

    # d[i] > 0 for i = 1:2n
    JuMP.@variable(m, d[1:(2n)] .>= 0)
    JuMP.@variable(m, beta[1:p])

    JuMP.@objective(
        m,
        Min,
        sum((1 - tau) * d[i] for i = 1:n) + sum(tau * d[i] for i = (n+1):2n)
    )

    for i = 1:n
        _ = JuMP.@constraint(m, y[i] - sum(X[i, :] .* beta) + d[i] - d[n+i] == 0)
    end

    JuMP.optimize!(m)

    betahats = JuMP.value.(beta)
    residuals = y .- X * betahats

    result = Dict{String, Any}()
    result["betas"] = betahats
    result["residuals"] = residuals
    result["model"] = m
    return result
end

end # end of module QuantileRegression
