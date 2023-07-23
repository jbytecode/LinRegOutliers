module LAD

export lad

using JuMP
using HiGHS

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns

import ..HookeJeeves: hj
import ..GA: ga

"""

    lad(setting; exact = true)

Perform Least Absolute Deviations regression for a given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `exact::Bool`: If true, use exact LAD regression. If false, estimate LAD regression parameters using GA. Default is true.

# Description 
The LAD estimator searches for regression the parameters estimates that minimize the sum of absolute residuals.
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
function lad(setting::RegressionSetting; exact::Bool = true)
    X, y = @extractRegressionSetting setting
    return lad(X, y, exact = exact)
end


"""

    lad(X, y, exact = true)

Perform Least Absolute Deviations regression for a given regression setting.

# Arguments
- `X::Matrix{Float64}`: Design matrix of the linear model.
- `y::Vector{Float64}`: Response vector of the linear model.
- `exact::Bool`: If true, use exact LAD regression. If false, estimate LAD regression parameters using GA. Default is true.
"""
function lad(X::Matrix{Float64}, y::Vector{Float64}; exact::Bool = true)
    if exact
        return lad_exact(X, y)
    else
        return lad_approx(X, y)
    end
end

function lad_approx(X::Matrix{Float64}, y::Vector{Float64})
    n, p = size(X)

    mins = ones(Float64, p) * 10^6 * (-1.0)
    maxs = ones(Float64, p) * 10^6
    popsize = 100

    function fcost(par)
        return sum(abs.(y .- X * par))
    end

    garesult = ga(popsize, p, fcost, mins, maxs, 0.90, 0.05, 1, p * 1000)
    best = garesult[1]
    hookejeevesresult =
        hj(fcost, best.genes, maxiter = 109000, startstep = 10.0, endstep = 0.000001)
    betas = hookejeevesresult["par"]
    result = Dict("betas" => betas, "residuals" => y .- X * betas)
    return result
end

function lad_exact(X::Matrix{Float64}, y::Vector{Float64})
    n, p = size(X)

    m = JuMP.Model(HiGHS.Optimizer)
    set_silent(m)

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
