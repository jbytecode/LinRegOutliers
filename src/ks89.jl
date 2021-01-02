"""
    ks89RecursiveResidual(setting; indices, k)
Calculate recursive residual for the given regression setting and observation.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `indices::ArrayInt,1`: Indices of observations used in the linear model.
- `k::Int`: Observation indice the recursive residual is calculated for.

# Notes
    This is a helper function for the ks89 function and it is not directly used.

# References
Kianifard, Farid, and William H. Swallow. "Using recursive residuals, calculated on
adaptively-ordered observations, to identify outliers in linear regression."
Biometrics (1989): 571-585.
"""
function ks89RecursiveResidual(setting::RegressionSetting, indices::Array{Int,1}, k::Int)
    X, y = @extractRegressionSetting setting
    return ks89RecursiveResidual(X, y, indices, k)
end

function ks89RecursiveResidual(X::Array{Float64,2}, y::Array{Float64,1}, indices::Array{Int,1}, k::Int)
    useX = X[indices, :]
    useY = y[indices]
    olsreg = ols(useX, useY)
    betas = coef(olsreg)
    XX = inv(useX'useX)
    w = (y[k] - X[k,:]' * betas) / sqrt(1 + X[k,:]' * XX * X[k,:])
    return w
end


"""
    ks89(setting; alpha = 0.05)

Perform the Kianifard & Swallow (1989) algorithm for the given regression setting.


# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `alpha::Float64`: Optional argument of the probability of rejecting the null hypothesis.

# Description
The algorithm starts with a clean subset of observations. This initial set is then enlarged 
using recursive residuals. When the calculated statistics exceeds a threshold it terminates. 


# Output
- `["outliers]`: Array of indices of outliers.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
julia> ks89(reg0001)
Dict{String,Array{Int64,1}} with 1 entry:
  "outliers" => [4, 21]
```
# References
Kianifard, Farid, and William H. Swallow. "Using recursive residuals, calculated on
adaptively-ordered observations, to identify outliers in linear regression."
Biometrics (1989): 571-585.
"""
function ks89(setting::RegressionSetting; alpha=0.05)
    X = designMatrix(setting)
    y = responseVector(setting)
    return ks89(X, y, alpha=alpha)
end


function ks89(X::Array{Float64,2}, y::Array{Float64,1}; alpha=0.05)
    stdres = studentizedResiduals(X, y)
    orderingindices = sortperm(abs.(stdres))
    n, p = size(X)
    basisindices = orderingindices[1:p]
    w = zeros(Float64, n)
    s = zeros(Float64, n)
    ws = zeros(Float64, n)
    @inbounds for i in (p + 1):n
        index = orderingindices[i]
        w[index] = ks89RecursiveResidual(X, y, basisindices, index)
        s[index] = jacknifedS(X, y, index)
        ws[index] = w[index] / s[index]
        basisindices = orderingindices[1:i]
    end
    td = TDist(n - p - 1)
    q = quantile(td, alpha)
    result = filter(i -> abs.(ws[i]) > abs(q), 1:n)
    result = Dict(
        "outliers" => result
    )
    return result
end