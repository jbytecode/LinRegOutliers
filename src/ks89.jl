module KS89


export ks89

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..OrdinaryLeastSquares: ols, predict, residuals, coef
import ..Diagnostics: studentizedResiduals, jacknifedS

import Distributions: TDist, quantile



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

function ks89RecursiveResidual(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    indices::Array{Int,1},
    k::Int,
)
    useX = X[indices, :]
    useY = y[indices]
    olsreg = ols(useX, useY)
    betas = coef(olsreg)
    XX = inv(useX'useX)
    w = (y[k] - X[k, :]' * betas) / sqrt(1 + X[k, :]' * XX * X[k, :])
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
- `["betas"]`: Vector of regression coefficients.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
julia> ks89(reg0001)
Dict{String, Vector} with 2 entries:
  "betas"    => [-42.4531, 0.956605, 0.555571, -0.108766]
  "outliers" => [4, 21]
```
# References
Kianifard, Farid, and William H. Swallow. "Using recursive residuals, calculated on
adaptively-ordered observations, to identify outliers in linear regression."
Biometrics (1989): 571-585.
"""
function ks89(setting::RegressionSetting; alpha = 0.05)
    X = designMatrix(setting)
    y = responseVector(setting)
    return ks89(X, y, alpha = alpha)
end


function ks89(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; alpha = 0.05)::Dict
    stdres = studentizedResiduals(X, y)
    orderingindices = sortperm(abs.(stdres))
    n, p = size(X)
    basisindices = orderingindices[1:p]
    w = zeros(Float64, n)
    s = zeros(Float64, n)
    ws = zeros(Float64, n)
    for i = (p+1):n
        index = orderingindices[i]
        w[index] = ks89RecursiveResidual(X, y, basisindices, index)
        s[index] = jacknifedS(X, y, index)
        ws[index] = w[index] / s[index]
        basisindices = orderingindices[1:i]
    end
    td = TDist(n - p - 1)
    q = quantile(td, alpha)

    outlierindices = filter(i -> abs.(ws[i]) > abs(q), 1:n)
    inlierindices = setdiff(1:n, outlierindices)
    cleanols = ols(X[inlierindices, :], y[inlierindices])
    cleanbetas = coef(cleanols)

    result = Dict("outliers" => outlierindices, "betas" => cleanbetas)

    return result
end



end # end of module KS89
