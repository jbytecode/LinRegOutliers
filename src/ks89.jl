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
    ols = lm(setting.formula, setting.data[indices, :])
    betas = coef(ols)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    useX = X[indices, :]
    useY = Y[indices]
    XX = inv(useX'useX)
    w = (Y[k] - X[k,:]' * betas) / sqrt(1 + X[k,:]' * XX * X[k,:])
    return w
end



"""
    ks89(setting; alpha = 0.05)
Perform the Kianifard & Swallow (1989) algorithm for the given regression setting.
# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `alpha::Float64`: Optional argument of the probability of rejecting the null hypothesis.
# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
julia> ks89(reg0001)
2-element Array{Int64,1}:
  4
 21
```
# References
Kianifard, Farid, and William H. Swallow. "Using recursive residuals, calculated on
adaptively-ordered observations, to identify outliers in linear regression."
Biometrics (1989): 571-585.
"""
function ks89(setting::RegressionSetting; alpha=0.05)
    stdres = studentizedResiduals(setting)
    orderingindices = sortperm(abs.(stdres))
    X = designMatrix(setting)
    n, p = size(X)
    basisindices = orderingindices[1:p]
    w = zeros(Float64, n)
    s = zeros(Float64, n)
    ws = zeros(Float64, n)
    for i in (p + 1):n
        index = orderingindices[i]
        w[index] = ks89RecursiveResidual(setting, basisindices, index)
        s[index] = jacknifedS(setting, index)
        ws[index] = w[index] / s[index]
        basisindices = orderingindices[1:i]
    end
    td = TDist(n - p - 1)
    q = quantile(td, alpha)
    result = filter(i -> abs.(ws[i]) > abs(q), 1:n)
    return result
end
