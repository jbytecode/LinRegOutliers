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
    w = Y[k] - sum(X[k,:] .* betas) / sqrt(1 + X[k,:]' * XX * X[k,:])
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
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> ks89(reg0001)
6-element Array{Int64,1}:
 15
 16
 17
 18
 19
 20
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
        w[i] = ks89RecursiveResidual(setting, basisindices, i)
        s[i] = jacknifedS(setting, i)
        ws[i] = w[i] / s[i]
        basisindices = orderingindices[1:i]
    end
    td = TDist(n - p - 1)
    q = quantile(td, alpha)
    result = filter(i -> abs.(ws[i]) > abs(q), 1:n)
    return result
end

