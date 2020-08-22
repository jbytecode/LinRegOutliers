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
