function lms(setting::RegressionSetting; iters=nothing, crit=2.5)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = Int(floor((n + 1.0) / 2.0))
    if iters === nothing
        iters = minimum([500 * p, 3000])
    end
    bestobjective = Inf
    bestparamaters = []
    bestres = []
    indices = collect(1:n)
    kindices = collect(p:n)
    for iter in 1:iters 
        try 
            k = rand(kindices, 1)[1]
            sampledindices = sample(indices, k, replace=false)
            ols = lm(setting.formula, setting.data[sampledindices,:])
            betas = coef(ols)
            origres = Y .- X * betas
            res = sort(origres.^2.0)
            m2 = res[h]
            if m2 < bestobjective
                bestparamaters = betas
                bestobjective = m2
                bestres = origres
            end
        catch e
            @warn e
        end
    end
    s = 1.4826 * sqrt((1.0 + (5.0 / (n - p))) * bestobjective)
    standardizedres = bestres / s
    d = Dict()
    d["coef"] = bestparamaters
    d["objective"] = bestobjective
    d["S"] = s
    d["stdres"] = standardizedres
    d["crit"] = crit
    d["outliers"] = filter(i -> abs(standardizedres[i]) > crit, 1:n)
    return d
end


