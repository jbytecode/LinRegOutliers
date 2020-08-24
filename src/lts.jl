function iterateCSteps(setting::RegressionSetting, subsetindices::Array{Int,1}, h::Int)
    Xall = designMatrix(setting)
    Yall = responseVector(setting)
    starterset = copy(subsetindices)
    oldobjective = Inf
    objective = Inf
    iter = 0
    maxiter = 10000
    while iter < maxiter
        n, p = size(Xall)
        X = Xall[subsetindices, :]
        Y = Yall[subsetindices, :]
        ols = lm(setting.formula, setting.data[subsetindices, :])
        betas = coef(ols)
        res = [Yall[i] - sum(Xall[i,:] .* betas) for i in 1:n]
        sortedresindices = sortperm(abs.(res))
        subsetindices = sortedresindices[1:h]
        objective = sum(sort(res.^2.0)[1:h])
        if oldobjective == objective 
            break
        end
        oldobjective = objective
        iter += 1
    end
    if iter >= maxiter
        @warn "in c-step stage of LTS, a h-subset is not converged for starting indices " starterset
    end
    return (objective, subsetindices)
end

function lts(setting::RegressionSetting; iters=nothing, crit=3)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))
    if iters === nothing
        iters = minimum([500 * p, 3000])
    end
    allindices = collect(1:n)
    bestobjective = Inf
    besthsubset = []
    for iter in 1:iters
        subsetindices = sample(allindices, p, replace=false)
        objective, hsubsetindices = iterateCSteps(setting, subsetindices, h)
        if objective < bestobjective
            bestobjective = objective 
            besthsubset = hsubsetindices
        end
    end
    ltsreg = lm(setting.formula, setting.data[besthsubset, :])
    ltsbetas = coef(ltsreg)
    ltsres = [Y[i] - sum(X[i,:] .* ltsbetas) for i in 1:n]
    ltsS = sqrt(sum((ltsres.^2.0)[1:h]) / (h - p))
    ltsresmean = mean(ltsres[besthsubset])
    ltsScaledRes = (ltsres .- ltsresmean) / ltsS
    outlierindices = filter(i -> abs(ltsScaledRes[i]) > crit, 1:n)
    result = Dict()
    result["objective"] = bestobjective
    result["hsubset"] = besthsubset
    result["betas"] = ltsbetas
    result["S"] = ltsS
    result["outliers"] = outlierindices
    result["scaled.residuals"] = ltsScaledRes
    return result
end