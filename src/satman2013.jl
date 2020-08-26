function satman2013(setting::RegressionSetting)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    X0 = X
    p0 = p
    if X0[:, 1] == ones(n)
        X0 = X[:, 2:end]
        p0 = p - 1
    end

    allindices = collect(1:n)

    covmat = zeros(p0, p0)

    for i in 1:p0
        for j in 1:p0
            if i == j 
                covmat[i, j] = median(abs.(X0[:, i] .- median(X0[:, i])))
            else
                covmat[i, j] = median((X0[:, i] .- median(X0[:, i])) .* (X0[:, j] .- median(X0[:, j])))
            end
        end
    end

    medians = applyColumns(median, X0)
    md2 = diag(mahalabonisSquaredMatrix(X0, meanvector = medians, covmatrix = covmat))
    md = sqrt.(md2)

    sorted_indices = sortperm(md)
    best_h_indices = sorted_indices[1:h]

    crit, bestset = iterateCSteps(setting, best_h_indices, h)
    
    regy = Y[bestset]
    regx = X[bestset,:]
    ols = lm(setting.formula, setting.data[bestset, :])
    betas = coef(ols)
    resids = Y .- (X * betas)
    med_res = median(resids)
    standardized_resids = (resids .- med_res) / median(abs.(resids .- med_res))

    outlierset = filter(i -> abs(standardized_resids[i]) > 2.5, allindices)
    
    result = Dict()
    result["outliers"] = outlierset

    return result
end
