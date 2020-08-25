function coordinatwisemedians(datamat::Array{Float64,2})::Array{Float64,1}
    n, p = size(datamat)
    meds = map(i -> median(datamat[:, i]), 1:p)
    return meds
end

function bch(setting::RegressionSetting; alpha=0.05, maxiter=1000, epsilon=0.000001)
    Xdesign = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(Xdesign)
    h = Int(floor((n + p + 1.0) / 2))
    X = Xdesign
    if Xdesign[:, 1] == ones(n)
        X = Xdesign[:, 2:end]
        n, p = size(X)
    end

    cnp = 1 + ((p + 1) / (n - p)) + (2 / (n - 1 - (3 * p)))
    chidist = Chisq(p)
    chicrit = quantile(chidist, 1 - alpha / n)
    crit = cnp * chicrit

    c = (2 * (n + 2 * p)) / (n - 2 * p)

    estimatedvariance = 0.0
    estimatedvariances = []

    # Algorithm 2 - Step 0.a
    coordmeds = coordinatwisemedians(X)
    A = ((X .- coordmeds')' * (X .- coordmeds')) / (n - 1)
    dsquared = diag(mahalabonisSquaredMatrix(DataFrame(X), meanvector=coordmeds, covmatrix=A))
    d = sqrt.(dsquared)

    # Algorithm 2 - Step 0.b
    bestindicesofd = sortperm(d)[1:h]
    colmeansofh = map(i -> mean(X[bestindicesofd, i]), 1:p)
    covmatofh = cov(X[bestindicesofd,:])
    newdsquared = diag(mahalabonisSquaredMatrix(DataFrame(X), meanvector=colmeansofh, covmatrix=covmatofh))
    newd = sqrt.(newdsquared)

    # Algorithm 2 - Steps 1, 2, 3
    basicsubsetindices = sortperm(newd)[1:(p + 1)]
    while length(basicsubsetindices) < h
        colmeanofbasicsubset = map(i -> mean(X[basicsubsetindices, i]), 1:p)
        covmatofbasicsubset = cov(X[basicsubsetindices,:]) 
        newdsquared = diag(mahalabonisSquaredMatrix(DataFrame(X), meanvector=colmeanofbasicsubset, covmatrix=covmatofbasicsubset))
        newd = sqrt.(newdsquared)
        basicsubsetindices = sortperm(newd)[1:(length(basicsubsetindices) + 1)]
    end

    # Algorithm 2 - Steps 4
    while length(basicsubsetindices) < n
        r = length(basicsubsetindices)
        colmeanofbasicsubset = map(i -> mean(X[basicsubsetindices, i]), 1:p)
        covmatofbasicsubset = cov(X[basicsubsetindices,:]) 
        newdsquared = diag(mahalabonisSquaredMatrix(DataFrame(X), meanvector=colmeanofbasicsubset, covmatrix=covmatofbasicsubset))
        newd = sqrt.(newdsquared)
        sortednewdsquared = sort(newdsquared)
        if sortednewdsquared[r + 1] >= crit 
            break
        end
        basicsubsetindices = sortperm(newd)[1:(r + 1)]
    end

    # Algorithm 3 - Fitting
    squared_normalized_robust_distances = newd.^2.0 / sum(newd.^2.0)
    md = median(newd)
    newdmd = [newd[i] / maximum([newd[i], md]) for i in 1:n]
    newdmd2 = newdmd.^2.0
    sumnewdmd2 = sum(newdmd2)
    weights = newdmd2 / sumnewdmd2 

    # Algorithm 3 - Step j
    betas = []
    squared_normalized_resids = []
    resids = []
    for i in 1:maxiter
        Yw = Y .* weights
        Xw = Xdesign .* weights
        betas = (inv((Xw)' * Xw) * Xw' * Yw) 
        resids = Yw - Xw * betas
        squared_normalized_resids = (resids.^2.0) / (sum(resids.^2.0))
        abssnresids = abs.(squared_normalized_resids)
        medsnresids = median(squared_normalized_resids)
        a = [(1 - weights[i]) / maximum([abssnresids[i], medsnresids]) for i in 1:n]
        weights = (a.^2.0) / (sum(a.^2.0))
        estimatedvariance = n * c * sum(resids.^2) / (n - p + 1)
        push!(estimatedvariances, estimatedvariance)
        if length(estimatedvariances) > 2
            if abs(estimatedvariance - estimatedvariances[end - 1]) < epsilon || abs(estimatedvariance - estimatedvariances[end - 2]) < epsilon
                break
            end
        end
    end

    # Report 
    result = Dict()
    result["weights"] = weights
    result["betas"] = betas
    result["squared.normalized.residuals"] = squared_normalized_resids
    result["squared.normalized.robust.distances"] = squared_normalized_robust_distances
    result["residuals"] = resids
    result["basic.subset"] = sort(basicsubsetindices)
    return result
end


function bchplot(setting::RegressionSetting; alpha=0.05, maxiter=1000, epsilon=0.00001)
    result = bch(setting, alpha=alpha, maxiter=maxiter, epsilon=epsilon)
    squared_normalized_residuals = result["squared.normalized.residuals"]
    squared_normalized_robust_distances = result["squared.normalized.robust.distances"]
    n = length(squared_normalized_robust_distances)
    scplot = scatter(squared_normalized_robust_distances, 
            squared_normalized_residuals, 
            legend=false, 
            series_annotations=text.(1:n, :bottom),
            tickfont=font(10), guidefont=font(10), labelfont=font(10)
            )
    title!("Billor & Chatterjee & Hadi Plot")
    xlabel!("Squared Normalized Robust Distances")
    ylabel!("Squared Normalized Residuals")
end