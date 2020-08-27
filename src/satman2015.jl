function dominates(p1::Array, p2::Array)::Bool
    n = length(p1)
    notworse = count(i -> p1[i] < p2[i], 1:n)
    better   = count(i -> p1[i] > p2[i], 1:n)
    return (notworse == 0) && (better > 0)
end

function ndsranks(data::DataFrame)::Array{Int}
    mat = convert(Matrix, data)
    return ndsranks(mat)
end

function ndsranks(data::Matrix)::Array{Int}
    n, p = size(data)
    ranks = zeros(Int, n)
    mat = convert(Matrix, data)
    for i in 1:n
        for j in 1:n
            if i != j 
                if dominates(mat[i,:], mat[j,:])
                    ranks[i] += 1
                end
            end
        end
    end
    return ranks
end

function midlist(n::Int, p::Int)::Array{Int, 1}
    midlist = []
    if (n - p) % 2 == 0
        start = ((n - p) / 2) + 1
        stop = start + p - 1
        midlist = collect(start:stop)
    else
        start = Int(floor((n - p) / 2)) + 1
        stop = start + p  
        midlist = collect(start:stop)
    end 
    return midlist
end

function satman2015(setting::RegressionSetting)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    allindices = collect(1:n)

    ranks = ndsranks(X)
    ranks_ordering = sortperm(ranks)

    basic_center_indices = midlist(n, p)
    basic_subset_indices = ranks_ordering[basic_center_indices]

    meanvector = applyColumns(mean, X[basic_subset_indices,:])
    covmat = cov(X[basic_subset_indices,:])
    md2 = diag(mahalabonisSquaredMatrix(X, meanvector = meanvector, covmatrix = covmat))
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