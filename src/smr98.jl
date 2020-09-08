function distances(resids::Array{Float64,1}, fitteds::Array{Float64})::Array{Float64,2}
    n = length(resids)
    d = zeros(Float64, n, n)
    for i in 1:n
        for j in i:n
            if i != j 
                p1 = [resids[i], fitteds[i]]
                p2 = [resids[j], fitteds[j]]
                d[i, j] = sqrt(sum((p1 .- p2).^2.0))
                d[j, i] = d[i, j]
            end
        end
    end
    return d
end

function majona(cluster::Hclust)::Float64
    heights = cluster.heights
    return mean(heights) + 1.25 * std(heights)
end

"""

    smr98(setting)

Perform the Sebert, Monthomery and Rollier (1998) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> smr98(reg0001)
10-element Array{Int64,1}:
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
```

# References
Sebert, David M., Douglas C. Montgomery, and Dwayne A. Rollier. "A clustering algorithm for 
identifying multiple outliers in linear regression." Computational statistics & data analysis 
27.4 (1998): 461-484.
"""
function smr98(setting::RegressionSetting)
    design = designMatrix(setting)
    ols = lm(setting.formula, setting.data)
    stdres = standardize(ZScoreTransform, residuals(ols), dims=1)
    stdfit = standardize(ZScoreTransform, predict(ols), dims=1)
    n, p = size(design)
    d = distances(stdres, stdfit)
    h = floor((n + p - 1) / 2)
    hcl = hclust(d, linkage=:single)
    majonacrit = majona(hcl)
    clustermappings = cutree(hcl, h=majonacrit)
    uniquemappings = unique(clustermappings)
    for clustid in uniquemappings
        cnt = count(x -> x == clustid, clustermappings)
        if cnt >= h 
            return filter(i -> clustermappings[i] != clustid, 1:n)
        end
    end
    return []
end

function mahalanobisSquaredBetweenPairs(pairs::Matrix; covmatrix=nothing)
    n, p = size(pairs)
    newmat = zeros(Float64, n, n)
    if covmatrix === nothing
        covmatrix = cov(pairs)
    end
    try
        invm = inv(covmatrix)
        for i in 1:n
            for j in i:n
                newmat[i, j] = ((pairs[i,:] .- pairs[j,:])' * invm * (pairs[i,:] .- pairs[j,:]))
                newmat[j, i] = newmat[i,j]
            end
        end
        return newmat
    catch e
        @warn e
        if det(covmatrix) == 0
            @warn "singular covariance matrix, mahalanobis distances can not be calculated"
        end
        return zeros(Float64, (n, n))
    end
end


"""

    asm2000(setting)

Perform the Setan, Halim and Mohd (2000) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> asm2000(reg0001)
Dict{Any,Any} with 1 entry:
  "outliers" => [15, 16, 17, 18, 19, 20]
```

# References
Setan, Halim, and Mohd Nor Mohamad. "Identifying multiple outliers in 
linear regression: Robust fit and clustering approach." (2000).
"""
function asm2000(setting::RegressionSetting)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = floor((n + p - 1) / 2)
    ltsreg = lts(setting)
    
    betas = ltsreg["betas"]
    hsubset = ltsreg["hsubset"]

    predicteds = [sum(X[i,:] .* betas) for i in 1:n]
    resids = Y .- predicteds
    stdres = standardize(ZScoreTransform, resids, dims=1)
    stdfit = standardize(ZScoreTransform, predicteds, dims=1)
    pairs = hcat(stdfit, stdres)

    pairs = hcat(resids, predicteds)

    covmatrix = cov(pairs[hsubset, :])
    mahdist = mahalanobisSquaredBetweenPairs(pairs, covmatrix=covmatrix)

    outlierset = Array{Int,1}()

    hcl = hclust(mahdist, linkage=:single)
    majonacrit = majona(hcl)
    clustermappings = cutree(hcl, h=majonacrit)
    uniquemappings = unique(clustermappings)
    for clustid in uniquemappings
        cnt = count(x -> x == clustid, clustermappings)
        if cnt >= h 
            outlierset = filter(i -> clustermappings[i] != clustid, 1:n)
        end
    end
    
    result = Dict()
    result["outliers"] = outlierset
    return result
end
