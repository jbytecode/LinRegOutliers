function mahalanobisSquaredBetweenPairs(pairs::Matrix; covmatrix=nothing)
    n, p = size(pairs)
    newmat = zeros(Float64, n, n)
    if covmatrix === nothing
        covmatrix = cov(pairs)
    end
    try
        invm = inv(covmatrix)
        for i in 1:n
            @inbounds for j in i:n
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

# Description
The algorithm performs a Least Trimmed Squares (LTS) estimate and yields standardized 
residual - fitted response pairs. A single linkage clustering algorithm is performed on these
pairs. Like `smr98`, the cluster tree is cut using the Majona criterion. Subtrees with 
relatively small number of observations are declared to be outliers.

# Output
- `["outliers"]`: Vector of indices of outliers.


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
    X, y = @extractRegressionSetting setting
    return asm2000(X, y)
end


function asm2000(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)
    h = floor((n + p - 1) / 2)
    ltsreg = lts(X, y)
    
    betas = ltsreg["betas"]
    hsubset = ltsreg["hsubset"]

    predicteds = [sum(X[i,:] .* betas) for i in 1:n]
    resids = y .- predicteds
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