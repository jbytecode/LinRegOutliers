function distances(resids::Array{Float64, 1}, fitteds::Array{Float64})::Array{Float64, 2}
    n = length(resids)
    d = zeros(Float64, n, n)
    for i in 1:n
        for j in i:n
            if i != j 
                p1 = [resids[i], fitteds[i]]
                p2 = [resids[j], fitteds[j]]
                d[i, j] = sqrt(sum((p1 .- p2).^ 2.0))
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

function smr98(setting::RegressionSetting)
    design = designMatrix(setting)
    ols = lm(setting.formula, setting.data)
    stdres = standardize(ZScoreTransform, residuals(ols), dims = 1)
    stdfit = standardize(ZScoreTransform, predict(ols), dims = 1)
    n, p = size(design)
    d = distances(stdres, stdfit)
    h = floor((n + p - 1) / 2)
    hcl = hclust(d, linkage = :single)
    majonacrit = majona(hcl)
    clustermappings = cutree(hcl, h = majonacrit)
    uniquemappings = unique(clustermappings)
    for clustid in uniquemappings
        cnt = count(x -> x == clustid, clustermappings)
        if cnt >= h 
            return filter(i -> clustermappings[i] != clustid, 1:n)
        end
    end
    return []
end