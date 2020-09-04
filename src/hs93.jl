function hs93initialset(setting::RegressionSetting)::Array{Int,1}
    n, p = size(designMatrix(setting))
    s = p + 1
    dfs = abs.(dffit(setting))
    sortedindices = sortperm(dfs)
    basicsetindices = sortedindices[1:s]
    return basicsetindices
end

function hs93basicsubset(setting::RegressionSetting, initialindices::Array{Int,1})::Array{Int,1}
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    h = floor((n + p - 1) / 2)
    s = length(initialindices)
    indices = initialindices
    for i in range(s + 1, stop=h)
        partialdata = setting.data[indices, :]
        ols = lm(setting.formula, partialdata)
        betas = coef(ols)
        d = zeros(Float64, n)
        XM = X[indices,:]
        for j in 1:n
            xxxx = X[j,:]' * inv(XM'XM) * X[j,:]
            if j in indices
                d[j] = abs.(Y[j] - sum(X[j,:] .* betas)) / sqrt(1 - xxxx)
            else
                d[j] = abs.(Y[j] - sum(X[j,:] .* betas)) / sqrt(1 + xxxx)
            end
        end
        orderingd = sortperm(abs.(d))
        indices = orderingd[1:Int(i)]
    end
    return indices
end


function hs93(setting::RegressionSetting; alpha=0.05, basicsubsetindices=nothing)
    if basicsubsetindices === nothing
        initialsetindices = hs93initialset(setting)
        basicsubsetindices = hs93basicsubset(setting, initialsetindices)
    end
    X = designMatrix(setting)
    Y = responseVector(setting)
    indices = basicsubsetindices
    n, p = size(X)
    s = length(indices)
    while s < n
        partialdata = setting.data[indices, :]
        ols = lm(setting.formula, partialdata)
        betas = coef(ols)
        resids = residuals(ols)
        sigma = sqrt(sum(resids.^2.0) / (length(resids) - p))
        d = zeros(Float64, n)
        XM = X[indices,:]
        iXmXm = inv(XM'XM)
        for j in 1:n
            xMMx = X[j,:]' * iXmXm * X[j,:]
            if j in indices
                d[j] = (Y[j] - sum(X[j,:] .* betas)) / (sigma * sqrt(1.0 - xMMx))
            else
                d[j] = (Y[j] - sum(X[j,:] .* betas)) / (sigma * sqrt(1.0 + xMMx))
            end
        end
        orderingd = sortperm(abs.(d))
        tdist = TDist(s - p)
        tcalc = quantile(tdist,  alpha / (2 * (s + 1)))
        if abs(d[orderingd][s + 1]) > abs(tcalc)
            result = Dict()
            result["d"] = d
            result["t"] = tcalc
            result["outliers"] = filter(x -> abs(d[x]) > abs(tcalc), 1:n)
            return result
        end
        s += 1
        indices = orderingd[1:s]
    end
    return []
end
