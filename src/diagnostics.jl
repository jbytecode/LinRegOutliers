function dffit(setting::RegressionSetting, i::Int)::Float64
    n = size(setting.data)[1]
    indices = [j for j in 1:n if i != j]
    olsfull = lm(setting.formula, setting.data) 
    olsjacknife = lm(setting.formula, setting.data[indices,:])
    return predict(olsfull, designMatrix(setting))[i] - predict(olsjacknife, designMatrix(setting))[i]
end

function dffit(setting::RegressionSetting)::Array{Float64,1}
    n = size(setting.data)[1]
    result = [dffit(setting, i) for i in 1:n]
    return result    
end

function hatmatrix(setting::RegressionSetting)::Array{Float64,2}
    X = designMatrix(setting)
    return X * inv(X'X) * X'
end

function studentizedResiduals(setting::RegressionSetting)::Array{Float64,1}
    ols = lm(setting.formula, setting.data)
    n, p = size(designMatrix(setting))
    e = residuals(ols)
    s = sqrt(sum(e.^2.0) / (n - p))
    hat = hatmatrix(setting)
    stde = [e[i] / (s * sqrt(1 - hat[i, i])) for i in 1:n]
    return stde
end

function adjustedResiduals(setting::RegressionSetting)::Array{Float64,1}
    ols = lm(setting.formula, setting.data)
    n, p = size(designMatrix(setting))
    e = residuals(ols)
    hat = hatmatrix(reg)
    stde = [e[i] / (sqrt(1 - hat[i, i])) for i in 1:n]
    return stde
end

function jacknifedS(setting::RegressionSetting, k::Int)::Float64
    n, p = size(designMatrix(setting))
    indices = [i for i in 1:n if i != k]
    ols = lm(setting.formula, setting.data[indices,:])
    e = residuals(ols)
    s = sqrt(sum(e.^2.0) / (n - p - 1))
    return s
end

function cooks(setting::RegressionSetting)::Array{Float64,1}
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    ols = lm(setting.formula, setting.data)
    res = residuals(ols)
    hat = hatmatrix(setting)
    s2 = sum(res .* res) / (n - p)
    d = zeros(Float64, n)
    for i in 1:n
        d[i] = ((res[i]^2.0) / (p * s2)) * (hat[i, i] / (1 - hat[i, i])^2.0)
    end
    return d
end

function mahalabonisSquaredMatrix(data::DataFrame; meanvector=nothing, covmatrix=nothing)::Array{Float64,2}
    datamat = convert(Matrix, data)
    return mahalabonisSquaredMatrix(datamat, meanvector = meanvector, covmatrix = covmatrix)
end


function mahalabonisSquaredMatrix(datamat::Matrix; meanvector=nothing, covmatrix=nothing)::Array{Float64,2}
    if meanvector === nothing
        meanvector = applyColumns(mean, data)
    end
    if covmatrix === nothing
        covmatrix = cov(datamat)
    end
    try
        invm = inv(covmatrix)
        MD2 = (datamat .- meanvector') * invm * (datamat .- meanvector')'
        return MD2
    catch e
        if det(covmatrix) == 0
            @warn "singular covariance matrix, mahalanobis distances can not be calculated"
        end
        n = size(datamat)[1]
        return zeros(Float64, (n, n))
    end
end