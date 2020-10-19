function imon2005(setting::RegressionSetting)
    X = designMatrix(setting)
    y = responseVector(setting)
    return imon2005(X, y)
end

function imon2005(X::Array{Float64,2}, y::Array{Float64,1})

    function SigmaWithoutIndex(X, y, R, i)
        n, p = size(X)
        Ri = filter(x -> x != i, R)
        XRi = X[Ri,:]
        yRi = y[Ri]
        reg = ols(XRi, yRi)
        betas = coef(reg)
        res = y .- X * betas 
        return sqrt(sum(res.^2.0) / (n - p))
    end

    n, p = size(X)
    allindex = collect(1:n)
    ltsreg = lts(X, y)
    R::Array{Int,1} = ltsreg["hsubset"]
    XR = X[R,:]
    yR = y[R]
    XRXR = transpose(XR) * XR
    invXRXR = inv(XRXR)
    wiiR = [X[i,:]' * invXRXR * X[i,:] for i in allindex]
    wiiRAsterix = zeros(Float64, n)
    for i in allindex
        if i in R
            wiiRAsterix[i] = wiiR[i] / (1.0 - wiiR[i])
        else
            wiiRAsterix[i] = wiiR[i] / (1.0 + wiiR[i])
        end
    end
    RegR = ols(XR, yR)
    BetaHatR = coef(RegR)
    resR = y - X * BetaHatR
    sigmaR = sum(resR.^2.0) / (n - p)
    tAsterix = zeros(Float64, n)
    for i in allindex
        if i in R
            tAsterix[i] = resR[i] / (SigmaWithoutIndex(X, y, R, i) * sqrt(1.0 - wiiR[i]))
        else
            tAsterix[i] = resR[i] / (SigmaWithoutIndex(X, y, R, i) * sqrt(1.0 + wiiR[i]))
        end
    end

    GDFFITS = zeros(Float64, n)
    for i in allindex
        GDFFITS[i] = sqrt(wiiRAsterix[i]) * tAsterix[i]
    end
    crit = 3.0 * sqrt(p / length(R))
    outlyingindex = filter(i -> abs(GDFFITS[i]) >= crit, allindex)
    result::Dict{String,Any} = Dict()
    result["crit"] = crit
    result["gdffits"] = GDFFITS
    result["outliers"] = outlyingindex
    return result
end