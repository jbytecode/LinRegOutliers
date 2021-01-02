"""
    imon2005(setting)

Perform the Imon 2005 algorithm for a given regression setting.

# Arguments 
- `setting::RegressionSetting`: A regression setting.

# Description
The algorithm estimates the GDFFITS diagnostic, which is an extension of well-known regression 
diagnostic DFFITS. Unlikely, GDFFITS is used for detecting multiple outliers whereas the original
one was used for detecting single outliers. 

# Output
- `["crit"]`: The critical value used
- `["gdffits"]`: Array of GDFFITS diagnostic calculated for observations
- `["outliers"]`: Array of indices of outliers.

# Notes
The implementation uses LTS rather than LMS as suggested in the paper. 

# References
A. H. M. Rahmatullah Imon (2005) Identifying multiple influential observations in linear regression, 
Journal of Applied Statistics, 32:9, 929-946, DOI: 10.1080/02664760500163599
 """
function imon2005(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
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
    @inbounds for i in allindex
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
    @inbounds for i in allindex
        if i in R
            tAsterix[i] = resR[i] / (SigmaWithoutIndex(X, y, R, i) * sqrt(1.0 - wiiR[i]))
        else
            tAsterix[i] = resR[i] / (SigmaWithoutIndex(X, y, R, i) * sqrt(1.0 + wiiR[i]))
        end
    end

    GDFFITS = zeros(Float64, n)
    @inbounds for i in allindex
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