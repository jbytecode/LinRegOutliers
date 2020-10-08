"""

    dffit(setting, i)

Calculate the effect of the ith observation on the linear regression fit.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `i::Int`: Index of the observation.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> dffit(reg, 1)
2.3008326745719785

julia> dffit(reg, 15)
2.7880619386124295

julia> dffit(reg, 16)
3.1116532421969794

julia> dffit(reg, 17)
4.367981450347031

julia> dffit(reg, 21)
-5.81610150322166
```

# References
Belsley, David A., Edwin Kuh, and Roy E. Welsch. Regression diagnostics: 
Identifying influential data and sources of collinearity. Vol. 571. John Wiley & Sons, 2005.
"""
function dffit(setting::RegressionSetting, i::Int)::Float64
    X = designMatrix(setting)
    y = responseVector(setting)
    return dffit(X, y, i)
end

function dffit(X::Array{Float64,2}, y::Array{Float64,1}, i::Int)::Float64
    n, _ = size(X)
    indices = [j for j in 1:n if i != j]
    olsfull = ols(X, y)
    Xsub = X[indices,:]
    ysub = y[indices]
    olsjacknife = ols(Xsub, ysub)
    return predict(olsfull, X)[i] - predict(olsjacknife, X)[i]
end



"""
    dffit(setting)

Calculate `dffit` for all observations.

# Arguments 
- `setting::RegressionSetting`: A regression setting object.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);

julia> dffit(reg)
24-element Array{Float64,1}:
   2.3008326745719785
   1.2189579001467337
   0.35535667547543426
  -0.14458523141740898
  -0.5558346324846752
  -0.8441316814464983
  -1.0329184407957257
  -1.16600692151232
  -1.2005633711667656
  -1.2549187193476428
  -1.3195581500053777
  -1.42383876236147
  -1.5917690629803474
  -1.6582086833534504
   2.7880619386124295
   3.1116532421969794
   4.367981450347031
   5.927603041427858
   8.442860517217582
  12.370243663029527
  -5.81610150322166
 -10.089153963127842
 -12.10803256546825
 -14.67006851119936
```

# References
Belsley, David A., Edwin Kuh, and Roy E. Welsch. Regression diagnostics: 
Identifying influential data and sources of collinearity. Vol. 571. John Wiley & Sons, 2005.
"""
function dffit(setting::RegressionSetting)::Array{Float64,1}
    n, _ = size(setting.data)
    result = [dffit(setting, i) for i in 1:n]
    return result    
end

function dffit(X::Array{Float64,2}, y::Array{Float64,1})::Array{Float64,1}
    n, _ = size(X)
    result = [dffit(X, y, i) for i in 1:n]
    return result    
end

"""
    hatmatrix(setting)

Calculate Hat matrix of dimensions n x n for a given regression setting with n observations.

# Arguments
- `setting::RegressionSetting`: A regression setting object.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> size(hatmatrix(reg))

(24, 24)
"""
function hatmatrix(setting::RegressionSetting)::Array{Float64,2}
    X = designMatrix(setting)
    return hatmatrix(X)
end

function hatmatrix(X::Array{Float64,2})::Array{Float64,2}
    return X * inv(X'X) * X'
end

"""
    studentizedResiduals(setting)

Calculate Studentized residuals for a given regression setting.

# Arguments:
- `setting::RegressionSetting`: A regression setting object.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);

julia> studentizedResiduals(reg)
24-element Array{Float64,1}:
  0.2398783264505892
  0.1463945666608097
  0.04934549995087145
 -0.023289236798461784
 -0.10408303320973748
 -0.18382934382804111
 -0.2609395640240455
 -0.33934473417314376
 -0.3973205657179429
 -0.46258080183149236
 -0.5261488085924144
 -0.5918396227060093
 -0.6616423337899147
 -0.6611792918262785
  1.0277190922689816
  1.0297863954540103
  1.2712201589839855
  1.4974523565936426
  1.8386296155264197
  2.316394853333409
 -0.9368354141338643
 -1.4009989983319822
 -1.4541520919831887
 -1.529459974327181
```
"""
function studentizedResiduals(setting::RegressionSetting)::Array{Float64,1}
    X = designMatrix(setting)
    y = responseVector(setting)
    return studentizedResiduals(X, y)
end

function studentizedResiduals(X::Array{Float64,2}, y::Array{Float64,1})::Array{Float64,1}
    olsreg = ols(X, y)
    n, p = size(X)
    e = residuals(olsreg)
    s = sqrt(sum(e.^2.0) / (n - p))
    hat = hatmatrix(X)
    stde = [e[i] / (s * sqrt(1.0 - hat[i, i])) for i in 1:n]
    return stde
end



"""
    adjustedResiduals(setting)

Calculate adjusted residuals for a given regression setting.

# Arguments:
- `setting::RegressionSetting`: A regression setting object.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> adjustedResiduals(reg)
24-element Array{Float64,1}:
  13.486773572526268
   8.2307993473897
   2.774371467851612
  -1.3093999279776498
  -5.851901346871404
 -10.335509559699863
 -14.670907823058053
 -19.07911256736661
 -22.338710565623828
 -26.00786250934617
 -29.58187157605512
 -33.27523207616458
 -37.19977737822219
 -37.173743587631165
  57.781855070799956
  57.898085871534626
  71.47231139524963
  84.19185329435882
 103.37399662263209
 130.23557965295348
 -52.6720662600165
 -78.76891816539992
 -81.75736547266746
 -85.9914301855088
```
"""
function adjustedResiduals(setting::RegressionSetting)::Array{Float64,1}
    X = designMatrix(setting)
    y = responseVector(setting)
    return adjustedResiduals(X, y)
end


function adjustedResiduals(X::Array{Float64,2}, y::Array{Float64,1})::Array{Float64,1}
    olsreg = ols(X, y)
    n, p = size(X)
    e = residuals(olsreg)
    hat = hatmatrix(X)
    stde = [e[i] / (sqrt(1 - hat[i, i])) for i in 1:n]
    return stde
end


"""

    jacknifedS(setting, k)

Estimate standard error of regression with the kth observation is dropped.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `k::Int`: Index of the omitted observation. 

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> jacknifedS(reg, 2)
57.518441664761035

julia> jacknifedS(reg, 15)
56.14810222161477
```
"""
function jacknifedS(setting::RegressionSetting, k::Int)::Float64
    X = designMatrix(setting)
    y = responseVector(setting)
    return jacknifedS(X, y, k)
end

function jacknifedS(X::Array{Float64,2}, y::Array{Float64,1}, k::Int)::Float64
    n, p = size(X)
    indices = [i for i in 1:n if i != k]
    Xsub = X[indices,:]
    ysub = y[indices]
    olsreg = ols(Xsub, ysub)
    e = residuals(olsreg)
    s = sqrt(sum(e.^2.0) / (n - p - 1))
    return s
end


"""

    cooks(setting)

Calculate Cook distances for all observations in a regression setting.

# Arguments 
- `setting::RegressionSetting`: A regression setting object.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);

julia> cooks(reg)
24-element Array{Float64,1}:
 0.005344774190779822
 0.0017088194691033689
 0.00016624914057962608
 3.1644452583114795e-5
 0.0005395058666404081
 0.0014375008774859539
 0.0024828140956511258
 0.0036279720445167277
 0.004357605989540906
 0.005288503758364767
 0.006313578057565415
 0.0076561205696857254
 0.009568574875389256
 0.009970039008782357
 0.02610396373381051
 0.029272523880917646
 0.05091236198400663
 0.08176555044049343
 0.14380266904640235
 0.26721539425047447
 0.051205153558783356
 0.13401084683481085
 0.16860324592350226
 0.2172819114905912
```

# References
Cook, R. Dennis. "Detection of influential observation in linear regression." 
Technometrics 19.1 (1977): 15-18.
"""
function cooks(setting::RegressionSetting)::Array{Float64,1}
    X = designMatrix(setting)
    y = responseVector(setting)
    return cooks(X, y)
end

function cooks(X::Array{Float64,2}, y::Array{Float64,1})::Array{Float64,1}
    n, p = size(X)
    olsreg = ols(X, y)
    res = residuals(olsreg)
    hat = hatmatrix(X)
    s2 = sum(res .* res) / (n - p)
    d = zeros(Float64, n)
    for i in 1:n
        d[i] = ((res[i]^2.0) / (p * s2)) * (hat[i, i] / (1 - hat[i, i])^2.0)
    end
    return d
end


"""
    mahalanobisSquaredMatrix(data::DataFrame; meanvector=nothing, covmatrix=nothing)

Calculate Mahalanobis distances.

# Arguments
- `data::DataFrame`: A DataFrame object of the multivariate data.
- `meanvector::Array{Float64, 1}`: Optional mean vector of variables.
- `covmatrix::Array{Float64, 2}`: Optional covariance matrix of data.

# References
Mahalanobis, Prasanta Chandra. "On the generalized distance in statistics." 
National Institute of Science of India, 1936.
"""
function mahalanobisSquaredMatrix(data::DataFrame; meanvector=nothing, covmatrix=nothing)::Union{Nothing,Array{Float64,2}}
    datamat = convert(Matrix, data)
    return mahalanobisSquaredMatrix(datamat, meanvector=meanvector, covmatrix=covmatrix)
end


function mahalanobisSquaredMatrix(datamat::Matrix; meanvector=nothing, covmatrix=nothing)::Union{Nothing,Array{Float64,2}}
    if meanvector === nothing
        meanvector = applyColumns(mean, datamat)
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
        return nothing
    end
end

