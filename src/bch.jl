"""

    coordinatwisemedians(datamat)

    Return vector of medians of each column in a matrix.

# Arguments
- `datamat::Array{Float64, 2}`: A matrix.

# Example
```julia-repl
julia> mat = [1.0 2.0; 3.0 4.0; 5.0 6.0]
3×2 Array{Float64,2}:
 1.0  2.0
 3.0  4.0
 5.0  6.0

julia> coordinatwisemedians(mat)
2-element Array{Float64,1}:
 3.0
 4.0
```
"""
function coordinatwisemedians(datamat::Array{Float64,2})::Array{Float64,1}
    n, p = size(datamat)
    meds = map(i -> median(datamat[:, i]), 1:p)
    return meds
end


"""

    bch(setting; alpha = 0.05, maxiter = 1000, epsilon = 0.000001)

Perform the Billor & Chatterjee & Hadi (2006) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `alpha::Float64`: Optional argument of the probability of rejecting the null hypothesis.
- `maxiter::Int`: Maximum number of iterations for calculating iterative weighted least squares estimates.
- `epsilon::Float64`: Accuracy for determining convergency.

# Description 
The algorithm initially constructs a basic subset. These basic subset is then used to 
generate initial weights for a iteratively least squares estimation. Regression coefficients obtained 
in this stage are robust regression estimates. Squared normalized distances and squared normalized 
residuals are used in `bchplot` which serves a visual way for investigation of outliers and their 
properties.


# Output
- `["betas"]`: Final estimate of regression coefficients                               
- `["squared.normalized.robust.distances"]`:  
- `["weights"]`: Final weights used in calculation of WLS estimates                             
- `["outliers"]`: Array of indices of outliers
- `["squared.normalized.residuals"]`: Array of squared normalized residuals
- `["residuals"]`: Array of regression residuals
- `["basic.subset"]`: Array of indices of basic subset.


# Examples
```julia-repl
julia> reg  = createRegressionSetting(@formula(calls ~ year), phones);
julia> Dict{Any,Any} with 7 entries:
"betas"                               => [-55.9205, 1.15572]
"squared.normalized.robust.distances" => [0.104671, 0.0865052, 0.0700692, 0.0553633, 0.0423875, 0.03…
"weights"                             => [0.00186158, 0.00952088, 0.0787321, 0.0787321, 0.0787321, 0…
"outliers"                            => [1, 14, 15, 16, 17, 18, 19, 20, 21]
"squared.normalized.residuals"        => [5.53742e-5, 2.42977e-5, 2.36066e-6, 2.77706e-6, 1.07985e-7…
"residuals"                           => [2.5348, 1.67908, 0.523367, 0.567651, 0.111936, -0.343779, …
"basic.subset"                        => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  15, 16, 17, 18, 19, 20, …
```

# References
Billor, Nedret, Samprit Chatterjee, and Ali S. Hadi. "A re-weighted least squares method 
for robust regression estimation." American journal of mathematical and management sciences 26.3-4 (2006): 229-252.
"""
function bch(setting::RegressionSetting; alpha=0.05, maxiter=1000, epsilon=0.000001)
    X, y = @extractRegressionSetting setting
    return bch(X, y, alpha=alpha, maxiter=maxiter, epsilon=epsilon)
end



function bch(Xdesign::Array{Float64,2}, y::Array{Float64,1}; alpha=0.05, maxiter=1000, epsilon=0.000001)
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
    dsquared = diag(mahalanobisSquaredMatrix(DataFrame(X), meanvector=coordmeds, covmatrix=A))
    d = sqrt.(dsquared)

    # Algorithm 2 - Step 0.b
    bestindicesofd = sortperm(d)[1:h]
    colmeansofh = map(i -> mean(X[bestindicesofd, i]), 1:p)
    covmatofh = cov(X[bestindicesofd,:])
    newdsquared = diag(mahalanobisSquaredMatrix(DataFrame(X), meanvector=colmeansofh, covmatrix=covmatofh))
    newd = sqrt.(newdsquared)

    # Algorithm 2 - Steps 1, 2, 3
    basicsubsetindices = sortperm(newd)[1:(p + 1)]
    while length(basicsubsetindices) < h
        colmeanofbasicsubset = map(i -> mean(X[basicsubsetindices, i]), 1:p)
        covmatofbasicsubset = cov(X[basicsubsetindices,:]) 
        newdsquared = diag(mahalanobisSquaredMatrix(DataFrame(X), meanvector=colmeanofbasicsubset, covmatrix=covmatofbasicsubset))
        newd = sqrt.(newdsquared)
        basicsubsetindices = sortperm(newd)[1:(length(basicsubsetindices) + 1)]
    end

    # Algorithm 2 - Steps 4
    while length(basicsubsetindices) < n
        r = length(basicsubsetindices)
        colmeanofbasicsubset = map(i -> mean(X[basicsubsetindices, i]), 1:p)
        covmatofbasicsubset = cov(X[basicsubsetindices,:]) 
        newdsquared = diag(mahalanobisSquaredMatrix(DataFrame(X), meanvector=colmeanofbasicsubset, covmatrix=covmatofbasicsubset))
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
        wols = wls(Xdesign, y, weights)
        betas = coef(wols)
        resids = residuals(wols)
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
    result["outliers"] = filter(i -> abs(resids[i]) > 2.5, 1:n)
    result["basic.subset"] = sort(basicsubsetindices)
    return result
end



"""

    bchplot(setting::RegressionSetting; alpha=0.05, maxiter=1000, epsilon=0.00001)

Perform the Billor & Chatterjee & Hadi (2006) algorithm and generates outlier plot 
for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `alpha::Float64`: Optional argument of the probability of rejecting the null hypothesis.
- `maxiter::Int`: Maximum number of iterations for calculating iterative weighted least squares estimates.
- `epsilon::Float64`: Accuracy for determining convergency.


# References
Billor, Nedret, Samprit Chatterjee, and Ali S. Hadi. "A re-weighted least squares method 
for robust regression estimation." American journal of mathematical and management sciences 26.3-4 (2006): 229-252.
"""
function bchplot(setting::RegressionSetting; alpha=0.05, maxiter=1000, epsilon=0.00001)
    X = designMatrix(setting)
    y = responseVector(setting)
    return bchplot(X, y, alpha=alpha, maxiter=maxiter, epsilon=epsilon)
end

function bchplot(Xdesign::Array{Float64,2}, y::Array{Float64,1}; alpha=0.05, maxiter=1000, epsilon=0.00001)
    result = bch(Xdesign, y, alpha=alpha, maxiter=maxiter, epsilon=epsilon)
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