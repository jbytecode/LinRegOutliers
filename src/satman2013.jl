"""

    satman2013(setting)

Perform Satman (2013) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description
The algorithm constructs a fast and robust covariance matrix to calculate robust mahalanobis
distances. These distances are then used to construct weights for later use in a weighted least 
squares estimation. In the last stage, C-steps are iterated on the basic subset found in previous
stages. 

# Output
- `["outliers"]`: Array of indices of outliers.

# Examples
```julia-repl
julia> eg0001 = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk);
julia> satman2013(reg0001)
Dict{Any,Any} with 1 entry:
  "outliers" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 47]

```

# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.
"""
function satman2013(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return satman2013(X, y)
end


function satman2013(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    X0 = X
    p0 = p
    if X0[:, 1] == ones(n)
        X0 = X[:, 2:end]
        p0 = p - 1
    end

    allindices = collect(1:n)

    covmat = zeros(p0, p0)

    for i in 1:p0
        for j in 1:p0
            if i == j 
                @inbounds covmat[i, j] = median(abs.(X0[:, i] .- median(X0[:, i])))
            else
                @inbounds covmat[i, j] = median((X0[:, i] .- median(X0[:, i])) .* (X0[:, j] .- median(X0[:, j])))
            end
        end
    end

    medians = applyColumns(median, X0)
    mhs = mahalanobisSquaredMatrix(X0, meanvector=medians, covmatrix=covmat)
    if mhs isa Nothing
        md2 = zeros(Float64, n)
    else
        md2 = diag(mhs)
    end
    md = sqrt.(md2)

    sorted_indices = sortperm(md)
    best_h_indices = sorted_indices[1:h]

    crit, bestset = iterateCSteps(X, y, best_h_indices, h)
    
    olsreg = ols(X[bestset, :], y[bestset])
    betas = coef(olsreg)
    resids = y .- (X * betas)
    med_res = median(resids)
    standardized_resids = (resids .- med_res) / median(abs.(resids .- med_res))

    outlierset = filter(i -> abs(standardized_resids[i]) > 2.5, allindices)
    
    result = Dict()
    result["outliers"] = outlierset

    return result
end
