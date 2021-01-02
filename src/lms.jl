"""
    lms(setting; iters = nothing, crit = 2.5)

Perform Least Median of Squares regression estimator with random sampling.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `iters::Int`: Number of random samples.
- `crit::Float64`: Critical value for standardized residuals. 


# Description 
LMS (Least Median of Squares) estimator is highly robust with 50% breakdown property. The algorithm
searches for regression coefficients which minimize (h)th ordered squared residual where h is Int(floor((n + 1.0) / 2.0))


# Output
- `["stdres"]`: Array of standardized residuals
- `["S"]`: Standard error of regression
- `["outliers"]`: Array of indices of outliers
- `["objective"]`: LMS objective value
- `["coef"]`: Estimated regression coefficients
- `["crit"]`: Threshold value.



# Examples 
```julia-repl 
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);

julia> lms(reg)
Dict{Any,Any} with 6 entries:
  "stdres"    => [2.28328, 1.55551, 0.573308, 0.608843, 0.220321, -0.168202, -0.471913, -0.860435, -0.31603, -0.110871  …  85.7265, 88.9849, 103.269, 116.705, 135.229, 159.69,…
  "S"         => 1.17908
  "outliers"  => [14, 15, 16, 17, 18, 19, 20, 21]
  "objective" => 0.515348
  "coef"      => [-56.1972, 1.1581]
  "crit"      => 2.5
```

# References
Rousseeuw, Peter J. "Least median of squares regression." Journal of the American 
statistical association 79.388 (1984): 871-880.
"""
function lms(setting::RegressionSetting; iters=nothing, crit=2.5)
    X, y = @extractRegressionSetting setting
    return lms(X, y, iters=iters, crit=crit)
end


function lms(X::Array{Float64,2}, y::Array{Float64,1}; iters=nothing, crit=2.5)
    n, p = size(X)
    h = Int(floor((n + 1.0) / 2.0))
    if iters === nothing
        iters = minimum([500 * p, 3000])
    end
    bestobjective = Inf
    bestparamaters = []
    bestres = []
    indices = collect(1:n)
    kindices = collect(p:n)
    for iter in 1:iters 
        try 
            k = rand(kindices, 1)[1]
            sampledindices = sample(indices, k, replace=false)
            olsreg = ols(X[sampledindices,:], y[sampledindices])
            betas = coef(olsreg)
            origres = y .- X * betas
            res = sort(origres.^2.0)
            m2 = res[h]
            if m2 < bestobjective
                bestparamaters = betas
                bestobjective = m2
                bestres = origres
            end
        catch e
            @warn e
        end
    end
    s = 1.4826 * sqrt((1.0 + (5.0 / (n - p))) * bestobjective)
    standardizedres = bestres / s
    d = Dict()
    d["coef"] = bestparamaters
    d["objective"] = bestobjective
    d["S"] = s
    d["stdres"] = standardizedres
    d["crit"] = crit
    d["outliers"] = filter(i -> abs(standardizedres[i]) > crit, 1:n)
    return d
end


