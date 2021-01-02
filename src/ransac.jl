"""

    ransac(setting; t, w=0.5, m=0, k=0, d=0, confidence=0.99)

Run the RANSAC (1981) algorithm for the given regression setting

# Arguments
 - `setting::RegressionSetting`: RegressionSetting object with a formula and a dataset.
 - `t::Float64`: The threshold distance of a sample point to the regression hyperplane to determine if it fits the model well.
 - `w::Float64`: The probability of a sample point being inlier, default=0.5.
 - `m::Int`: The number of points to sample to estimate the model parameter for each iteration. If set to 0, defaults to picking p points which is the minimum required.
 - `k::Int`: The number of iterations to run. If set to 0, is calculated according to the formula given in the paper based on outlier probability and the sample set size.
 - `d::Int`: The number of close data points required to accept the model. Defaults to number of data points multiplied by inlier ratio.
 - `confidence::Float64`: Required to determine the number of optimum iterations if k is not specified.

 # Output
- `["outliers"]`: Array of indices of outliers.


 # Examples
```julia-repl
julia> df = DataFrame(y=[0,1,2,3,3,4,10], x=[0,1,2,2,3,4,2])
julia> reg = createRegressionSetting(@formula(y ~ x), df)
julia> ransac(reg, t=0.8, w=0.85)
Dict{String,Array{Int64,1}} with 1 entry:
  "outliers" => [7]
```

# References
Martin A. Fischler & Robert C. Bolles (June 1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography"
Comm. ACM. 24 (6): 381–395.
"""
function ransac(setting::RegressionSetting; t::Float64, w::Float64=0.5, m::Int=0, k::Int=0, d::Int=0, confidence::Float64=0.99)
    X, y = @extractRegressionSetting setting
    return ransac(X, y, t=t, w=w, m=m, k=k, d=d, confidence=confidence)
end

function ransac(X::Array{Float64,2}, y::Array{Float64,1}; t::Float64, w::Float64=0.5, m::Int=0, k::Int=0, d::Int=0, confidence::Float64=0.99)

    n, p = size(X)
    if d == 0
        d = Int(floor(n * w))
    end

    if k == 0
        k = Int(ceil(log(1 - confidence) / log(1 - w^d)))
    end

    if m == 0
        m = p
    end

    iteration_inlier_indices = zeros(Int, n)

    maximum_count = d - 1
    maximum_inlier_indices = zeros(Int, n)
    minimum_error = Inf

    for iteration in 1:k
        inliers_count = 0
        sampled_indices = sample(1:n, m, replace=false)
        ols_sampled_points = ols(X[sampled_indices, :], y[sampled_indices])
        betas = coef(ols_sampled_points)

        e = abs.(y - X * betas) ./ norm([1; betas[2:end]], 2)

        iteration_inlier_indices = filter(i -> e[i] < t, 1:n)
        inliers_count = length(iteration_inlier_indices)

        if inliers_count >= d
            ols_inliers = ols(X[iteration_inlier_indices, :], y[iteration_inlier_indices])
            error_vector = residuals(ols_inliers)
            iteration_error = norm(error_vector, 2) / length(error_vector)

            if iteration_error < minimum_error
                maximum_inlier_indices = iteration_inlier_indices
                maximum_count = inliers_count
                minimum_error = iteration_error
            end
        end
    end

    result = Dict(
        "outliers" => setdiff(1:n, maximum_inlier_indices)
    )
    return result
end

