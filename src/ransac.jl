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

 # Examples
```julia-repl
julia> df = DataFrame(y=[0,1,2,3,3,4,10], x=[0,1,2,2,3,4,2])
julia> reg = createRegressionSetting(@formula(y ~ x), df)
julia> ransac(reg, t=0.8, w=0.85)
1-element Array{Int64,1}:
 7
```

# References
Martin A. Fischler & Robert C. Bolles (June 1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography"
Comm. ACM. 24 (6): 381–395.
"""
function ransac(setting::RegressionSetting; t::Float64, w::Float64=0.5, m::Int=0, k::Int=0, d::Int=0, confidence::Float64=0.99)

    X = designMatrix(setting)
    Y = responseVector(setting)
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

        sampled_indices = sample(1:n, m, replace=false)

        ols = lm(setting.formula, setting.data[sampled_indices, :])
        betas = coef(ols)

        e = abs.(Y - X * betas) ./ norm([1; betas[2:end]], 2)

        iteration_inlier_indices = filter(i -> e[i] < t, 1:n)
        inliers_count = length(iteration_inlier_indices)

        if inliers_count >= d
            ols = lm(setting.formula, setting.data[iteration_inlier_indices, :])
            error_vector = residuals(ols)
            iteration_error = norm(error_vector, 2) / length(error_vector)

            if iteration_error < minimum_error
                maximum_inlier_indices = iteration_inlier_indices
                maximum_count = inliers_count
                minimum_error = iteration_error
            end
        end
    end

    return setdiff(1:n, maximum_inlier_indices)
end

