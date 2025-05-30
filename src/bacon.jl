module Bacon


export bacon



import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..OrdinaryLeastSquares: ols, predict, residuals, coef
import ..Diagnostics: mahalanobisSquaredMatrix
import LinearAlgebra: diag, norm, rank
import Distributions: median, cov, mean, quantile
import Distributions: Chisq, TDist

"""
        initial_basic_subset_multivariate_data(X, m, method="mahalanobis")
This function returns the m-subset according to algorithm #1 for multivariate data.
Two methods V1 and V2 are defined in the paper which use Mahalanobis distance or distance from the coordinate-wise median. The m subset returned is guaranteed to be full rank.

# Arguments
 - `X`: The multivariate matrix where each row is a data point
 - `m`: The number of points to include in the initial subset
 - `method`: The distance method to use for selecting the points for initial subset
"""
function initial_basic_subset_multivariate_data(
    X::AbstractMatrix{Float64},
    m::Int;
    method::String = "mahalanobis",
)
    n, _ = size(X)
    if method == "mahalanobis"
        msm = mahalanobisSquaredMatrix(X)
        
        if isnothing(msm)
            throw(ErrorException("Mahalanobis distances are not calculated"))
        end

        distances = sqrt.(diag(msm))
    elseif method == "median"
        median_vector = applyColumns(median, X)
        distances = [norm(X[i, :] - median_vector, 2) for i = 1:n]
    else
        return
    end

    return select_subset(X, m, distances)
end

"""
        select_subset(X, m, distances)
This function returns the list of indices which have the least distances as given in the distances array.
It also guarantees that at least m indices are returned and that the selected indices have the full rank.

# Arguments
 - `X`: The multivariate matrix where each row is a data point.
 - `m`: The minimum number of points to include in the subset indices.
 - `distances`: The distances vector used for selecting minimum distance indices.
"""
function select_subset(X::AbstractMatrix{Float64}, m::Int, distances::Vector{Float64})
    rank_x = rank(X)
    sorted_distances = sortperm(distances)
    subset = sorted_distances[1:m]

    # Test for full rank of the covariance matrix and
    # increase the m until resulting covariance matrix is full rank
    cov_matrix = X[subset, :]'X[subset, :]
    rank_cov_matrix = rank(cov_matrix)
    computed_m = m
    iter = 0
    while (rank_cov_matrix < rank_x)
        computed_m = computed_m + 1
        subset = sorted_distances[1:computed_m]
        cov_matrix = X[subset, :]'X[subset, :]
        rank_cov_matrix = rank(cov_matrix)
        iter += 1
        if iter > m
            break
        end
    end
    return subset
end

"""
        bacon_multivariate_outlier_detection(X, m, method, alpha)
This function performs the outlier detection for multivariate data according to algorithm #3. This is used by algorithm #4 to compute the initial subset.

# Arguments
 - `X`: The multivariate data matrix.
 - `m`: The minimum number of points to include in the initial subset
 - `method`: The distance method to use for selecting the points for initial subset
 - `alpha`: The quantile used for cutoff
"""
function bacon_multivariate_outlier_detection(
    X::AbstractMatrix{Float64},
    m::Int;
    method::String = "mahalanobis",
    alpha::Float64 = 0.025,
)
    n, p = size(X)
    chisquared = Chisq(p)
    chisqcrit = quantile(chisquared, 1.0 - (alpha / n))
    chi = sqrt(chisqcrit)
    h = Int(floor((n + p + 1) / 2))
    c_np = 1 + (p + 1) / (n - p) + 1 / (n - h - p)
    initial_basic_subset = initial_basic_subset_multivariate_data(X, m, method = method)
    r_prev = 0
    r = length(initial_basic_subset)
    subset = initial_basic_subset
    distances = zeros(Float64, n)

    # iterate until the size of the subset no longer changes
    iter = 0
    while (r_prev != r)
        mean_basic_subset = mean(X[subset], dims = 1)
        cov_basic_subset = X[subset]'X[subset]

        msm = mahalanobisSquaredMatrix(X, meanvector = mean_basic_subset, covmatrix = cov_basic_subset)

        if isnothing(msm)
            throw(ErrorException("Mahalanobis distances are not calculated"))
        end

        distances = sqrt.(diag(msm))
        c_hr = (h - r) / (h + r)
        c_hr = c_hr < 0 ? 0 : c_hr
        c_npr = c_hr + c_np
        cutoff = c_npr * chi
        subset = filter(x -> distances[x] < cutoff, 1:n)

        # update r
        r_prev = r
        r = length(subset)

        iter += 1
        if iter > n
            break
        end
    end
    d = Dict{String, Any}()
    d["outliers"] = setdiff(1:n, subset)
    d["distances"] = distances
    return d
end

"""
        compute_t_distance(X, y, subset)
This function computes the t distance for each point and returns the distance vector.

# Arguments:
 - `X`: The multivariate data matrix.
 - `y`: The output vector
 - `subset`: The vector which denotes the points inside the subset, used to scale the residuals accordingly.
"""
function compute_t_distance(X::AbstractMatrix{Float64}, y::Vector{Float64}, subset::Vector{Int64})
    
    n, p = size(X)

    t = zeros(Float64, n)

    least_squares_fit = ols(X[subset, :], y[subset])

    betas = coef(least_squares_fit)

    err = residuals(least_squares_fit)

    sigma = sqrt((err'err) / (n - p))

    covmatrix_inv = inv(X[subset, :]'X[subset, :])

    for i = 1:n
        scale_factor = (X[i, :]') * (covmatrix_inv * X[i, :])
        residual = (y[i] - X[i, :]' * betas)
        if i in subset
            t[i] = residual / (sigma * sqrt(1 - scale_factor))
        else
            t[i] = residual / (sigma * sqrt(1 + scale_factor))
        end
    end

    return abs.(t)
end

"""
        bacon_regression_initial_subset(X, y, m, method, alpha)
This function computes the initial subset having at least m elements which are likely to be free of outliers used for the BACON algorithm.

# Arguments:
 - `X`: The multivariate data matrix.
 - `y`: The response vector.
 - `m`: The minimum number of points to include in the initial subset
 - `method`: The distance method to use for selecting the points for initial subset
 - `alpha`: The quantile used for cutoff
"""
function bacon_regression_initial_subset(
    X::AbstractMatrix{Float64},
    y::Vector{Float64},
    m::Int;
    method::String = "mahalanobis",
    alpha = 0.025,
)
    n, p = size(X)

    # remove the constant column and apply bacon_multivariate algorithm
    distances = bacon_multivariate_outlier_detection(
        X[:, 2:end],
        m,
        method = method,
        alpha = alpha,
    )["distances"]

    initial_subset = select_subset(X, m, distances)

    t = compute_t_distance(X, y, initial_subset)
    basic_subset = select_subset(X, p + 1, t)

    r = length(basic_subset)
    iter = 0
    while (r < m)
        t = compute_t_distance(X, y, basic_subset)
        basic_subset = select_subset(X, r + 1, t)
        r = length(basic_subset)
        iter += 1
        if iter > n
            break
        end
    end

    return basic_subset
end



"""
        bacon(setting, m, method, alpha)
Run the BACON algorithm to detect outliers on regression data.

# Arguments:
 - `setting`: RegressionSetting object with a formula and a dataset.
 - `m`: The number of elements to be included in the initial subset.
 - `method`: The distance method to use for selecting the points for initial subset
 - `alpha`: The quantile used for cutoff

# Description 
BACON (Blocked Adaptive Computationally efficient Outlier Nominators) algorithm, defined in the citation below,
has many versions, e.g BACON for multivariate data, BACON for regression etc. Since the design matrix of a
regression model is multivariate data, BACON for multivariate data is performed in early stages of the algorithm.
After selecting a clean subset of observations, then a forward search is applied. Observations with high
studendized residuals are reported as outliers.

# Output 
- `["outliers"]`: Array of indices of outliers.
- `["betas"]`: Array of estimated coefficients.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
julia> bacon(reg, m=12)
Dict{String, Vector} with 2 entries:
  "betas"    => [-37.6525, 0.797686, 0.57734, -0.0670602]
  "outliers" => [1, 3, 4, 21]
```
# References
Billor, Nedret, Ali S. Hadi, and Paul F. Velleman. "BACON: blocked adaptive computationally efficient outlier nominators."
Computational statistics & data analysis 34.3 (2000): 279-298.
"""
function bacon(
    X::AbstractMatrix{Float64},
    y::Vector{Float64};
    m::Int,
    method::String = "mahalanobis",
    alpha = 0.025,
)::Dict

    n, p = size(X)

    subset = bacon_regression_initial_subset(X, y, m, method = method, alpha = alpha)

    r_prev = 0

    r = length(subset)

    iter = 0

    while (r != r_prev)
        t = compute_t_distance(X, y, subset)
        tdist = TDist(r - p)
        cutoff = quantile(tdist, 1 - alpha / (2 * (r + 1)))
        subset = filter(x -> t[x] < cutoff, 1:n)
        r_prev = r
        r = length(subset)
        iter += 1
        if iter > n
            break
        end
    end

    outlierindices = setdiff(1:n, subset)
    inlierindices = subset

    cleanols = ols(X[inlierindices, :], y[inlierindices])

    cleanbetas = coef(cleanols)

    result = Dict("outliers" => outlierindices, "betas" => cleanbetas)
    
    return result
end

function bacon(
    setting::RegressionSetting;
    m::Int,
    method::String = "mahalanobis",
    alpha = 0.025,
)::Dict
    X, y = @extractRegressionSetting setting
    return bacon(X, y, m = m, method = method, alpha = alpha)
end


end # end of module Bacon 
