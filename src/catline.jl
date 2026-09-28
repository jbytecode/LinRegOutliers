module Catline

import ..Basis: RegressionSetting, @extractRegressionSetting, designMatrix, responseVector

export catline

"""
    catline(setting)

Fit the Catline estimator of Hubert and Rousseeuw (1998) to a simple linear
regression setting.

The Catline simultaneously bisects the left-middle and middle-right thirds
of the observations when ordered by the predictor. It is defined only for
one-predictor regression models.

# Output
- `["betas"]`: Intercept and slope of a Catline.
- `["residuals"]`: Residuals in the input order.

# References
Hubert, M. and Rousseeuw, P. J. (1998). "The Catline for Deep Regression."
_Journal of Multivariate Analysis_, 66, 270-296.
"""
function catline(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return catline(X, y)
end

"""
    catline(X, y)

Fit the Catline estimator to a design matrix consisting of an intercept and
one predictor column, and a response vector.

If multiple lines satisfy the Catline criterion, the returned line is the
deterministic representative with the smallest absolute slope.
"""
function catline(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})::Dict{String,Any}
    n, p = size(X)
    n == length(y) || throw(DimensionMismatch("X and y must have the same number of rows"))
    p == 2 || throw(ArgumentError("catline supports exactly one predictor plus an intercept"))
    n >= 3 || throw(ArgumentError("catline requires at least three observations"))
    all(isone, X[:, 1]) || throw(ArgumentError("the first column of X must be an intercept"))

    x = Float64.(X[:, 2])
    response = Float64.(y)
    all(isfinite, x) && all(isfinite, response) ||
        throw(ArgumentError("catline requires finite predictor and response values"))
    all(==(x[1]), x) && throw(ArgumentError("catline requires at least two distinct predictor values"))

    order = sortperm(1:n, by = i -> (x[i], response[i]))
    sorted_x = x[order]
    sorted_y = response[order]
    left, middle, right = _catline_groups(n)
    left_middle = vcat(left, middle)
    middle_right = vcat(middle, right)

    candidates = Set{Tuple{Float64,Float64}}()
    for i in 1:(n - 1), j in (i + 1):n
        sorted_x[i] == sorted_x[j] && continue
        slope = (sorted_y[j] - sorted_y[i]) / (sorted_x[j] - sorted_x[i])
        intercept = sorted_y[i] - slope * sorted_x[i]
        push!(candidates, (slope, intercept))
    end

    # A non-unique Catline can include a horizontal line not determined by
    # two observations, so include the median intercepts at zero slope.
    for intercept in _median_endpoints(sorted_y[left_middle])
        push!(candidates, (0.0, intercept))
    end
    for intercept in _median_endpoints(sorted_y[middle_right])
        push!(candidates, (0.0, intercept))
    end

    fits = Tuple{Float64,Float64}[]
    for (slope, intercept) in candidates
        residuals = sorted_y .- slope .* sorted_x .- intercept
        _bisects(residuals[left_middle]) && _bisects(residuals[middle_right]) &&
            push!(fits, (slope, intercept))
    end

    isempty(fits) && error("unable to construct a Catline for the supplied data")

    # The Catline need not be unique. Select a deterministic representative.
    sort!(fits, by = fit -> (abs(fit[1]), fit[1], fit[2]))
    intercept, slope = fits[1][2], fits[1][1]
    betas = [intercept, slope]
    return Dict{String,Any}("betas" => betas, "residuals" => response .- Float64.(X) * betas)
end

function _catline_groups(n::Int)
    m, remainder = divrem(n, 3)
    lengths = remainder == 0 ? (m, m, m) :
              remainder == 1 ? (m, m + 1, m) : (m + 1, m, m + 1)
    left_end = lengths[1]
    middle_end = left_end + lengths[2]
    return 1:left_end, (left_end + 1):middle_end, (middle_end + 1):n
end

function _median_endpoints(values::AbstractVector{Float64})
    ordered = sort(values)
    midpoint = length(ordered) ÷ 2
    return isodd(length(ordered)) ? (ordered[midpoint + 1],) :
                                    (ordered[midpoint], ordered[midpoint + 1])
end

function _bisects(residuals::AbstractVector{Float64})
    maximum_allowed = length(residuals) ÷ 2
    return count(>(0.0), residuals) <= maximum_allowed &&
           count(<(0.0), residuals) <= maximum_allowed
end

end # module Catline