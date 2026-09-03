module PY99

export py99

import ..Basis: RegressionSetting, @extractRegressionSetting, designMatrix, responseVector
import ..OrdinaryLeastSquares: olsf
import LinearAlgebra: Diagonal, PosDefException, Symmetric, cholesky, diag, dot, eigen, rank

const _RHO_BREAKPOINT = 0.810
const _RHO_CUTOFF = 1.215
const _RHO_MAXIMUM = 3.2
const _RHO_EXPECTATION = 1.6

function _rho(u::Float64)::Float64
    au = abs(u)
    if au < _RHO_BREAKPOINT
        return 3.048 * u^2
    elseif au < _RHO_CUTOFF
        u2 = u^2
        return 2.763 * u2^4 - 11.783 * u2^3 + 16.057 * u2^2 - 5.926 * u2 + 1.792
    end
    return _RHO_MAXIMUM
end

function _robustscale(residuals::AbstractVector{Float64})::Float64
    maxresidual = maximum(abs, residuals)
    maxresidual == 0.0 && return 0.0

    function score(scale::Float64)
        total = 0.0
        @inbounds for residual in residuals
            total += _rho(residual / scale)
        end
        return total / length(residuals)
    end

    lower = maxresidual * eps(Float64)
    upper = maxresidual
    while score(upper) > _RHO_EXPECTATION
        upper *= 2.0
    end
    for _ = 1:80
        middle = (lower + upper) / 2.0
        if score(middle) > _RHO_EXPECTATION
            lower = middle
        else
            upper = middle
        end
    end
    return upper
end

function _least_squares(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}, indices::AbstractVector{Int})
    length(indices) < size(X, 2) && return nothing
    Xsubset = view(X, indices, :)
    rank(Xsubset) < size(X, 2) && return nothing
    return olsf(Xsubset, view(y, indices))
end

function _principal_sensitivity_components(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    indices::AbstractVector{Int},
)::Union{Nothing, Matrix{Float64}}
    betas = _least_squares(X, y, indices)
    isnothing(betas) && return nothing
    Xsubset = Matrix(view(X, indices, :))
    residuals = view(y, indices) .- Xsubset * betas
    gram = Symmetric(Xsubset' * Xsubset)
    factor = try
        cholesky(gram)
    catch error
        error isa PosDefException || rethrow()
        nothing
    end
    isnothing(factor) && return nothing
    hat = Xsubset * (factor \ Xsubset')
    leverage = diag(hat)
    any(value -> 1.0 - value <= sqrt(eps(Float64)), leverage) && return nothing
    weights = residuals ./ (1.0 .- leverage)
    components = eigen(Symmetric(hat * Diagonal(weights .^ 2) * hat)).vectors
    return components[:, (end - size(X, 2) + 1):end]
end

function _candidate_betas(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    indices::AbstractVector{Int},
)::Vector{Vector{Float64}}
    candidates = Vector{Vector{Float64}}()
    base = _least_squares(X, y, indices)
    !isnothing(base) && push!(candidates, base)
    components = _principal_sensitivity_components(X, y, indices)
    isnothing(components) && return candidates

    removed = fld(length(indices), 2)
    remaining = length(indices) - removed
    remaining < size(X, 2) && return candidates
    for component in eachcol(components)
        for ordering in (sortperm(component), sortperm(component, rev = true), sortperm(abs.(component), rev = true))
            retained = indices[ordering[(removed + 1):end]]
            beta = _least_squares(X, y, retained)
            !isnothing(beta) && push!(candidates, beta)
        end
    end
    return candidates
end

function _best_candidate(
    candidates::Vector{Vector{Float64}},
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
)::Tuple{Vector{Float64}, Float64}
    isempty(candidates) && throw(ArgumentError("PY99 could not construct a full-rank least-squares candidate."))
    bestbeta = candidates[1]
    bestscale = _robustscale(y .- X * bestbeta)
    for beta in Iterators.drop(candidates, 1)
        scale = _robustscale(y .- X * beta)
        if scale < bestscale
            bestbeta, bestscale = beta, scale
        end
    end
    return bestbeta, bestscale
end

"""
    py99(setting; c1 = 2.0, c2 = 2.5, c3 = 2.5, maxiter = 100, atol = 1e-8)

Perform the Peña & Yohai (1999) fast procedure for detecting multiple
regression outliers. The first stage selects least-squares fits after removing
extreme principal sensitivity components, minimizing the paper's robust
M-scale. The second stage retests candidate outliers with externally
studentized residuals and refits least squares to the retained observations.

# Output
- `["outliers"]`: Indices classified as outliers in the second stage.
- `["betas"]`: Final least-squares coefficients fitted after outlier removal.
- `["initial.betas"]`: Robust first-stage coefficient estimate.
- `["scale"]`: First-stage robust M-scale.
- `["iterations"]`: Number of first-stage iterations performed.
- `["converged"]`: Whether the first stage satisfied the convergence tolerance.

# References
Peña, Daniel, and Victor Yohai. "A Fast Procedure for Outlier Diagnostics in
Large Regression Problems." Journal of the American Statistical Association
94.446 (1999): 434-445.
"""
function py99(
    setting::RegressionSetting;
    c1::Float64 = 2.0,
    c2::Float64 = 2.5,
    c3::Float64 = 2.5,
    maxiter::Int = 100,
    atol::Float64 = 1e-8,
)
    X, y = @extractRegressionSetting setting
    return py99(X, y; c1 = c1, c2 = c2, c3 = c3, maxiter = maxiter, atol = atol)
end

"""
    py99(X, y; c1 = 2.0, c2 = 2.5, c3 = 2.5, maxiter = 100, atol = 1e-8)

Perform the Peña & Yohai (1999) fast procedure for detecting multiple
regression outliers. The first stage selects least-squares fits after removing
extreme principal sensitivity components, minimizing the paper's robust
M-scale. The second stage retests candidate outliers with externally
studentized residuals and refits least squares to the retained observations.

# Arguments

- `X`: Design matrix of regression predictors.
- `y`: Response vector of regression outcomes.
- `c1`: Cutoff for first-stage outlier detection (default: 2.0).
- `c2`: Cutoff for second-stage suspected outlier detection (default: 2.5).
- `c3`: Cutoff for second-stage outlier confirmation (default: 2.5).
- `maxiter`: Maximum number of first-stage iterations (default: 100).
- `atol`: Absolute tolerance for first-stage convergence (default: 1e-8).

# References
Peña, Daniel, and Victor Yohai. "A Fast Procedure for Outlier Diagnostics in
Large Regression Problems." Journal of the American Statistical Association
94.446 (1999): 434-445.
"""
function py99(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64};
    c1::Float64 = 2.0,
    c2::Float64 = 2.5,
    c3::Float64 = 2.5,
    maxiter::Int = 100,
    atol::Float64 = 1e-8,
)
    n, p = size(X)
    n > p || throw(ArgumentError("PY99 requires more observations than regression parameters."))
    length(y) == n || throw(DimensionMismatch("X and y must have the same number of observations."))
    all(value -> value > 0.0, (c1, c2, c3)) || throw(ArgumentError("PY99 cutoffs must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))

    allindices = collect(1:n)
    beta, scale = _best_candidate(_candidate_betas(X, y, allindices), X, y)
    converged = false
    iterations = 1

    for iteration = 2:maxiter
        residuals = y .- X * beta
        retained = scale == 0.0 ? allindices : findall(abs.(residuals) .<= c1 * scale)
        candidates = _candidate_betas(X, y, retained)
        push!(candidates, beta)
        nextbeta, nextscale = _best_candidate(candidates, X, y)
        iterations = iteration
        if maximum(abs.(nextbeta .- beta)) <= atol * max(1.0, maximum(abs.(beta)))
            beta, scale = nextbeta, nextscale
            converged = true
            break
        end
        beta, scale = nextbeta, nextscale
    end

    residuals = y .- X * beta
    suspected = scale == 0.0 ? Int[] : findall(abs.(residuals) .> c2 * scale)
    clean = setdiff(allindices, suspected)
    cleanbeta = _least_squares(X, y, clean)
    isnothing(cleanbeta) && (cleanbeta = beta)
    cleanresiduals = y[clean] .- X[clean, :] * cleanbeta
    degreesoffreedom = length(clean) - p
    standarderror = degreesoffreedom > 0 ? sqrt(sum(abs2, cleanresiduals) / degreesoffreedom) : 0.0

    outliers = Int[]
    if standarderror > 0.0
        factor = cholesky(Symmetric(X[clean, :]' * X[clean, :]))
        for index in suspected
            leverage = dot(X[index, :], factor \ X[index, :])
            studentized = (y[index] - dot(X[index, :], cleanbeta)) / (standarderror * sqrt(1.0 + leverage))
            abs(studentized) > c3 && push!(outliers, index)
        end
    end
    finalindices = setdiff(allindices, outliers)
    finalbeta = _least_squares(X, y, finalindices)
    isnothing(finalbeta) && (finalbeta = cleanbeta)

    return Dict(
        "outliers" => outliers,
        "betas" => finalbeta,
        "initial.betas" => beta,
        "scale" => scale,
        "iterations" => iterations,
        "converged" => converged,
    )
end

end # module PY99
