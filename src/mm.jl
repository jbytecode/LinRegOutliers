module Yohai87

export mm

import ..Basis: RegressionSetting, @extractRegressionSetting, designMatrix, responseVector
import ..LMS: lms
import ..LTS: lts
import ..OrdinaryLeastSquares: wls, coef

import LinearAlgebra: dot, mul!

# Tukey bisquare constants from Yohai (1987, Remark 4.1)
const _K0 = 1.56
const _K1 = 4.68
const _RHO_MAX = 1.0 / 6.0
const _B = 0.5 * _RHO_MAX

"""
    _tukey_rho(u)

Tukey's bisquare ``\\rho`` function ``\\rho_B`` as in Yohai (1987, eq. 4.5).
"""
function _tukey_rho(u::Float64)::Float64
    au = abs(u)
    if au >= 1.0
        return _RHO_MAX
    end
    u2 = u * u
    u4 = u2 * u2
    return 0.5 * u2 - 0.5 * u4 + u4 * u2 / 6.0
end

function _mm_objective(
    residuals::AbstractVector{Float64},
    scale::Float64,
    k1::Float64,
)::Float64
    denom = scale * k1
    total = 0.0
    @inbounds for r in residuals
        total += _tukey_rho(r / denom)
    end
    return total
end

function _residuals!(
    residuals::AbstractVector{Float64},
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    betas::AbstractVector{Float64},
)::AbstractVector{Float64}
    mul!(residuals, X, betas)
    @inbounds for i in eachindex(y)
        residuals[i] = y[i] - residuals[i]
    end
    return residuals
end

function _mscale(
    residuals::AbstractVector{Float64},
    k0::Float64,
    b::Float64,
)::Float64
    n = length(residuals)
    nzero = 0
    maxabs = 0.0
    @inbounds for r in residuals
        ar = abs(r)
        if ar == 0.0
            nzero += 1
        elseif ar > maxabs
            maxabs = ar
        end
    end
    if nzero / n >= 1.0 - b / _RHO_MAX || maxabs == 0.0
        return 0.0
    end

    function meanrho(scale::Float64)::Float64
        total = 0.0
        denom = scale * k0
        @inbounds for r in residuals
            total += _tukey_rho(r / denom)
        end
        return total / n
    end

    lower = maxabs * eps(Float64)
    upper = maxabs
    while meanrho(upper) > b
        upper *= 2.0
        if !isfinite(upper)
            return maxabs
        end
    end
    for _ = 1:80
        middle = (lower + upper) / 2.0
        if meanrho(middle) > b
            lower = middle
        else
            upper = middle
        end
    end
    return upper
end

function _irls_weights!(
    weights::AbstractVector{Float64},
    residuals::AbstractVector{Float64},
    scale::Float64,
    k1::Float64,
)::Nothing
    invk1sq = 1.0 / (k1 * k1)
    thresh = scale * k1
    tiny = eps(Float64) * max(scale, 1.0)
    @inbounds for i in eachindex(residuals)
        ar = abs(residuals[i])
        if ar <= tiny
            weights[i] = invk1sq
        elseif ar >= thresh
            weights[i] = 0.0
        else
            v = residuals[i] / thresh
            t = 1.0 - v * v
            weights[i] = (t * t) * invk1sq
        end
    end
    return nothing
end

function _gradient!(
    g::AbstractVector{Float64},
    X::AbstractMatrix{Float64},
    residuals::AbstractVector{Float64},
    weights::AbstractVector{Float64},
    scale::Float64,
)::Nothing
    fill!(g, 0.0)
    invs2 = 1.0 / (scale * scale)
    n, p = size(X)
    @inbounds for i = 1:n
        wr = weights[i] * residuals[i] * invs2
        for j = 1:p
            g[j] += wr * X[i, j]
        end
    end
    return nothing
end

function _initial_betas(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    initial,
    iters,
)::Vector{Float64}
    if initial isa AbstractVector
        return Vector{Float64}(initial)
    elseif initial === :lms
        if isnothing(iters)
            return lms(X, y)["betas"]
        end
        return lms(X, y, iters = iters)["betas"]
    elseif initial === :lts
        if isnothing(iters)
            return lts(X, y)["betas"]
        end
        return lts(X, y, iters = iters)["betas"]
    else
        throw(ErrorException("Unknown initial estimator: $initial. Use :lms, :lts, or a coefficient vector."))
    end
end

"""
    mm(setting; initial = :lms, k0 = 1.56, k1 = 4.68, crit = 2.5, maxiter = 500, atol = 1e-8, delta = 0.1, iters = nothing)
    mm(X, y; ...)

Yohai (1987) MM-estimator for linear regression.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `initial`: High-breakdown starting estimator. `:lms` (default), `:lts`, or a coefficient vector.
- `k0::Float64`: Tuning constant of ``\\rho_0``. Default `1.56` gives breakdown point 0.5.
- `k1::Float64`: Tuning constant of ``\\rho_1``. Default `4.68` gives 95% efficiency at the Gaussian model.
- `crit::Float64`: Cutoff for scaled residuals used to report outliers.
- `maxiter::Int`: Maximum number of modified IWLS iterations.
- `atol::Float64`: Convergence tolerance on the coefficient update.
- `delta::Float64`: Armijo line-search constant ``\\delta \\in (0, 1)``.
- `iters`: Iteration budget forwarded to `lms` or `lts` when they are used as the initial estimator.

# Description
The estimator follows the three-stage definition in Yohai (1987). Stage 1 computes a
high-breakdown initial fit ``T_{0,n}``. Stage 2 computes an M-scale ``s_n`` of the
initial residuals with Tukey bisquare ``\\rho_0`` and ``b / a = 0.5``. Stage 3
minimizes ``S(\\theta) = \\sum \\rho_1(r_i(\\theta) / s_n)`` with the modified iterated
weighted least-squares algorithm of Section 5, which enforces ``S(T_{1,n}) \\le S(T_{0,n})``.

# Output
- `["betas"]`: MM regression coefficients.
- `["S"]`: Residual M-scale ``s_n``.
- `["objective"]`: Value of ``S(T_{1,n})``.
- `["initial.betas"]`: Coefficients of the initial estimator.
- `["initial.objective"]`: Value of ``S(T_{0,n})``.
- `["outliers"]`: Indices of observations with absolute scaled residuals larger than `crit`.
- `["scaled.residuals"]`: Residuals divided by ``s_n``.
- `["iterations"]`: Number of IWLS iterations.
- `["converged"]`: `true` if the coefficient update fell below `atol`.

# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> mm(reg)
```

# References
Yohai, Victor J. "High breakdown-point and high efficiency robust estimates
for regression." The Annals of Statistics 15.2 (1987): 642-656.
"""
function mm(
    setting::RegressionSetting;
    initial = :lms,
    k0::Float64 = _K0,
    k1::Float64 = _K1,
    crit::Float64 = 2.5,
    maxiter::Int = 500,
    atol::Float64 = 1e-8,
    delta::Float64 = 0.1,
    iters = nothing,
)
    X, y = @extractRegressionSetting setting
    return mm(
        X,
        y;
        initial = initial,
        k0 = k0,
        k1 = k1,
        crit = crit,
        maxiter = maxiter,
        atol = atol,
        delta = delta,
        iters = iters,
    )
end

function mm(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64};
    initial = :lms,
    k0::Float64 = _K0,
    k1::Float64 = _K1,
    crit::Float64 = 2.5,
    maxiter::Int = 500,
    atol::Float64 = 1e-8,
    delta::Float64 = 0.1,
    iters = nothing,
)::Dict{String, Any}

    if !(0.0 < delta < 1.0)
        throw(ErrorException("delta must lie in (0, 1)."))
    end
    if k0 <= 0.0 || k1 <= 0.0
        throw(ErrorException("k0 and k1 must be positive."))
    end
    if k1 < k0
        throw(ErrorException("k1 must be at least k0 so that ρ₁ ≤ ρ₀."))
    end

    n, p = size(X)
    if length(y) != n
        throw(ErrorException("X and y have incompatible sizes."))
    end
    if n <= p
        throw(ErrorException("n must be larger than p."))
    end

    initialbetas = _initial_betas(X, y, initial, iters)
    if length(initialbetas) != p
        throw(ErrorException("initial coefficient vector must have length p = $p."))
    end

    residuals = Vector{Float64}(undef, n)
    _residuals!(residuals, X, y, initialbetas)
    sn = _mscale(residuals, k0, _B)

    betas = copy(initialbetas)
    trial = Vector{Float64}(undef, p)
    delta_beta = Vector{Float64}(undef, p)
    g = Vector{Float64}(undef, p)
    weights = Vector{Float64}(undef, n)
    trialres = Vector{Float64}(undef, n)
    iterations = 0
    converged = false

    if sn == 0.0
        scaled = fill(0.0, n)
        return Dict{String, Any}(
            "betas" => betas,
            "S" => sn,
            "objective" => 0.0,
            "initial.betas" => initialbetas,
            "initial.objective" => 0.0,
            "outliers" => Int[],
            "scaled.residuals" => scaled,
            "iterations" => 0,
            "converged" => true,
        )
    end

    objective = _mm_objective(residuals, sn, k1)
    initialobjective = objective

    for iter = 1:maxiter
        iterations = iter
        _irls_weights!(weights, residuals, sn, k1)
        npos = 0
        @inbounds for w in weights
            if w > 0.0
                npos += 1
            end
        end
        if npos < p
            break
        end

        candidate = coef(wls(X, y, weights))
        @inbounds for j = 1:p
            delta_beta[j] = candidate[j] - betas[j]
        end

        maxstep = 0.0
        maxbeta = 0.0
        @inbounds for j = 1:p
            aj = abs(delta_beta[j])
            if aj > maxstep
                maxstep = aj
            end
            bj = abs(betas[j])
            if bj > maxbeta
                maxbeta = bj
            end
        end
        if maxstep <= atol * max(1.0, maxbeta)
            converged = true
            break
        end

        _gradient!(g, X, residuals, weights, sn)
        directional = dot(delta_beta, g)
        if directional <= atol * max(1.0, abs(objective))
            converged = true
            break
        end
        k1star = 20
        found = false
        @inbounds for k = 0:20
            lam = 0.5^k
            for j = 1:p
                trial[j] = betas[j] + lam * delta_beta[j]
            end
            _residuals!(trialres, X, y, trial)
            newobj = _mm_objective(trialres, sn, k1)
            if newobj <= objective - delta * lam * directional
                k1star = k
                found = true
                break
            end
        end
        if !found
            break
        end

        bestobj = Inf
        bestk = k1star
        @inbounds for k = 0:k1star
            lam = 0.5^k
            for j = 1:p
                trial[j] = betas[j] + lam * delta_beta[j]
            end
            _residuals!(trialres, X, y, trial)
            newobj = _mm_objective(trialres, sn, k1)
            if newobj < bestobj
                bestobj = newobj
                bestk = k
            end
        end

        lam = 0.5^bestk
        @inbounds for j = 1:p
            betas[j] += lam * delta_beta[j]
        end
        _residuals!(residuals, X, y, betas)
        objective = bestobj
    end

    scaled = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        scaled[i] = residuals[i] / sn
    end

    outlierindices = Int[]
    sizehint!(outlierindices, n)
    @inbounds for i = 1:n
        if abs(scaled[i]) > crit
            push!(outlierindices, i)
        end
    end

    return Dict{String, Any}(
        "betas" => betas,
        "S" => sn,
        "objective" => objective,
        "initial.betas" => initialbetas,
        "initial.objective" => initialobjective,
        "outliers" => outlierindices,
        "scaled.residuals" => scaled,
        "iterations" => iterations,
        "converged" => converged,
    )
end

end
