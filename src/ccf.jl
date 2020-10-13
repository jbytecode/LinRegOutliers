"""

    ccf(setting; starting_lambdas = nothing)

Perform signed gradient descent for clipped convex functions for a given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `starting_lambdas::Array{Float64,1}`: Starting values of weighting parameters used by signed gradient descent.
- `alpha::Float64`: Loss at which a point is labeled as an outlier (points with loss ≥ alpha will be called outliers).
- `max_iter::Int64`: Maximum number of iterations to run signed gradient descent.
- `beta::Float64`: Step size parameter.
- `tol::Float64`: Tolerance below which convergence is declared.


# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> ccf(reg0001)
Dict{Any,Any} with 2 entries:
  "betas"     => [-57.3269, 1.19155]
  "residuals" => [2.14958, 1.25803, 0.0664872, 0.0749413, -0.416605, -0.90815, -1.2997, -1.79124,…

```

"""
function ccf(setting::RegressionSetting; starting_lambdas=nothing, alpha=1.0, max_iter=100, beta=.1, tol=1e-4)
    X = designMatrix(setting)
    y = responseVector(setting)
    return ccf(X, y, starting_lambdas=starting_lambdas, alpha=alpha, max_iter=max_iter, beta=beta, tol=tol)
end


"""

    ccf(X, y; starting_lambdas = nothing)

Perform signed gradient descent for clipped convex functions for a given regression setting.

# Arguments
- `X::Array{Float64, 2}`: Design matrix of the linear model.
- `y::Array{Float64, 1}`: Response vector of the linear model.
- `starting_lambdas::Array{Float64,1}`: Starting values of weighting parameters used by signed gradient descent.
- `alpha::Float64`: Loss at which a point is labeled as an outlier (points with loss ≥ alpha will be called outliers).
- `max_iter::Int64`: Maximum number of iterations to run signed gradient descent.
- `beta::Float64`: Step size parameter.
- `tol::Float64`: Tolerance below which convergence is declared.

"""
function ccf(X::Array{Float64,2}, y::Array{Float64,1}; starting_lambdas=nothing, alpha=1.0, max_iter=100, beta=.1, tol=1e-4)
    n, p = size(X)

    if starting_lambdas isa Nothing
        starting_lambdas = ones(Float64, p)/2
    end

    # Solves the weighted least squares problem: minimize ∑_i λ_i(X_i'β - y_i)^2
    function solve_weighted_regression(λ)
        diag_weights = Diagonal(sqrt.(λ))

        Q, R = qr(diag_weights*X)

        return Q' * (UpperTriangular(R) \ (diag_weights*y))
    end

    curr_lambdas = copy(starting_lambdas)
    old_lambdas = copy(starting_lambdas)

    for iter=1:max_iter
        curr_betas = solve_weighted_regression(curr_lambdas)
        residuals = X * curr_betas - y

        curr_lambdas .-= beta*sign.(residuals.^2 .- alpha)

        if norm(new_lambdas - old_lambdas, Inf) <= tol
            break
        end

        old_lambdas .= curr_lambdas
    end

    result = Dict()
    result["betas"] = curr_betas
    result["lambdas"] = old_lambdas
    result["residuals"] = residuals
    result["outliers"] = findall(residuals.^2 .> alpha)

    return result
end