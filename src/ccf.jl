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


# Output 
- `["betas"]`: Robust regression coefficients
- `[""outliers"]`: Array of indices of outliers
- `[""lambdas"]`: Lambda coefficients estimated in each iteration 
- `[""residuals"]`: Regression residuals.


# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(calls ~ year), phones);
julia> ccf(reg0001)
Dict{Any,Any} with 4 entries:
  "betas"     => [-63.4816, 1.30406]
  "outliers"  => [15, 16, 17, 18, 19, 20]
  "lambdas"   => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  2.77556e-17, 2.77556e-17, 0…
  "residuals" => [-2.67878, -1.67473, -0.37067, -0.266613, 0.337444, 0.941501, 1.44556, 2.04962, 1…

```

# References 
Barratt, S., Angeris, G. & Boyd, S. Minimizing a sum of clipped convex functions. Optim Lett 14, 2443–2459 (2020). https://doi.org/10.1007/s11590-020-01565-4

"""
function ccf(setting::RegressionSetting; starting_lambdas=nothing, alpha=nothing, p=3, max_iter=100, gamma=.1, tol=1e-4)
    X, y = @extractRegressionSetting setting
    return ccf(X, y, starting_lambdas=starting_lambdas, alpha=alpha, p=p, max_iter=max_iter, gamma=gamma, tol=tol)
end


"""

    ccf(X, y; starting_lambdas = nothing)

Perform signed gradient descent for clipped convex functions for a given regression setting.

# Arguments
- `X::Array{Float64, 2}`: Design matrix of the linear model.
- `y::Array{Float64, 1}`: Response vector of the linear model.
- `starting_lambdas::Array{Float64,1}`: Starting values of weighting parameters used by signed gradient descent.
- `alpha::Float64`: Loss at which a point is labeled as an outlier. If unspecified, will be chosen as p*mean(residuals.^2), where residuals are OLS residuals.
- `p::Float64`: Points that have squared OLS residual greater than p times the mean squared OLS residual are considered outliers.
- `max_iter::Int64`: Maximum number of iterations to run signed gradient descent.
- `beta::Float64`: Step size parameter.
- `tol::Float64`: Tolerance below which convergence is declared.

# Output 
- `["betas"]`: Robust regression coefficients
- `[""outliers"]`: Array of indices of outliers
- `[""lambdas"]`: Lambda coefficients estimated in each iteration 
- `[""residuals"]`: Regression residuals.


# References
Barratt, S., Angeris, G. & Boyd, S. Minimizing a sum of clipped convex functions. Optim Lett 14, 2443–2459 (2020). https://doi.org/10.1007/s11590-020-01565-4

"""
function ccf(X::Array{Float64,2}, y::Array{Float64,1}; starting_lambdas=nothing, alpha=nothing, p=3, max_iter=100, gamma=.1, tol=1e-4)
    n, p = size(X)

    if isnothing(starting_lambdas)
        starting_lambdas = ones(Float64, n) / 2
    end

    curr_lambdas = copy(starting_lambdas)
    old_lambdas = copy(starting_lambdas)
    curr_betas = nothing
    residuals = nothing

    for iter = 1:max_iter
        curr_betas = wls(X, y, curr_lambdas).betas
        residuals = X * curr_betas - y

        if isnothing(alpha)
            alpha = p * mean(residuals.^2)
        end

        @. curr_lambdas -= gamma * sign(residuals^2 - alpha)
        clamp!(curr_lambdas, 0., 1.)

        if norm(curr_lambdas - old_lambdas, Inf) <= tol
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
