module CM97


export cm97

import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..Diagnostics: hatmatrix
import ..OrdinaryLeastSquares: ols, predict, residuals, coef, wls
import Distributions: median

"""

    cm97(setting; maxiter = 1000)

Perform the Chatterjee and Mächler (1997) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description 
The algorithm performs a iteratively weighted least squares estimation to obtain
robust regression coefficients.

# Output
- `["betas"]`: Robust regression coefficients
- `["iterations"]`: Number of iterations performed
- `["converged"]`: true if the algorithm converges, otherwise, false.

# Examples
```julia-repl
julia> myreg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
julia> result = cm97(myreg)
Dict{String,Any} with 3 entries:
  "betas"      => [-37.0007, 0.839285, 0.632333, -0.113208]
  "iterations" => 22
  "converged"  => true
```

# References
Chatterjee, Samprit, and Martin Mächler. "Robust regression: A weighted least squares approach." 
Communications in Statistics-Theory and Methods 26.6 (1997): 1381-1394.
"""
function cm97(setting::RegressionSetting; maxiter::Int = 1000)

    X, y = @extractRegressionSetting setting
    return cm97(X, y, maxiter = maxiter)

end



function cm97(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; maxiter::Int = 1000)::Dict

    n, p = size(X)
    pbar::Float64 = p / n
    hat = hatmatrix(X)

    w_is = Vector{Float64}(undef, n)
    betas = Vector{Float64}(undef, p)

    converged::Bool = false
    iter::Int = 0

    # initial weights
    for i = 1:n
        w_is[i] = 1.0 / max(hat[i, i], pbar)
    end

    while iter <= maxiter
        oldbetas = betas

        wols = wls(X, y, w_is)
        betas = coef(wols)
        r = y - X * betas
        medi = median(abs.(r))

        for i = 1:n
            w_is[i] = (1.0 - hat[i, i])^2.0 / max(abs(r[i]), medi)
        end

        if isapprox(oldbetas, betas)
            converged = true
            break
        end

        iter += 1

    end

    result =
        Dict{String,Any}("converged" => converged, "betas" => betas, "iterations" => iter)

    return result
end


end # end of module CM97
