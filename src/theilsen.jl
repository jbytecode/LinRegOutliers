module TheilSen 

export theilsen 

import ..Basis: RegressionSetting,  designMatrix, responseVector, @extractRegressionSetting
import ..OrdinaryLeastSquares: ols, coef 
import ..HookeJeeves: hj
import Distributions: sample, mean

"""
    theilsen(setting, m, nsamples = 5000)

Theil-Sen estimator for multiple regression. 

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `m::Int`: Number of observations to be used in each iteration. This number must be in the range [p, n], where p is the number of regressors and n is the number of observations.
- `nsamples::Int`: Number of m-samples. Default is 5000.

# Description
The function starts with a regression formula and datasets. The number of observations to be used in each iteration 
is specified by the user. The function then randomly selects m observations from the dataset and performs an ordinary 
least squares estimation. The estimated coefficients are saved. The process is repeated until nsamples regressions are estimated. 
The multivariate median of the estimated coefficients is then calculated. In this case, the multivariate median is the
point that minimizes the sum of distances to all the estimated coefficients. Hooke & Jeeves algorithm is used for the 
optimization problem. 

# References
Dang, X., Peng, H., Wang, X., & Zhang, H. (2008). Theil-sen estimators in a multiple linear regression model. 
Olemiss Edu.
"""
function theilsen(setting::RegressionSetting, m::Int; nsamples::Int = 5000)
    X, y = @extractRegressionSetting setting
    return theilsen(X, y, m, nsamples = nsamples)
end

function theilsen(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}, m::Int; nsamples::Int = 5000)
    
    n, p = size(X)

    if m > n || m < p  
        error("m must be in the range [p, n]")
    end 

    allbetas = Matrix{Float64}(undef, nsamples, p)

    for i in 1:nsamples
        luckyindices = sample(1:n, m, replace = false)
        olsresult = ols(X[luckyindices, :], y[luckyindices])
        betas = coef(olsresult)
        allbetas[i, :] = betas
    end 

    multimed = multivariatemedian(allbetas)
    return Dict("betas" => multimed)
end 

function multivariatemedian(X::AbstractMatrix{Float64})
    n, p = size(X)
   
    function dist(x::AbstractVector{Float64}, y::AbstractVector{Float64})
        return sum(abs.(x .- y))
    end

    function objective(candidate::AbstractVector{Float64})
        val = sum(dist(candidate, X[i, :]) for i in 1:n)
        return val
    end 
   
    initials = map(i -> mean(X[:, i]), 1:p)
    optresult = hj(objective, initials, maxiter = 10000, startstep = 500.0, endstep = 10^-6)
    return optresult["par"]
    
end


end # end of module