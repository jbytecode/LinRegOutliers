module RobustHatRegression


export robhatreg

import ..Basis: RegressionSetting, @extractRegressionSetting, designMatrix, responseVector
import ..OrdinaryLeastSquares: ols, residuals, coef
import ..LTS: iterateCSteps

import Distributions: quantile
import LinearAlgebra: inv, diag


function trimean(u::AbstractVector{T})::Float64 where {T<:Real}
    return (quantile(u, 0.25) + 2.0 * quantile(u, 0.50) + quantile(u, 0.75)) / 4.0
end

function m(v::AbstractVector{T}, u::AbstractVector{T})::Float64 where {T<:Real}
    return trimean(u .* v) * length(u)
end

function m(mat::AbstractMatrix{T}, u::AbstractVector{T})::AbstractMatrix where {T<:Real}
    L = length(u)
    y = zeros(Float64, L, 1)
    for i in 1:L
        y[i, 1] = u[i]
    end
    result = m(mat, y)
    return result
end

function m(m1::AbstractMatrix{T}, m2::AbstractMatrix{T})::AbstractMatrix where {T<:Real}
    n1, _ = size(m1)
    _, p2 = size(m2)
    newmat = zeros(Float64, n1, p2)
    for i in 1:n1
        for j in 1:p2
            newmat[i, j] = m(m1[i, :], m2[:, j])
        end
    end
    return newmat
end

function hatrob(x::AbstractMatrix{T}) where {T<:Real}
    return x * inv(m(x', x)) * x'
end


"""
    robhatreg(setting::RegressionSetting)

Perform robust regression using the robust hat matrix method.

# Arguments
- `setting::RegressionSetting`: The regression setting.

# Returns

- A dictionary containing the following
    - `betas::AbstractVector`: The estimated coefficients.

# References

Satman, Mehmet Hakan, A robust initial basic subset selection 
method for outlier detection algorithms in linear regression, In Press
"""
function robhatreg(setting::RegressionSetting)::Dict
    X, y = @extractRegressionSetting setting
    return robhatreg(X, y)
end


"""
    robhatreg(X, y)

Perform robust regression using the robust hat matrix method.

# Arguments

- `X::AbstractMatrix`: The design matrix.
- `y::AbstractVector`: The response vector.

# Returns

- A dictionary containing the following
    - `betas::AbstractVector`: The estimated coefficients.

# References

Satman, Mehmet Hakan, A robust initial basic subset selection 
method for outlier detection algorithms in linear regression, In Press
"""
function robhatreg(X::AbstractMatrix{T}, y::AbstractVector{T})::Dict where {T<:Real}
    n, p = size(X)
    h = Int(ceil((n + p + 1) / 2))
    myhat = hatrob(X)
    diagonals = diag(myhat)
    prms = sortperm(diagonals)
    bestindices = prms[1:(p+1)]
    _, indices = iterateCSteps(X, y, bestindices, h)
    betas = X[indices, :] \ y[indices]
    return Dict("betas" => betas)
end



end # end of module RobustHatRegression