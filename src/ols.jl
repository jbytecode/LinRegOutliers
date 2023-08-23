module OrdinaryLeastSquares


export OLS, ols, wls, residuals, predict, coef

import LinearAlgebra: ColumnNorm, qr, Diagonal
import ..Basis:
    RegressionSetting, @extractRegressionSetting, designMatrix, responseVector

"""
    struct OLS
        X::AbstractMatrix{Float64}
        y::AbstractVector{Float64}
        betas::AbstractVector{Float64}
    end 
    
    Immutable data structure that holds design matrix, response vector, and estimated regression parameters. 

# Arguments
- `X::AbstractMatrix{Float64}`: Design matrix.
- `y::AbstractVector{Float64}`: Response vector.
- `betas::AbstractVector{Float64}`: Regression coefficients.

"""
struct OLS
    X::AbstractMatrix{Float64}
    y::AbstractVector{Float64}
    betas::AbstractVector{Float64}
end


"""
    ols(X, y)

    Create OLS object with estimated regression coefficients.

# Arguments
- `X::AbstractMatrix{Float64}`: Design matrix.
- `y::AbstractVector{Float64}`: Response vector.

# Examples
```julia-repl
julia> X = hcat(ones(24), phones[:,"year"]);
julia> y = phones[:,"calls"];
julia> reg = ols(X, y)
julia> reg.betas
2-element Vector{Float64}:
 -260.0592463768119
    5.04147826086957
```

"""
ols(X::AbstractMatrix{Float64}, y::AbstractVector{Float64})::OLS = OLS(X, y, qr(X, ColumnNorm()) \ y)
#ols(X::AbstractMatrix{Float64}, y::AbstractVector{Float64})::OLS = OLS(X, y, qr(X, Val(true)) \ y)


function ols(setting::RegressionSetting)::OLS
    X, y = @extractRegressionSetting setting
    return ols(X, y)
end 



"""
    wls(X, y, wts)

    Estimate weighted least squares regression and create OLS object with estimated parameters.

# Arguments
- `X::AbstractMatrix{Float64}`: Design matrix.
- `y::AbstractVector{Float64}`: Response vector.
- `wts::AbstractVector{Float64}`: Weights vector.


# Examples
```julia-repl
julia> X = hcat(ones(24), phones[:,"year"]);
julia> y = phones[:,"calls"];
julia> w = ones(24)
julia> w[15:20] .= 0.0
julia> reg = wls(X, y, w)
julia> reg.betas
2-element Vector{Float64}:
 -63.481644325290425
   1.3040571939231453
```


"""
function wls(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}, wts::AbstractVector{Float64})::OLS
    W = Diagonal(sqrt.(wts))
    #  I commented this because passing weighted values of X and y to OLS 
    #  causes wrong calculations of residuals.
    #  return ols(W * X, W * y)
    # return OLS(X, y, qr(W * X, Val(true)) \ (W * y))
    return OLS(X, y, qr(W * X, ColumnNorm()) \ (W * y))
end


function wls(setting::RegressionSetting; weights = nothing)::OLS
    X, y = @extractRegressionSetting setting
    if isnothing(weights)
        return ols(X, y)
    else
        return wls(X, y, weights)
    end 
end 

"""
    residuals(ols)

Estimate weighted least squares regression and create OLS object with estimated parameters.

# Arguments
- `ols::OLS`: OLS object, possible created using `ols` or `wls`.

"""
residuals(ols::OLS)::AbstractVector{Float64} = ols.y .- ols.X * ols.betas


"""
    coef(ols)

Extract regression coefficients from an `OLS` object.

# Arguments
- `ols::OLS`: OLS object, possible created using `ols` or `wls`.

"""
coef(ols::OLS)::AbstractVector{Float64} = ols.betas


"""
    predict(ols)

Calculate estimated response using an `OLS` object.

# Arguments
- `ols::OLS`: OLS object, possible created using `ols` or `wls`.

"""
predict(ols::OLS)::AbstractVector{Float64} = ols.X * ols.betas

predict(ols::OLS, X::AbstractMatrix{Float64})::AbstractVector{Float64} = X * ols.betas



end # end of module OrdinaryLeastSquares 
