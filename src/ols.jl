"""
    struct OLS
        X::Array{Float64,2}
        y::Array{Float64,1}
        betas::Array{Float64,1}
    end 
    
    Immutable data structure that holds design matrix, response vector, and estimated regression parameters. 

# Arguments
- `X::Array{Float64, 2}`: Design matrix.
- `y::Array{Float64, 1}`: Response vector.
- `betas::Array{Float64, 1}`: Regression coefficients.
"""
struct OLS
    X::Array{Float64,2}
    y::Array{Float64,1}
    betas::Array{Float64,1}
end


"""
    ols(X, y)

    Create OLS object with estimated regression coefficients.

# Arguments
- `X::Array{Float64, 2}`: Design matrix.
- `y::Array{Float64, 1}`: Response vector.

# Examples
```julia-repl
julia> X = hcat(ones(24), phones[:,"year"]);
julia> y = phones[:,"calls"];
julia> reg = ols(X, y)
julia> reg.betas
2-element Array{Float64,1}:
 -260.0592463768119
    5.04147826086957
```
"""
ols(X::Array{Float64,2}, y::Array{Float64,1})::OLS = OLS(X, y, qr(X, Val(true)) \ y)



"""
    wls(X, y, wts)

    Estimate weighted least squares regression and create OLS object with estimated parameters.

# Arguments
- `X::Array{Float64, 2}`: Design matrix.
- `y::Array{Float64, 1}`: Response vector.
- `wts::Array{Float64, 1}`: Weights vector.


# Examples
```julia-repl
julia> X = hcat(ones(24), phones[:,"year"]);
julia> y = phones[:,"calls"];
julia> w = ones(24)
julia> w[15:20] .= 0.0
julia> reg = wls(X, y, w)
julia> reg.betas
2-element Array{Float64,1}:
 -63.481644325290425
   1.3040571939231453
```
"""
function wls(X::Array{Float64,2}, y::Array{Float64,1}, wts::Array{Float64,1})
    W = Diagonal(sqrt.(wts))
    #  I commented this because passing weighted values of X and y to OLS 
    #  causes wrong calculations of residuals.
    #  return ols(W * X, W * y)
    return OLS(X, y, qr(W * X, Val(true)) \ (W * y))
end 


residuals(ols::OLS) = ols.y .- ols.X * ols.betas

coef(ols::OLS) = ols.betas

predict(ols::OLS) = ols.X * ols.betas

predict(ols::OLS, X::Array{Float64,2}) = X * ols.betas

