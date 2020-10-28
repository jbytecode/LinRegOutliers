struct OLS
    X::Array{Float64,2}
    y::Array{Float64,1}
    betas::Array{Float64,1}
end

ols(X::Array{Float64,2}, y::Array{Float64,1})::OLS = OLS(X, y, qr(X, Val(true)) \ y)

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

