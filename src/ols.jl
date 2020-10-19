struct OLS
    X::Array{Float64,2}
    y::Array{Float64,1}
    betas::Array{Float64,1}
end

function ols(X::Array{Float64,2}, y::Array{Float64,1})::OLS
    @show X
    return OLS(X, y, qr(X) \ y)
end

function wls(X::Array{Float64,2}, y::Array{Float64,1}, wts::Array{Float64,1})
    W = Diagonal(sqrt.(wts))
    return ols(W*X, W*y)
end 

residuals(ols::OLS) = ols.y .- ols.X * ols.betas

coef(ols::OLS) = ols.betas

predict(ols::OLS) = ols.X * ols.betas

predict(ols::OLS, X::Array{Float64,2}) = X * ols.betas

