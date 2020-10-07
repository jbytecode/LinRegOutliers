struct OLS
    X::Array{Float64,2}
    y::Array{Float64,1}
    betas::Array{Float64,1}
end

ols(X::Array{Float64,2}, y::Array{Float64,1})::OLS = OLS(X, y, inv(X' * X) * X' * y)

function wls(X::Array{Float64,2}, y::Array{Float64,1}, wts::Array{Float64,1}) 
    n = length(y)
    W = zeros(Float64, n, n)
    for i in 1:n
        W[i, i] = wts[i]
    end
    betas = inv(X' * W * X) * X' * W * y
    return OLS(X, y, betas)
end 

residuals(ols::OLS) = ols.y .- ols.X * ols.betas

coef(ols::OLS) = ols.betas

predict(ols::OLS) = ols.X * ols.betas

predict(ols::OLS, X::Array{Float64,2}) = X * ols.betas

