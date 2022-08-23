@testset "Ordinary Least Squares" begin
    tol = 0.0001
    #  The model is exatly y = 5 + 5x
    var1 = Float64[1, 2, 3, 4, 5]
    X = hcat(ones(5), var1)
    betas = [5.0, 5.0]
    y = X * betas

    # OLS
    olsreg = ols(X, y)
    @test isapprox(coef(olsreg), betas, atol=tol)
    @test isapprox(residuals(olsreg), zeros(Float64, 5), atol=tol)
    @test isapprox(predict(olsreg), y, atol=tol)
end

@testset "Weighted Least Squares" begin
    tol = 0.0001
    n = 7
    #  The model is exatly y = 5 + 5x
    var1 = Float64[1, 2, 3, 4, 5, 6, 7]
    X = hcat(ones(n), var1)
    betas = [5.0, 5.0]
    y = X * betas
    y[n - 1] = 5000.0
    y[n] = 5000.0
    wts = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    # WLS
    olsreg = wls(X, y, wts)
    @test isapprox(coef(olsreg), betas, atol=tol)
    @test isapprox(residuals(olsreg)[1:(n - 2)], zeros(Float64, n - 2), atol=tol)
    @test isapprox(predict(olsreg)[1:(n - 2)], y[1:(n - 2)], atol=tol)
end
