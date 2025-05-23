@testset "OLS and WLS" begin

    @testset "Ordinary Least Squares" begin
        tol = 0.0001
        #  The model is exactly y = 5 + 5x
        var1 = Float64[1, 2, 3, 4, 5]
        X = hcat(ones(5), var1)
        betas = [5.0, 5.0]
        y = X * betas

        # OLS
        olsreg = ols(X, y)
        @test isapprox(coef(olsreg), betas, atol = tol)
        @test isapprox(residuals(olsreg), zeros(Float64, 5), atol = tol)
        @test isapprox(predict(olsreg), y, atol = tol)
    end

    @testset "OLS with setting" begin 
        eps = 0.001
        sett = createRegressionSetting(@formula(calls ~ year), phones)
        result = ols(sett)
        @test isapprox(result.betas[1], -260.0592463768119, atol = eps)
        @test isapprox(result.betas[2], 5.04147826086957, atol = eps)
    end 


    @testset "Weighted Least Squares" begin
        tol = 0.0001
        n = 7
        #  The model is exactly y = 5 + 5x
        var1 = Float64[1, 2, 3, 4, 5, 6, 7]
        X = hcat(ones(n), var1)
        betas = [5.0, 5.0]
        y = X * betas
        y[n-1] = 5000.0
        y[n] = 5000.0
        wts = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        # WLS
        olsreg = wls(X, y, wts)
        @test isapprox(coef(olsreg), betas, atol = tol)
        @test isapprox(residuals(olsreg)[1:(n-2)], zeros(Float64, n - 2), atol = tol)
        @test isapprox(predict(olsreg)[1:(n-2)], y[1:(n-2)], atol = tol)
    end

    @testset "WLS with setting and equal weights" begin 
        eps = 0.001
        sett = createRegressionSetting(@formula(calls ~ year), phones)
        result = wls(sett)
        @test isapprox(result.betas[1], -260.0592463768119, atol = eps)
        @test isapprox(result.betas[2],  5.04147826086957, atol = eps)
    end 

    @testset "WLS with setting and inequal weights" begin
        eps = 0.001
        w = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0
        ] 
        sett = createRegressionSetting(@formula(calls ~ year), phones)
        result = wls(sett, weights = w)
        @test isapprox(result.betas[1], -51.644554455445444, atol = eps)
        @test isapprox(result.betas[2],  1.0846534653465334, atol = eps)
    end 
end
