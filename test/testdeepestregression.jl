import LinRegOutliers: DataSets 

@testset "Deepest Regression" begin 

    @testset "Simple Data" begin 
        eps = 0.1

        n = 100000
        x1 = rand(n)
        x2 = rand(n)
        o = ones(Float64, n)
        e = randn(n)
        y = 15 .+ 10 .* x1 + 5 .* x2 + e
        X = hcat(x1, x2)

        result = deepestregression(X, y)

        betas = result["betas"]

        @test isapprox(betas[1], 15, atol = eps)
        @test isapprox(betas[2], 10, atol = eps)
        @test isapprox(betas[3], 5, atol = eps)
    end 

    @testset "Stackloss Data Example" begin 

        eps = 0.001

        setting = createRegressionSetting(
            @formula(stackloss ~ airflow + watertemp + acidcond), 
            DataSets.stackloss)
        
        result = deepestregression(setting)

        betas = result["betas"]

        @test isapprox(betas[1], -35.37610619, atol = eps)
        @test isapprox(betas[2], 0.82522124, atol = eps)
        @test isapprox(betas[3], 0.44247788, atol = eps)
        @test isapprox(betas[4], -0.07964602, atol = eps)

    end 
end 