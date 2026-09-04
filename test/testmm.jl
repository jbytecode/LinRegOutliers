@testset "Yohai 1987 MM-estimator" verbose = true begin

    @testset "Phone data" begin
        Random.seed!(12345)
        reg = createRegressionSetting(@formula(calls ~ year), phones)
        result = mm(reg)

        betas = result["betas"]
        @test isapprox(betas[1], -52.4, atol = 7.5)
        @test isapprox(betas[2], 1.1, atol = 0.3)
        @test 15 in result["outliers"]
        @test 16 in result["outliers"]
        @test 17 in result["outliers"]
        @test 18 in result["outliers"]
        @test 19 in result["outliers"]
        @test 20 in result["outliers"]
        @test length(result["scaled.residuals"]) == 24
        @test result["S"] > 0.0
    end

    @testset "Matrix interface with LTS start" begin
        Random.seed!(12345)
        X = hcat(ones(40), collect(1.0:40.0))
        y = -4.0 .+ 1.5 .* X[:, 2]
        y[38:40] .+= 80.0

        result = mm(X, y, initial = :lts)

        @test isapprox(result["betas"][1], -4.0, atol = 0.25)
        @test isapprox(result["betas"][2], 1.5, atol = 0.05)
        @test 38 in result["outliers"]
        @test 39 in result["outliers"]
        @test 40 in result["outliers"]
    end

    @testset "Exact fit property" begin
        X = hcat(ones(10), collect(1.0:10.0))
        y = 2.0 .+ 3.0 .* X[:, 2]
        result = mm(X, y, initial = [2.0, 3.0])

        @test result["S"] == 0.0
        @test result["betas"] ≈ [2.0, 3.0] atol = 1e-12
        @test isempty(result["outliers"])
    end
end
