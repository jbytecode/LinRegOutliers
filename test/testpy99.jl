@testset "Peña & Yohai 1999" begin
    @testset "Phone data" begin
        reg = createRegressionSetting(@formula(calls ~ year), phones)
        result = py99(reg)

        @test all(index -> index in result["outliers"], 14:21)
        @test length(result["betas"]) == 2
        @test result["scale"] > 0.0
        @test result["iterations"] <= 100
    end

    @testset "Matrix interface" begin
        X = hcat(ones(30), collect(1.0:30.0))
        y = 3.0 .+ 2.0 .* X[:, 2]
        y[29:30] .+= 100.0

        result = py99(X, y)

        @test 29 in result["outliers"]
        @test 30 in result["outliers"]
        @test result["betas"] ≈ [3.0, 2.0] atol = 1e-8
    end
end
