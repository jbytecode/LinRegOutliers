@testset "LAD" begin

    @testset "LAD - Algorithm" begin
        eps = 0.0001
        df = phones
        reg = createRegressionSetting(@formula(calls ~ year), df)
        result = lad(reg)
        betas = result["betas"]
        @test betas[1] < 0
        @test betas[2] > 0

        df2 = DataFrame(
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 1000],
        )
        reg2 = createRegressionSetting(@formula(y ~ x), df2)
        result2 = lad(reg2)
        betas2 = result2["betas"]
        @test abs(betas2[1] - 0) < eps
        @test abs(betas2[2] - 2) < eps
    end

    @testset "LAD with (X, y)" begin
        eps = 0.0001
        df2 = DataFrame(
            x = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            y = Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 1000],
        )
        n = length(df2[!, "x"])
        X = hcat(ones(Float64, n), df2[!, "x"])
        y = df2[!, "y"]
        result2 = lad(X, y)
        betas2 = result2["betas"]
        @test abs(betas2[1] - 0) < eps
        @test abs(betas2[2] - 2) < eps
    end

end
