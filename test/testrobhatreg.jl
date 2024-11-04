
@testset "Robust Hat Matrix based Robust Regression" verbose = true begin

    @testset "Random data" begin
        # Create simple data
        rng = MersenneTwister(12345)
        n = 50
        x = collect(1:n)
        e = randn(rng, n) .* 2.0
        y = 5 .+ 5 .* x .+ e

        # Contaminate some values 
        y[n] = y[n] * 2.0
        y[n-1] = y[n-1] * 2.0
        y[n-2] = y[n-2] * 2.0
        y[n-3] = y[n-3] * 2.0
        y[n-4] = y[n-4] * 2.0

        df = DataFrame(x=x, y=y)

        reg = createRegressionSetting(@formula(y ~ x), df)
        result = robhatreg(reg)

        betas = result["betas"]

        atol = 1.0

        @test isapprox(betas[1], 5.0, atol=atol)
        @test isapprox(betas[2], 5.0, atol=atol)
    end

    @testset "Phone data" begin
        df = phones
        reg = createRegressionSetting(@formula(calls ~ year), df)
        result = robhatreg(reg)

        betas = result["betas"]

        atol = 0.001

        @test isapprox(betas[1], -54.967349441923226, atol=atol)
        @test isapprox(betas[2], 1.1406353489513064, atol=atol)
    end

    @testset "Large Data" begin
        X = randn(10000, 10)
        y = randn(10000)

        result = robhatreg(X, y)

        betas = result["betas"]

        atol = 0.1

        for i in 1:10
            @test isapprox(betas[i], 0.0, atol=atol)
        end
    end

    @testset "Single Y outlier" begin
        @testset "LAD - Algorithm - Exact" begin
            df2 = DataFrame(
                x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                y=[2, 4, 6, 8, 10, 12, 14, 16, 18, 1000],
            )
            reg2 = createRegressionSetting(@formula(y ~ x), df2)
            result2 = lad(reg2)
            betas2 = result2["betas"]
            @test betas2[1] == 0.0
            @test betas2[2] == 2.0
        end
    end
end

