@testset "LTS" verbose = true begin

    @testset "LTS - Algorithm - Random data" begin
        # Create simple data
        rng = MersenneTwister(12345)
        n = 50
        x = collect(1:n)
        e = randn(rng, n) .* 2.0
        y = 5 .+ 5 .* x .+ e
        y[n] = y[n] * 2.0
        y[n-1] = y[n-1] * 2.0
        y[n-2] = y[n-2] * 2.0
        x[n-3] = x[n-3] * 2.0
        df = DataFrame(x = x, y = y)
        reg = createRegressionSetting(@formula(y ~ x), df)
        outset = lms(reg)["outliers"]
        @test 48 in outset
        @test 49 in outset
        @test 50 in outset
    end

    @testset "LTS - Algorithm - Phone data" begin
        df = phones
        reg = createRegressionSetting(@formula(calls ~ year), df)
        outset = lts(reg)["outliers"]
        @test 15 in outset
        @test 16 in outset
        @test 17 in outset
        @test 18 in outset
        @test 19 in outset
        @test 20 in outset
        @test 21 in outset
    end

    @testset "LTS - Stress test (n = 1000, p = 10)" begin
        n = 1000
        p = 10
        x = rand(Float64, n, p)
        y = rand(Float64, n)
        result = lts(x, y)
        @test true 
    end 

    @testset "LTS - Stress test (n = 10000, p = 10)" begin
        n = 10000
        p = 10
        x = rand(Float64, n, p)
        y = rand(Float64, n)
        result = lts(x, y)
        @test true 
    end 
end
