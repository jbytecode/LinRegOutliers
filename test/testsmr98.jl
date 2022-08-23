@testset "Sebert & Montgomery & Rollier 1998 - Algorithm" begin
    # Create simple data
    rng = MersenneTwister(12345)
    n = 50
    x = collect(1:n)
    e = randn(rng, n) .* 2.0
    y = 5 .+ 5 .* x .+ e
    y[n] = y[n] * 2.0
    y[n - 1] = y[n - 1] * 2.0
    y[n - 2] = y[n - 2] * 2.0
    df = DataFrame(x=x, y=y)
    reg = createRegressionSetting(@formula(y ~ x), df)
    outset = smr98(reg)["outliers"]
    @test 49 in outset
    @test 50 in outset
end