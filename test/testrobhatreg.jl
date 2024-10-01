

@testset "Robust Hat Matrix based Robust Regression" begin
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





