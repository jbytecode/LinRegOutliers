@testset "LTA - Algorithm - Phone data" begin
    eps = 0.0001
    df = phones
    reg = createRegressionSetting(@formula(calls ~ year), df)
    result = lta(reg, exact = true)
    betas = result["betas"]
    @test abs(betas[1] - -55.5) < eps
    @test abs(betas[2] - 1.15) < eps
end
