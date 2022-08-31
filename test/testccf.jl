@testset "ccf - Algorithm - Phone data" begin
    eps = 0.0001
    df = phones
    reg = createRegressionSetting(@formula(calls ~ year), df)
    result = ccf(reg)
    outliers = result["outliers"]
    for i = 15:20
        @test i in outliers
    end

    @test all(isapprox.(result["betas"], [-63.4816, 1.30406], atol = eps, rtol = 0.0))
end
