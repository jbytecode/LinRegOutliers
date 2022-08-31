@testset "Kianifard & Swallow 1989 - Algorithm" begin
    df = phones
    reg = createRegressionSetting(@formula(calls ~ year), df)
    result = ks89(reg, alpha = 0.1)
    outset = result["outliers"]
    @test 15 in outset
    @test 16 in outset
    @test 17 in outset
    @test 18 in outset
    @test 19 in outset
    @test 20 in outset

    df2 = stackloss
    reg2 = createRegressionSetting(
        @formula(stackloss ~ airflow + watertemp + acidcond),
        stackloss,
    )
    result2 = ks89(reg2)
    outset2 = result2["outliers"]
    @test 4 in outset2
    @test 21 in outset2
end
