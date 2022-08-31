@testset "Atkinson 1994 - Algorithm" begin
    df = stackloss
    reg = createRegressionSetting(
        @formula(stackloss ~ airflow + watertemp + acidcond),
        stackloss,
    )
    result = atkinson94(reg)
    @test result["outliers"] == [1, 3, 4, 21]
end
