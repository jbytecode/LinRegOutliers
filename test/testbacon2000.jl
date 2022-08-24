@testset "BACON 2000 - Algorithm" begin
    df = stackloss
    reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
    result = bacon(reg, m=12)["outliers"]
    @test result == [1, 3, 4, 21]
end