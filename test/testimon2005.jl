@testset "Imon 2005 - Algorithm" begin
    Random.seed!(12345)
    reg = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)
    result = imon2005(reg)
    @test result["outliers"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
end
