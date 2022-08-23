@testset "RANSAC (1981) Algorithm - Paper example" begin
    Random.seed!(12345)
    df = DataFrame(y=[0,1,2,3,3,4,10], x=[0,1,2,2,3,4,2])
    reg = createRegressionSetting(@formula(y ~ x), df)
    result = ransac(reg, t=0.8, w=0.85)["outliers"]
    @test result == [7]
end