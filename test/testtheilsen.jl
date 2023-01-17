@testset "Theil-Sen for multiple regression" begin
    eps = 0.001

    x = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    y = [2.0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    mydata = DataFrame("x" => x, "y" => y)
    setting = createRegressionSetting(@formula(y ~ x), mydata)

    result = theilsen(setting, 2, nsamples = 1000)

    betas = result["betas"]

    @test isapprox(betas[1], 0.0, atol = eps)
    @test isapprox(betas[2], 2.0, atol = eps)
end
