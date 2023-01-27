

    @testset "Satman(2012) (Csteps and GA based LTS) Algorithm - Phones data" begin
        Random.seed!(12345)
        epsilon = 10.0^(-3.0)
        df = phones
        reg = createRegressionSetting(@formula(calls ~ year), df)
        result = galts(reg)
        betas = result["betas"]
        @test abs(betas[1] - (-56.5219)) < epsilon
        @test abs(betas[2] - 1.16488) < epsilon
    end





