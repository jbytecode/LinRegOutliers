@testset "Satman(2012)" begin

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


    @testset "gwcga -  Modified (Satman, 2012) - Algorithm - phone data" begin
        df = phones
        reg = createRegressionSetting(@formula(calls ~ year), df)
        result = gwcga(reg)
        clean_indices = result["clean.subset"]
        for i = 15:20
            @test !(i in clean_indices)
        end
    end

end
