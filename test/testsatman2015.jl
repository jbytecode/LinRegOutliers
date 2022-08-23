@testset "satman2015 - Algorithm - Phone data" begin
    @test dominates([1,2,4], [1,2,3])
    @test dominates([1,2,3], [0,1,2])
    @test !dominates([1,2,3], [1,2,3])
    @test !dominates([1,2,3], [1,2,4])

    df = phones
    reg = createRegressionSetting(@formula(calls ~ year), df)
    result = satman2015(reg)
    outliers = result["outliers"]
    for i in 15:20
        @test i in outliers
    end
end