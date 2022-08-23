
@testset "asm2000 - Algorithm - Phone data" begin
    df = phones
    reg = createRegressionSetting(@formula(calls ~ year), df)
    result = asm2000(reg)
    outliers = result["outliers"]
    for i in 15:20
        @test i in outliers
    end
end