@testset "satman2013 - Algorithm - Hawkins & Bradu & Kass data" begin
    df = hbk
    reg = createRegressionSetting(@formula(y ~ x1 + x2 + x3), df)
    result = satman2013(reg)
    outliers = result["outliers"]
    for i = 1:10
        @test i in outliers
    end
end
