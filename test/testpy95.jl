@testset "PY95 - Algorithm - Hawkins & Bradu & Kass data" begin
    df = hbk
    reg = createRegressionSetting(@formula(y ~ x1 + x2 + x3), df)
    result = py95(reg)
    outliers = result["outliers"]
    for i in 1:14
        @test i in outliers
    end
end