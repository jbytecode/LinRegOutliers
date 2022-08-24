@testset "Chatterjee & Mächler (1997) cm97 with stackloss" begin
    df2 = stackloss
    reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
    regresult = cm97(reg)
    betas = regresult["betas"]
    betas_in_article = [-37.00, 0.84, 0.63, -0.11]
    @test isapprox(betas, betas_in_article, atol=0.01)
end