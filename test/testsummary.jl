@testset "Summary" begin
    smiley = " 😔 "
    methods = [
            "hs93",
            "ks89",
            "smr98",
            "lts"
        ]
    sett = LinRegOutliers.createRegressionSetting(@formula(calls ~ year), phones)
    result = LinRegOutliers.detectOutliers(sett, methods = methods)
    display(result)
    for i in 15:21
        @test result[:, "hs93"][i] == smiley
    end 

    for i in 18:20
        @test result[:, "ks89"][i] == smiley
    end 

    for i in 15:24
        @test result[:, "smr98"][i] == smiley
    end 

    for i in 15:21
        @test result[:, "lts"][i] == smiley
    end 
end