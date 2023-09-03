@testset "Summary" begin
    smiley = " 😔 "
    methods = [
            "hs93",
            "ks89",
            "smr98",
            "lts",
            "sat13",
            "sat15",
            "asm20",
            "bch",
            "bacon",
            "imon2005",
            "unknown"
        ]
    sett = LinRegOutliers.createRegressionSetting(@formula(calls ~ year), phones)
    result = LinRegOutliers.detectOutliers(sett, methods = methods)
    
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