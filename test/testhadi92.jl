
@testset "Hadi 1992" begin

    @testset "Hadi 1992 - Algorithm - with several case" begin
        phones_matrix = hcat(phones[:, "calls"], phones[:, "year"])
        result = hadi1992(phones_matrix)
        outlier_indices = result["outliers"]
        for i = 15:20
            @test i in outlier_indices
        end

        hbk_matrix = hcat(hbk[:, "x1"], hbk[:, "x2"], hbk[:, "x3"])
        hbk_n, hbk_p = size(hbk_matrix)
        result = hadi1992(hbk_matrix)
        outlier_indices = result["outliers"]
        for i = 15:hbk_n
            @test !(i in outlier_indices)
        end
    end

    @testset "Hadi 1992 - Handle Singularity" begin 
        S = [1.0 0.95; 0.95 1.0]

        result = LinRegOutliers.Hadi92.hadi1992_handle_singularity(S)
        expectedresult = [10.256410256410234 -9.743589743589723; -9.743589743589723 10.256410256410234]

        @test  expectedresult == result
    end 

end

