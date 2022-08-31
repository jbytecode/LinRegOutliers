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
