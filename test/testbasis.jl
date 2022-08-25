@testset "Basis" begin

    @testset "Apply function to columns" begin
        ones_vector = ones(Float64, 10)
        zeros_vector = zeros(Float64, 10)
        mat = hcat(ones_vector, zeros_vector)

        @test applyColumns(sum, mat) == [10.0, 0.0]
        @test applyColumns(mean, mat) == [1.0, 0.0]
    end


    @testset "Basis - createRegressionSetting, designMatrix, responseVector" begin
        dataset = DataFrame(x = [1.0, 2, 3, 4, 5], y = [2.0, 4, 6, 8, 10])
        setting = createRegressionSetting(@formula(y ~ x), dataset)

        @test designMatrix(setting) == hcat(ones(5), dataset[:, "x"])
        @test responseVector(setting) == dataset[:, "y"]
        @test setting.formula == @formula(y ~ x)
    end

end
