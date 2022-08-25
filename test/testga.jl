@testset "GA optimizers" begin

    @testset "Compact Genetic Algorithm" begin
        Random.seed!(12345)
        function fcost(bits)
            return sum(bits)
        end
        result = cga(chsize = 10, costfunction = fcost, popsize = 100)
        @warn result
        for element in result
            @test (element == 0) || (element == 1)
        end
    end

    @testset "Floating-point Genetic Algorithm" begin
        Random.seed!(12345)
        fcost(genes) = (genes[1] - 3.14159265)^2 + (genes[2] - 2.71828)^2
        epsilon = 10.0^(-6.0)
        popsize = 30
        chsize = 2
        mins = [-100.0, -100.0]
        maxs = [100.0, 100.0]
        result = ga(popsize, chsize, fcost, mins, maxs, 0.90, 0.05, 1, 100)
        best::RealChromosome = result[1]

        @test length(result) == 30
        @test abs(best.genes[1] - 3.14159265) < epsilon
        @test abs(best.genes[2] - 2.71828) < epsilon
    end

end
