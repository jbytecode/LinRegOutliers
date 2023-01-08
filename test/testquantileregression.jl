@testset "Quantile Regression" begin

    eps = 0.001

    @testset "Quantile Regression - q = 0.5" begin
        income = [420.157651, 541.411707, 901.157457, 639.080229, 750.875606]
        foodexp = [255.839425, 310.958667, 485.680014, 402.997356, 495.560775]

        n = length(income)
        X = hcat(ones(Float64, n), income)

        result = quantileregression(X, foodexp, tau = 0.5)

        betas2 = result["betas"]
        @test abs(betas2[1] - 55.0716060) < eps
        @test abs(betas2[2] - 0.4778393) < eps
    end

    @testset "Quantile Regression - q = 0.25" begin
        income = [420.157651, 541.411707, 901.157457, 639.080229, 750.875606]
        foodexp = [255.839425, 310.958667, 485.680014, 402.997356, 495.560775]

        n = length(income)
        X = hcat(ones(Float64, n), income)

        result = quantileregression(X, foodexp, tau = 0.25)

        betas2 = result["betas"]
        @test abs(betas2[1] - 48.0057823) < eps
        @test abs(betas2[2] - 0.4856801) < eps
    end

    @testset "Quantile Regression - q = 0.95" begin
        income = [420.157651, 541.411707, 901.157457, 639.080229, 750.875606]
        foodexp = [255.839425, 310.958667, 485.680014, 402.997356, 495.560775]

        n = length(income)
        X = hcat(ones(Float64, n), income)

        result = quantileregression(X, foodexp, tau = 0.95)

        betas2 = result["betas"]
        @test abs(betas2[1] - (-48.7124077)) < eps
        @test abs(betas2[2] - 0.7248513) < eps
    end

end
