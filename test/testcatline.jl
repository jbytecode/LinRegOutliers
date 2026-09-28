@testset "Catline" begin
    phone_setting = createRegressionSetting(@formula(calls ~ year), phones)
    phone_result = catline(phone_setting)
    n = length(phones.calls)

    @test length(phone_result["betas"]) == 2
    @test length(phone_result["residuals"]) == n
    @test phone_result["betas"] ≈ [-57.525, 1.1875]
    @test phone_result["residuals"] ≈ phones.calls .-
                                      [ones(n) Float64.(phones.year)] *
                                      phone_result["betas"]

    stars_result = catline(createRegressionSetting(@formula(log_light ~ log_te), starscyg))
    @test length(stars_result["residuals"]) == 47
    @test argmax(stars_result["residuals"]) == 34

    exact_x = collect(1.0:9.0)
    exact_y = 3.0 .+ 2.0 .* exact_x
    exact_result = catline([ones(9) exact_x], exact_y)
    @test exact_result["betas"] ≈ [3.0, 2.0]
    @test exact_result["residuals"] ≈ zeros(9)

    @test_throws ArgumentError catline(ones(4, 3), ones(4))
    @test_throws ArgumentError catline(ones(2, 2), ones(2))
    @test_throws ArgumentError catline([zeros(4) collect(1.0:4.0)], ones(4))
    @test_throws ArgumentError catline(ones(4, 2), ones(4))
end
