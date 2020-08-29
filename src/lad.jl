function lad(setting::RegressionSetting; starting_betas=nothing)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)

    if starting_betas isa Nothing
        starting_betas = zeros(Float64, p)
    end

    function goal(betas::Array{Float64,1})::Float64
        sum(abs.(Y .- X * betas))
    end

    optim_result = optimize(goal, starting_betas, NelderMead())
    betas = optim_result.minimizer 
    residuals = Y .- X * betas

    result = Dict()
    result["betas"] = optim_result.minimizer
    result["residuals"] = residuals

    return result
end