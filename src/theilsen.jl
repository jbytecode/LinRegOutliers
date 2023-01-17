module TheilSen 

export theilsen 

import ..Basis: RegressionSetting,  designMatrix, responseVector, @extractRegressionSetting
import ..OrdinaryLeastSquares: ols, coef 
import ..GA: ga
import Distributions: sample, mean

function theilsen(setting::RegressionSetting; nsamples::Int = 5000)
    X, y = @extractRegressionSetting setting
    return theilsen(X, y)
end

function theilsen(X::Array{Float64, 2}, y::Array{Float64, 1}; nsamples::Int = 5000)
    
    n, p = size(X)

    h = Int(floor((n + p + 1.0) / 2.0))
    mrange = (p+1):h

    allbetas = Array{Float64, 2}(undef, nsamples, p)

    for i in 1:nsamples
        m = rand(mrange)
        luckyindices = sample(1:n, p + 1 , replace = false)
        olsresult = ols(X[luckyindices, :], y[luckyindices])
        betas = coef(olsresult)
        allbetas[i, :] = betas
    end 

    return (multivariatemedian(allbetas), allbetas)
end 

function multivariatemedian(X::Array{Float64, 2})
    n, p = size(X)
   
    function dist(x::Vector{Float64}, y::Vector{Float64})
        return sum(abs.(x .- y))
    end

    function objective(candidate::Vector{Float64})
        val = sum(dist(candidate, X[i, :]) for i in 1:n)
        return val
    end 

    mins = map(i -> minimum(X[:, i]), 1:p)
    maxs = map(i -> maximum(X[:, i]), 1:p)

    @info "Mins Maxs" mins maxs 

    optresult = ga(
        100,            # popsize::Int,
        p,              # chsize::Int,
        objective,      # fcost::Function,
        mins,           # mins::Array{Float64,1},
        maxs,           # maxs::Array{Float64,1},
        1.0,            # pcross::Float64,
        0.1,            # pmutate::Float64,
        2,              # elitisim::Int,
        10,           # iterations::Int
    )

    return optresult[1].genes
end


end # end of module