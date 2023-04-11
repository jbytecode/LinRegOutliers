using LinRegOutliers

import LinRegOutliers.OrdinaryLeastSquares: ols, wls, coef, residuals
import Distributions: median

#sett = createRegressionSetting(@formula(calls ~ year), phones)
#sett = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)

function generatedata(
    n::Int,
    p::Int,
    contamination::Float64,
    direction::Symbol,
)::Tuple{Array{Float64,2},Array{Float64,1}}
    if !(direction in [:y, :x])
        error("- Direction must be :x or :y")
    end 
    on = ones(Float64, n)
    pxvars = p - 1 
    xvars = randn(n, pxvars)
    b = [5.0 for i = 1:pxvars]
    y = 5.0 .+ xvars * b + randn(n)
    totalcontamination = round(Int, contamination * n)
    if direction == :y
        y[1:totalcontamination] .= maximum(y) .+ abs.(rand(totalcontamination) * 5.0)
    elseif direction == :x
        xvars[1:totalcontamination, :] .=
            maximum(xvars) .+ abs.(rand(totalcontamination) * 5.0)
    end 
    return (hcat(on, xvars), y)
end



sett = createRegressionSetting(@formula(calls ~ year), phones)
x = designMatrix(sett)
y = responseVector(sett)
x, y = generatedata(1000, 25, 0.30, :x)
result = gwcga(x, y)