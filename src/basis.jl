struct RegressionSetting
    formula::FormulaTerm
    data::DataFrame
end

function createRegressionSetting(formula::FormulaTerm, data::DataFrame)::RegressionSetting
    return RegressionSetting(formula, data)
end

function designMatrix(setting::RegressionSetting)::Array{Float64,2}
    mf = ModelFrame(setting.formula, setting.data)
    mm = ModelMatrix(mf)
    return mm.m
end

function responseVector(setting::RegressionSetting)::Array{Float64,1}
    mf = ModelFrame(setting.formula, setting.data)
    return setting.data[:,mf.f.lhs.sym]
end

function applyColumns(f::Function, data::DataFrame)
    return [f(col) for col = eachcol(data)]
end

function applyColumns(f::Function, data::Matrix)
    return [f(col) for col = eachcol(data)]
end

function find_minimum_nonzero(arr::Array{Float64,1})
    arr_sorted = sort(arr)
    minval = arr_sorted[length(arr)]
    for val in arr_sorted
        if val < minval && val > 0
            minval = val
        end
    end
    return minval
end
