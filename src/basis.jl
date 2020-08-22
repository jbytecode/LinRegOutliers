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
    return setting.data[:, mf.f.lhs.sym]
end