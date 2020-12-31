function makeColorColumn(indices::Array{Int,1}, n::Int)::Array{String,1}
    colors = Array{String}(undef, n)
    for i in 1:n
        if i in indices
            colors[i] = " 😔 "
        else
            colors[i] = ""
        end
    end
    return colors
end

function detectOutliers(setting::RegressionSetting; methods=[])
    
    X = designMatrix(setting)
    y = responseVector(setting)
    return detectOutliers(X, y, methods=methods)
    
end


function detectOutliers(X::Array{Float64,2}, y::Array{Float64,1}; methods=[])
    if length(methods) == 0
        methods = [
            "hs93",
            "ks89",
            "py95",
            "smr98",
            "lts",
            "sat13",
            "sat15",
            "asm20",
            "bch",
            "bacon",
            "imon2005"
            ]
    end

    n, p = size(X)
    num_algs = length(methods)

    outlier_matrix = DataFrame()

    for method in methods
        if method == "hs93"
            try
                result = hs93(X, y)["outliers"]
            catch
                result = Int[]
            end
        elseif method == "ks89"
            try
                result = ks89(X, y)["outliers"]
            catch
                result = Int[]
            end
        elseif method == "py95"
            result = py95(X, y)["outliers"]
        elseif method == "smr98"
            result = smr98(X, y)["outliers"]
        elseif method == "lts"
            try
                result = lts(X, y)["outliers"]
            catch
                result = Int[]
            end
        elseif method == "sat13"
            result = satman2013(X, y)["outliers"]
        elseif method == "sat15"
            result = satman2015(X, y)["outliers"]
        elseif method == "asm20"
            try
                result = asm2000(X, y)["outliers"]
            catch
                result = Int[]
            end
        elseif method == "bch"
            try
                result = bch(X, y)["outliers"]
            catch
                result = Int[]
            end
        elseif method == "bacon"
            try            
                _, p = size(X)
                result = bacon(X, y, m=p + 1)["outliers"]
            catch
                result = Int[]
            end
        elseif method == "imon2005"
            try            
                result = imon2005(X, y)["outliers"]
            catch
                result = Int[]
            end
        else
            @error "Method not found " method
            result = Int[]
        end
        outlier_matrix[:, method] = makeColorColumn(result, n)
    end
    
    
    return outlier_matrix
end