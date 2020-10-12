"""

    struct RegressionSetting
        formula::FormulaTerm
        data::DataFrame
    end

    Immutable data structure for a regression setting.

# Arguments
- `formula::FormulaTerm`: A formula object describes the linear regression model.
- `data::DataFrame`: DataFrame object holds the data

# Notes
    Implemented methods in this packages accepts linear models as RegressionSetting objects.
    This objects holds the model formula and the data used in regression estimations.

# Examples
```julia-repl
julia> setting = RegressionSetting(@formula(calls ~ year), phones)
RegressionSetting(calls ~ year, 24×2 DataFrame
│ Row │ year  │ calls   │
│     │ Int64 │ Float64 │
├─────┼───────┼─────────┤
│ 1   │ 50    │ 4.4     │
│ 2   │ 51    │ 4.7     │
│ 3   │ 52    │ 4.7     │
│ 4   │ 53    │ 5.9     │
│ 5   │ 54    │ 6.6     │
│ 6   │ 55    │ 7.3     │
│ 7   │ 56    │ 8.1     │
│ 8   │ 57    │ 8.8     │
│ 9   │ 58    │ 10.6    │
│ 10  │ 59    │ 12.0    │
⋮
│ 14  │ 63    │ 21.2    │
│ 15  │ 64    │ 119.0   │
│ 16  │ 65    │ 124.0   │
│ 17  │ 66    │ 142.0   │
│ 18  │ 67    │ 159.0   │
│ 19  │ 68    │ 182.0   │
│ 20  │ 69    │ 212.0   │
│ 21  │ 70    │ 43.0    │
│ 22  │ 71    │ 24.0    │
│ 23  │ 72    │ 27.0    │
│ 24  │ 73    │ 29.0    │)
```
"""
struct RegressionSetting
    formula::FormulaTerm
    data::DataFrame
end


"""

    createRegressionSetting(formula, data)

    Create a regression setting for a given formula and data

# Arguments
- `formula::FormulaTerm`: A formula object describes the linear regression model.
- `data::DataFrame`: DataFrame object holds the data

# Notes
    Implemented methods in this packages accepts linear models as RegressionSetting objects.
    This objects holds the model formula and the data used in regression estimations.
    createRegressionSetting is a helper function for creating RegressionSetting objects.

# Examples
```julia-repl
julia> setting = createRegressionSetting(@formula(calls ~ year), phones)
RegressionSetting(calls ~ year, 24×2 DataFrame
│ Row │ year  │ calls   │
│     │ Int64 │ Float64 │
├─────┼───────┼─────────┤
│ 1   │ 50    │ 4.4     │
│ 2   │ 51    │ 4.7     │
│ 3   │ 52    │ 4.7     │
│ 4   │ 53    │ 5.9     │
│ 5   │ 54    │ 6.6     │
│ 6   │ 55    │ 7.3     │
│ 7   │ 56    │ 8.1     │
│ 8   │ 57    │ 8.8     │
│ 9   │ 58    │ 10.6    │
│ 10  │ 59    │ 12.0    │
⋮
│ 14  │ 63    │ 21.2    │
│ 15  │ 64    │ 119.0   │
│ 16  │ 65    │ 124.0   │
│ 17  │ 66    │ 142.0   │
│ 18  │ 67    │ 159.0   │
│ 19  │ 68    │ 182.0   │
│ 20  │ 69    │ 212.0   │
│ 21  │ 70    │ 43.0    │
│ 22  │ 71    │ 24.0    │
│ 23  │ 72    │ 27.0    │
│ 24  │ 73    │ 29.0    │)
```
"""
function createRegressionSetting(formula::FormulaTerm, data::DataFrame)::RegressionSetting
    return RegressionSetting(formula, data)
end



"""

    designMatrix(setting)

    Return matrix of independent variables including the variable (ones) of the constanst term for a given regression setting.

# Arguments
- `setting::RegressionSetting`: A regression setting object.

# Notes
    Design matrix is a matrix which holds values of independent variables on its columns for a given linear regression model.

# Examples
```julia-repl
julia> setting = createRegressionSetting(@formula(calls ~ year), phones);
julia> designMatrix(setting)
24×2 Array{Float64,2}:
 1.0  50.0
 1.0  51.0
 1.0  52.0
 1.0  53.0
 1.0  54.0
 1.0  55.0
 1.0  56.0
 1.0  57.0
 1.0  58.0
 1.0  59.0
 1.0  60.0
 1.0  61.0
 1.0  62.0
 1.0  63.0
 1.0  64.0
 1.0  65.0
 1.0  66.0
 1.0  67.0
 1.0  68.0
 1.0  69.0
 1.0  70.0
 1.0  71.0
 1.0  72.0
 1.0  73.0
```
"""
function designMatrix(setting::RegressionSetting)::Array{Float64,2}
    mf = ModelFrame(setting.formula, setting.data)
    mm = ModelMatrix(mf)
    return convert(Array{Float64, 2}, mm.m)
end



"""

    responseVector(setting)

    Return vector of dependent variable of a given regression setting.

# Arguments
- `setting::RegressionSetting`: A regression setting object.


# Examples
```julia-repl
julia> setting = createRegressionSetting(@formula(calls ~ year), phones);
julia> responseVector(setting)
24-element Array{Float64,1}:
   4.4
   4.7
   4.7
   5.9
   6.6
   7.3
   8.1
   8.8
  10.6
  12.0
  13.5
  14.9
  16.1
  21.2
 119.0
 124.0
 142.0
 159.0
 182.0
 212.0
  43.0
  24.0
  27.0
  29.0
```
"""
function responseVector(setting::RegressionSetting)::Array{Float64,1}
    mf = ModelFrame(setting.formula, setting.data)
return convert(Array{Float64,1}, setting.data[:,mf.f.lhs.sym])
end


"""

    applyColumns(f, data)

    Apply function f to each columns of data.

# Arguments
- `f::Function`: A function that takes a one dimensional array as argument.
- `data::DataFrame`: A DataFrame object.

"""
function applyColumns(f::Function, data::DataFrame)
    return [f(col) for col = eachcol(data)]
end


"""

    applyColumns(f, data)

    Apply function f to each columns of data.

# Arguments
- `f::Function`: A function that takes a one dimensional array as argument.
- `data::Matrix`: A Matrix object.
"""
function applyColumns(f::Function, data::Matrix)
    return [f(col) for col = eachcol(data)]
end



"""

    find_minimum_nonzero(arr)

    Return minimum of numbers greater than zero.

# Arguments
- `arr::Array{Float64, 1}`: A function that takes a one dimensional array as argument.

# Example
```julia-repl
julia> find_minimum_nonzero([0.0, 0.0, 5.0, 1.0])
1.0
```
"""
function find_minimum_nonzero(arr::Array{Float64,1})
    return minimum(filter(x -> x > 0, arr))
end


"""
    covratio(setting, omittedIndex)

Apply covariance ratio diagnostic for a given regression setting and observation index.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `omittedIndex::Int`: Index of the omitted observation.

# Example
```julia-repl
julia> setting = createRegressionSetting(@formula(calls ~ year), phones);
julia> covratio(setting, 1)
1.2945913799871505
```
"""
function covratio(setting::RegressionSetting, omittedIndex::Int)
    X = designMatrix(setting)
    y = responseVector(setting)
    return covratio(X, y, omittedIndex)
end

function covratio(X::Array{Float64, 2}, y::Array{Float64, 1}, omittedIndex::Int)
    n, p = size(X)
    reg = ols(X, y)
    r = residuals(reg)
    s2 = sum(r .^ 2.0) / Float64(n - p)
    xxinv = inv(X'X)

    indices = filter(x -> x != omittedIndex, 1:n)
    
    Xomitted = X[indices,:]
    yomitted = y[indices]
    xxinvomitted = inv(Xomitted' * Xomitted)
    regomitted = ols(Xomitted, yomitted)
    resomitted = residuals(regomitted)
    s2omitted = sum(resomitted .^ 2.0) / Float64(n - p - 1)

    covrat = det(s2omitted * xxinvomitted) / det(s2 * xxinv)

    return covrat 
end