module Basis

export RegressionSetting
export createRegressionSetting
export @extractRegressionSetting
export applyColumns, applyColumns!
export find_minimum_nonzero


import DataFrames: DataFrame
import StatsModels: @formula, FormulaTerm, ModelFrame, ModelMatrix
import Distributions: mean, std

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
24×2 Matrix{Float64}:
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
function designMatrix(setting::RegressionSetting)::AbstractMatrix{Float64}
    mf = ModelFrame(setting.formula, setting.data)
    mm = ModelMatrix(mf)
    return convert(Matrix{Float64}, mm.m)
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
24-element Vector{Float64}:
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
function responseVector(setting::RegressionSetting)::AbstractVector{Float64}
    mf = ModelFrame(setting.formula, setting.data)
    return convert(Vector{Float64}, setting.data[:, mf.f.lhs.sym])
end



"""

    @extractRegressionSetting setting

    Return a tuple of design matrix and response vector for a given regression setting. 

# Arguments
- `setting::RegressionSetting`: A regression setting object.


# Examples
```julia-repl
julia> setting = createRegressionSetting(@formula(calls ~ year), phones);
julia> X, y = @extractRegressionSetting setting;
julia> size(X)
 (24,2)
julia> y
24-element Vector{Float64}:
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
macro extractRegressionSetting(setting)
    return esc(:(designMatrix($setting), responseVector($setting)))
end



"""

    applyColumns(f, data)

    Apply function f to each columns of data.

# Arguments
- `f <: Function`: A function that takes a one dimensional array as argument.
- `data::DataFrame`: A DataFrame object.

"""
function applyColumns(f::F, data::DataFrame) where {F <: Function}
    return [f(col) for col in eachcol(data)]
end


"""

    applyColumns(f, data)

    Apply function f to each columns of data.

# Arguments
- `f <: Function`: A function that takes a one dimensional array as argument.
- `data::AbstractMatrix`: A Matrix object.
"""
function applyColumns(f::F, data::AbstractMatrix{Float64}) where {F <: Function}
    return [f(col) for col in eachcol(data)]
end


function applyColumns!(target::Vector, f::F, data::AbstractMatrix) where {F <: Function}
    for i in 1:size(data, 2)
        target[i] = f(data[:, i])
    end
    return target
end



"""

    find_minimum_nonzero(arr)

    Return minimum of numbers greater than zero.

# Arguments
- `arr::AbstractVector{Float64}`: A function that takes a one dimensional array as argument.

# Example
```julia-repl
julia> find_minimum_nonzero([0.0, 0.0, 5.0, 1.0])
1.0
```
"""
function find_minimum_nonzero(arr::AbstractVector{Float64})
    return minimum(filter(x -> x > 0, arr))
end


function zstandardize(v::AbstractVector{Float64})::AbstractVector{Float64}
    return (v .- mean(v)) ./ std(v)
end

end # End of module Basis
