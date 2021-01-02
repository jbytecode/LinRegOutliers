"""

    dominates(p1::Array, p2::Array)

Return true if each element in p1 is not less than the corresponding element in p2 and at least one element in p1 is bigger than the corresponding element in p2.

# Arguments
- `p1::Array`: Numeric array of n elements.
- `p2::Array`: Numeric array of n elements.

# Examples
```julia-repl
julia> dominates([1,2,3], [1,2,1])
true

julia> dominates([0,0,0,0], [1,0,0,0])
false
```

# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.

Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II." 
International conference on parallel problem solving from nature. Springer, Berlin, Heidelberg, 2000.
"""
function dominates(p1::Array, p2::Array)::Bool
    n = length(p1)
    notworse = count(i -> p1[i] < p2[i], 1:n)
    better   = count(i -> p1[i] > p2[i], 1:n)
    return (notworse == 0) && (better > 0)
end


"""

    ndsranks(data)

Sort multidimensional data usin non-dominated sorting algorithm.

# Arguments
- `data::DataFrame`: DataFrame of variables.


# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.

Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II." 
International conference on parallel problem solving from nature. Springer, Berlin, Heidelberg, 2000.
"""
function ndsranks(data::DataFrame)::Array{Int}
    mat = convert(Matrix, data)
    return ndsranks(mat)
end



"""

    ndsranks(data)

Sort multidimensional data usin non-dominated sorting algorithm.

# Arguments
- `data::Matrix`: n x k matrix of observations where n is number of observations and k is number of variables.

# Examples
```julia-repl
julia> datamat = convert(Matrix, hbk)
75×4 Array{Float64,2}:
 10.1  19.6  28.3   9.7
  9.5  20.5  28.9  10.1
 10.7  20.2  31.0  10.3
  9.9  21.5  31.7   9.5
 10.3  21.1  31.1  10.0
 10.8  20.4  29.2  10.0
 10.5  20.9  29.1  10.8
  9.9  19.6  28.8  10.3
  9.7  20.7  31.0   9.6
  9.3  19.7  30.3   9.9
 11.0  24.0  35.0  -0.2
 12.0  23.0  37.0  -0.4
  ⋮                
  2.8   3.0   2.9  -0.5
  2.0   0.7   2.7   0.6
  0.2   1.8   0.8  -0.9
  1.6   2.0   1.2  -0.7
  0.1   0.0   1.1   0.6
  2.0   0.6   0.3   0.2
  1.0   2.2   2.9   0.7
  2.2   2.5   2.3   0.2
  0.6   2.0   1.5  -0.2
  0.3   1.7   2.2   0.4
  0.0   2.2   1.6  -0.9
  0.3   0.4   2.6   0.2

julia> ndsranks(datamat)
75-element Array{Int64,1}:
 61
 61
 64
 61
 64
 62
 64
 61
 61
 61
 30
 23
  ⋮
 11
  7
  0
  2
  0
  0
 12
 14
  4
  1
  0
  0
```

# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.

Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II." 
International conference on parallel problem solving from nature. Springer, Berlin, Heidelberg, 2000.
"""
function ndsranks(data::Matrix)::Array{Int}
    n, p = size(data)
    ranks = zeros(Int, n)
    mat = convert(Matrix, data)
    for i in 1:n
        for j in 1:n
            if i != j 
                if dominates(mat[i,:], mat[j,:])
                    ranks[i] += 1
                end
            end
        end
    end
    return ranks
end


"""

    midlist(n::Int, p::Int)

Return p indices in the middle of 1:n.

# Arguments
- `n::Int`: Number of observations.
- `p::Int`: Number of elements in the middle of indices.

# Notes 
    If n is even and p is odd then p + 1 observation indices are returned.

# Examples
```julia-repl
julia> midlist(10,2)
2-element Array{Int64,1}:
 5
 6

julia> midlist(10,3)
4-element Array{Int64,1}:
 4
 5
 6
 7
```

# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.
"""
function midlist(n::Int, p::Int)::Array{Int, 1}
    midlist = []
    if (n - p) % 2 == 0
        start = ((n - p) / 2) + 1
        stop = start + p - 1
        midlist = collect(start:stop)
    else
        start = Int(floor((n - p) / 2)) + 1
        stop = start + p  
        midlist = collect(start:stop)
    end 
    return midlist
end


"""

    satman2013(setting)

Perform Satman (2015) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description 
The algorithm starts with sorting the design matrix using the Non-dominated sorting algorithm.
An initial basic subset is then constructed using the ranks obtained in previous stage. After many 
C-steps, observations with high standardized residuals are reported to be outliers.


# Output
- `["outliers]`": Array of indices of outliers.

# Examples
```julia-repl
julia> eg0001 = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk);
julia> satman2013(reg0001)
Dict{Any,Any} with 1 entry:
  "outliers" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 47]

```

# References
Satman, Mehmet Hakan. "A new algorithm for detecting outliers in linear regression." 
International Journal of statistics and Probability 2.3 (2013): 101.
"""
function satman2015(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return satman2015(X, y)
end


function satman2015(X::Array{Float64, 2}, y::Array{Float64, 1})
    n, p = size(X)
    h = Int(floor((n + p + 1.0) / 2.0))

    allindices = collect(1:n)

    ranks = ndsranks(X)
    ranks_ordering = sortperm(ranks)

    basic_center_indices = midlist(n, p)
    basic_subset_indices = ranks_ordering[basic_center_indices]

    meanvector = applyColumns(mean, X[basic_subset_indices,:])
    covmat = cov(X[basic_subset_indices,:])
    mhs = mahalanobisSquaredMatrix(X, meanvector = meanvector, covmatrix = covmat)
    if mhs isa Nothing
        md2 = zeros(Float64, n)
    else
        md2 = diag(mhs)
    end
    md = sqrt.(md2)
    sorted_indices = sortperm(md)
    best_h_indices = sorted_indices[1:h]

    crit, bestset = iterateCSteps(X, y, best_h_indices, h)

    olsreg = ols(X[bestset, :], y[bestset])
    betas = coef(olsreg)
    resids = y .- (X * betas)
    med_res = median(resids)
    standardized_resids = (resids .- med_res) / median(abs.(resids .- med_res))
    
    outlierset = filter(i -> abs(standardized_resids[i]) > 2.5, allindices)
    
    result = Dict()
    result["outliers"] = outlierset

    return result
end