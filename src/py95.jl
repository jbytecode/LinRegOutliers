"""

    py95ProcessEigenVector(v)

Process eigen vectors of EDHDE matrix as defined in the Pena & Yohai (1995) algorithm.

# Arguments
- `v::Array{Float64, 1}`: Eigen vector of EDHDE matrix.

# References
Peña, Daniel, and Victor J. Yohai. "The detection of influential subsets in linear 
regression by using an influence matrix." Journal of the Royal Statistical Society: 
Series B (Methodological) 57.1 (1995): 145-156.
"""
function py95ProcessEigenVector(v::Array{Float64,1})
    eps = 0.0001
    n = length(v)
    k = 2.5
    c1 = Int(floor(n / 4.0))
    c2 = Int(floor(n / 4.0))
    ordered_coordinates = sortperm(v)
    a = zeros(Float64, n)
    b = zeros(Float64, n)
    @inbounds for i in n:(-1):(n - c1)
        a[i] = v[i] / v[i - 1]
    end
    @inbounds for i in 1:c2
        b[i] = v[i] / v[i + 1] 
    end
    set_of_a = filter(i -> abs(a[i]) > k, ordered_coordinates)
    set_of_b = filter(i -> abs(b[i]) > k, ordered_coordinates)
    outliersetJ = Int[]
    outliersetI = Int[]
    
    if length(set_of_a) == 0 || length(set_of_b) == 0
        return (nothing, nothing)
    end 
    i0 = set_of_b[1]
    j0 = set_of_a[1]
    if i0 > 1 
        outliersetI = ordered_coordinates[1:(j0 - 1)]
    end
    if j0 > 1
        outliersetJ = ordered_coordinates[n:(-1):(n - i0 + 1)]
    end
    (outliersetI, outliersetJ)
end

"""

    py95SuspectedObservations(setting)

Determine suspected observations (outliers) as defined in the Pena & Yohai (1995) algorithm.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# References
Peña, Daniel, and Victor J. Yohai. "The detection of influential subsets in linear 
regression by using an influence matrix." Journal of the Royal Statistical Society: 
Series B (Methodological) 57.1 (1995): 145-156.
"""
function py95SuspectedObservations(setting::RegressionSetting)
    X, y = @extractRegressionSetting setting
    return py95SuspectedObservations(X, y)
end



function py95SuspectedObservations(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)
    nhalf = Int(floor(n / 2.0))
    olsreg = ols(X, y)
    resids = residuals(olsreg)
    H = hatmatrix(X)
    s2 = sum(resids.^2.0) / (n - p)
    D = zeros(Float64, (n, n))
    E = zeros(Float64, (n, n))
    @inbounds for i in 1:n
        D[i, i] = 1 / (1 - H[i, i])
        E[i, i] = resids[i]
    end
    M = (1.0 / (p * s2)) * E * D * H * D * E
    eig = eigen(M)
    eig_values = eig.values
    eig_vectors = eig.vectors
    nonzero_eigen_indices = filter(i -> imag(eig_values[i]) == 0, 1:n)
    nonzero_eigen_vectors = eig_vectors[:, nonzero_eigen_indices]
    real_vectors = convert(Array{Float64,2}, nonzero_eigen_vectors) 
    s1, s2 = size(real_vectors)
    suspected_observation_sets = Set{Array{Int,1}}()
    for i in 1:s2
        set1, set2 = py95ProcessEigenVector(real_vectors[:,i])
        if !(set1 isa Nothing)
            if length(set1) < nhalf 
                push!(suspected_observation_sets, set1)
            end
        end
        if !(set2 isa Nothing)
            if length(set2) < nhalf
                push!(suspected_observation_sets, set2)
            end
        end
    end
    return suspected_observation_sets
end



"""

    jacknifedS(setting, omittedIndices)

Calculate Jacknife standard error in which the given indices are omitted from the data.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `omittedIndices::Array{Int, 1}`: Indices of omitted variables.

# References
Peña, Daniel, and Victor J. Yohai. "The detection of influential subsets in linear 
regression by using an influence matrix." Journal of the Royal Statistical Society: 
Series B (Methodological) 57.1 (1995): 145-156.
"""
function jacknifedS(setting::RegressionSetting, omittedIndices::Array{Int,1})::Float64
    X = designMatrix(setting)
    y = responseVector(setting)
    return jacknifedS(X, y, omittedIndices)
end

function jacknifedS(X::Array{Float64,2}, y::Array{Float64,1}, omittedIndices::Array{Int,1})::Float64
    n, p = size(X)
    indices = [i for i in 1:n if !(i in omittedIndices)]
    olsreg = ols(X[indices,:], y[indices])
    e = residuals(olsreg)
    s = sqrt(sum(e.^2.0) / (n - p - length(omittedIndices)))
    return s
end



"""

    py95(setting)

Perform the Pena & Yohai (1995) algorithm for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.

# Description 
The algorithm starts by constructing an influence matrix using results 
of an ordinary least squares estimate for a given model and data. In the second stage, 
the eigen structure of the influence matrix is examined for detecting suspected subsets of data.

# Output 
- `["outliers"]`: Array of indices of outliers
- `["suspected.sets"]`: Arrays of indices of observations for corresponding eigen value of the influence matrix.


# Examples
```julia-repl
julia> reg0001 = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk);
julia> py95(reg0001)
ict{Any,Any} with 2 entries:
  "outliers"       => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  "suspected.sets" => Set([[14, 13], [43, 54, 24, 38, 22], [6, 10], [14, 7, 8, 3, 10, 2, 5, 6, 1, 9, 4…
```

# References
Peña, Daniel, and Victor J. Yohai. "The detection of influential subsets in linear 
regression by using an influence matrix." Journal of the Royal Statistical Society: 
Series B (Methodological) 57.1 (1995): 145-156.
"""
function py95(setting::RegressionSetting)
    X = designMatrix(setting)
    y = responseVector(setting)
    return py95(X, y)
end


function py95(X::Array{Float64,2}, y::Array{Float64,1})
    n, p = size(X)
    all_indices = collect(1:n)
    suspicious_sets = py95SuspectedObservations(X, y)
    outlierset = Set{Int}()
    for aset in suspicious_sets
        clean_indices = setdiff(all_indices, aset)
        olsreg = ols(X[clean_indices,:], y[clean_indices])
        betas = coef(olsreg)
        e = [y[i] - sum(X[i,:] .* betas) for i in 1:n]
        jks = jacknifedS(X, y, aset)
        stds = e / jks
        outlier_indices = filter(i -> abs(stds[i]) > 2.5, 1:n)
        for element in outlier_indices
            push!(outlierset, element)
        end
    end
    result = Dict()
    result["suspected.sets"] = suspicious_sets
    result["outliers"] = sort(collect(outlierset))
    return result
end