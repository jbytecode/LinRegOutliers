function py95ProcessEigenVector(v)
    eps = 0.0001
    n = length(v)
    k = 2.5
    c1 = Int(floor(n / 4.0))
    c2 = Int(floor(n / 4.0))
    ordered_coordinates = sortperm(v)
    a = zeros(Float64, n)
    b = zeros(Float64, n)
    for i in n:(-1):(n - c1)
        a[i] = v[i] / v[i - 1]
    end
    for i in 1:c2
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


function py95SuspectedObservations(setting::RegressionSetting)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    nhalf = Int(floor(n / 2.0))
    ols = lm(setting.formula, setting.data)
    resids = residuals(ols)
    H = hatmatrix(setting)
    s2 = sum(resids.^2.0) / (n - p)
    D = zeros(Float64, (n, n))
    E = zeros(Float64, (n, n))
    for i in 1:n
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


function jacknifedS(setting::RegressionSetting, omittedIndices::Array{Int,1})::Float64
    n, p = size(designMatrix(setting))
    indices = [i for i in 1:n if !(i in omittedIndices)]
    ols = lm(setting.formula, setting.data[indices,:])
    e = residuals(ols)
    s = sqrt(sum(e.^2.0) / (n - p - length(omittedIndices)))
    return s
end

function py95(setting::RegressionSetting)
    X = designMatrix(setting)
    Y = responseVector(setting)
    n, p = size(X)
    all_indices = collect(1:n)
    suspicious_sets = py95SuspectedObservations(setting)
    outlierset = Set{Int}()
    for aset in suspicious_sets
        clean_indices = setdiff(all_indices, aset)
        ols = lm(setting.formula, setting.data[clean_indices,:])
        betas = coef(ols)
        e = [Y[i] - sum(X[i,:] .* betas) for i in 1:n]
        jks = jacknifedS(setting, aset)
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