function hadi1992_handle_singularity(S::Array{Float64,2})::Array{Float64,2}
    p, _ = size(S)
    eigen_structure = eigen(S)
    values = eigen_structure.values
    vectors = eigen_structure.vectors
    lambda_s = find_minimum_nonzero(values)
    W = zeros(Float64, p, p)
    for i in 1:p
        W[i, i] = 1 / max(values[i], lambda_s) 
    end
    newS = vectors * W * vectors
    return newS
end

function hadi1992(multivariateData::Array{Float64,2}; alpha=0.05)
    n, p = size(multivariateData)
    h = Int(floor((n + p + 1.0) / 2.0))
    chi_50_quantile = quantile(Chisq(p), 0.50)
    critical_quantile = quantile(Chisq(p), 1 - alpha)
    allindices = collect(1:n)

    # Step 0
    meds = coordinatwisemedians(multivariateData)
    Sm = (1.0 / (n - 1.0)) * (multivariateData .- meds')' * (multivariateData .- meds')
    mah0 = diag(mahalabonisSquaredMatrix(multivariateData, meanvector=meds, covmatrix=Sm))
    ordering_indices_mah0 = sortperm(mah0)
    best_indices_mah0 = ordering_indices_mah0[1:h]
    starting_data = multivariateData[best_indices_mah0, :]

    Cv = coordinatwisemedians(starting_data)
    Sv = (1.0 / (h - 1.0)) * (starting_data .- Cv')' * (starting_data .- Cv')
    mah1 = diag(mahalabonisSquaredMatrix(multivariateData, meanvector=Cv, covmatrix=Sv))
    ordering_indices_mah1 = sortperm(mah1)

    r = p + 1
    basic_subset_indices = []
    basic_subset = []
    sorted_mah1 = []

    while r < n
        cnpr = 1 + (r / (n - p))^2.0
        basic_subset_indices = ordering_indices_mah1[1:r]
        basic_subset = multivariateData[basic_subset_indices, :]
        Cb = applyColumns(mean, basic_subset)
        Sb = cov(basic_subset)

        r += 1
        cfactor = cnpr * sqrt(sort(mah1)[h]) / chi_50_quantile
        if det(cfactor * Sb) == 0
            @info "singular Sb case"
            newSb = hadi1992_handle_singularity(cfactor * Sb) 
            mah1 = diag(mahalabonisSquaredMatrix(multivariateData, meanvector=Cb, covmatrix=newSb))
            ordering_indices_mah1 = sortperm(mah1)
            basic_subset_indices = ordering_indices_mah1[1:r]
        else
            mah1 = diag(mahalabonisSquaredMatrix(multivariateData, meanvector=Cb, covmatrix=(cfactor * Sb)))
            ordering_indices_mah1 = sortperm(mah1)
            basic_subset_indices = ordering_indices_mah1[1:r]
        end
        
        sorted_mah1 = sort(mah1)
        if sorted_mah1[r] >= critical_quantile
            break 
        end
    end

    outlierset = setdiff(allindices, basic_subset_indices)

    result = Dict()
    result["outliers"] = sort(outlierset)
    result["criticial.chi.squared"] = critical_quantile
    result["rth.robust.distance"] = sorted_mah1[r - 1]
    return result
end