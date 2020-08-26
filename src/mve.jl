function enlargesubset(initialsubset, data::DataFrame, dataMatrix::Matrix, h::Int)
    n, p = size(dataMatrix)
    indices = collect(1:n)
    basicsubset = copy(initialsubset)
    while length(basicsubset) < h
        meanvector = applyColumns(mean, data[basicsubset,:])
        covmatrix = cov(dataMatrix[basicsubset, :])
        md2mat = mahalabonisSquaredMatrix(data, meanvector=meanvector, covmatrix=covmatrix)
        md2 = diag(md2mat)
        md2sortedindex = sortperm(md2)
        basicsubset = md2sortedindex[1:(length(basicsubset) + 1)]
    end
    return basicsubset
end

function mve(data::DataFrame; alpha=0.05)
    dataMatrix = convert(Matrix, data)
    n, p = size(dataMatrix)
    chisquared = Chisq(p)
    chisqcrit = quantile(chisquared, 1 - alpha)
    c = sqrt(chisqcrit)
    h = Int(floor((n + p + 1.0) / 2.0)) 
    indices = collect(1:n)
    k = p + 1
    mingoal = Inf
    bestinitialsubset = []
    besthsubset = []
    maxiter = minimum([p * 500, 3000])
    initialsubset = []
    hsubset = []
    for iter in 1:maxiter
        goal = Inf
        try
            initialsubset = sample(indices, k, replace=false)
            hsubset = enlargesubset(initialsubset, data, dataMatrix, h) 
            covmatrix = cov(dataMatrix[hsubset, :])
            meanvector = applyColumns(mean, data[hsubset, :])
            md2mat = mahalabonisSquaredMatrix(data, meanvector=meanvector, covmatrix=covmatrix)
            DJ = sqrt(sort(diag(md2mat))[h])
            goal = (DJ / c)^p * det(covmatrix)
        catch e
            # Possibly singularity
        end
        if goal < mingoal
            mingoal = goal
            bestinitialsubset = initialsubset
            besthsubset = hsubset
        end
    end
    meanvector = applyColumns(mean, data[besthsubset, :])
    covariancematrix = cov(dataMatrix[besthsubset, :])
    md2 = diag(mahalabonisSquaredMatrix(data, meanvector=meanvector, covmatrix=covariancematrix))
    outlierset = filter(x -> md2[x] > chisqcrit, 1:n)
    result = Dict()
    result["goal"] = mingoal
    result["best.subset"] = sort(besthsubset)
    result["robust.location"] = meanvector
    result["robust.covariance"] = covariancematrix
    result["squared.mahalanobis"] = md2 
    result["outliers"] = outlierset
    return result 
end