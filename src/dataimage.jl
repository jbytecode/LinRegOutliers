module DataImage

export dataimage

import ..Diagnostics:
    mahalanobisSquaredMatrix, euclideanDistances, mahalanobisSquaredBetweenPairs

import ..RGBX

"""

    dataimage(dataMatrix; distance = :)

Generate the Marchette & Solka (2003) data image for a given data matrix. 

# Arguments
- `dataMatrix::AbstractVector{Float64}`: Data matrix with dimensions n x p, where n is the number of observations and p is the number of variables.
- `distance::Symbol`: Optional argument for the distance function.

# Notes
    distance is :mahalanobis by default, for the Mahalanobis distances. 
    use 

        dataimage(mat, distance = :euclidean)
    
    to use Euclidean distances.
    
# Examples
```julia-repl
julia> x1 = hbk[:,"x1"];
julia> x2 = hbk[:,"x2"];
julia> x3 = hbk[:,"x3"];
julia> mat = hcat(x1, x2, x3);
julia> di = dataimage(mat, distance = :euclidean)
julia> Plots.plot(di)
```

# References
Marchette, David J., and Jeffrey L. Solka. "Using data images for outlier detection." 
Computational Statistics & Data Analysis 43.4 (2003): 541-552.

!!! warning "Dependencies"
    This method is enabled when the Plots package is installed and loaded.
    
"""
function dataimage(
    dataMatrix::AbstractMatrix{Float64};
    distance = :mahalanobis,
)::Matrix{RGBX{Float64}}
    d = nothing
    if distance == :mahalanobis
        
        d = mahalanobisSquaredBetweenPairs(dataMatrix)

        if isnothing(d)
            throw(ErrorException("Mahalanobis distances are not calculated"))
        end

    elseif distance == :euclidean
        d = euclideanDistances(dataMatrix)
    else
        @error "Distance function unknown: " distance
        @error "Using mahalanobis instead"
        d = mahalanobisSquaredBetweenPairs(dataMatrix)

        if isnothing(d)
            throw(ErrorException("Mahalanobis distances are not calculated"))
        end
        
    end
    colours = 1.0 .- d / maximum(d)
    n, _ = size(d)
    colormatrix = Array{RGBX{Float64},2}(undef, n, n)
    for i = 1:n
        for j = 1:n
            colormatrix[i, j] = RGBX(colours[i, j])
        end
    end
    return colormatrix
end


end # end of module DataImage
