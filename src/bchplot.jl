module BCHPlot 


export bchplot


import ..BCH: bch
import ..Basis: RegressionSetting, designMatrix, responseVector


"""

    bchplot(setting::RegressionSetting; alpha=0.05, maxiter=1000, epsilon=0.00001)

Perform the Billor & Chatterjee & Hadi (2006) algorithm and generates outlier plot 
for the given regression setting.

# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `alpha::Float64`: Optional argument of the probability of rejecting the null hypothesis.
- `maxiter::Int`: Maximum number of iterations for calculating iterative weighted least squares estimates.
- `epsilon::Float64`: Accuracy for determining convergency.


# References
Billor, Nedret, Samprit Chatterjee, and Ali S. Hadi. "A re-weighted least squares method 
for robust regression estimation." American journal of mathematical and management sciences 26.3-4 (2006): 229-252.

!!! warning "Dependencies"
    This method is enabled when the Plots package is installed and loaded.
        
"""
function bchplot(
    setting::RegressionSetting;
    alpha = 0.05,
    maxiter = 1000,
    epsilon = 0.00001,
)
    X = designMatrix(setting)
    y = responseVector(setting)
    return bchplot(X, y, alpha = alpha, maxiter = maxiter, epsilon = epsilon)
end

function bchplot(
    Xdesign::Array{Float64,2},
    y::Array{Float64,1};
    alpha = 0.05,
    maxiter = 1000,
    epsilon = 0.00001,
)
    result = bch(Xdesign, y, alpha = alpha, maxiter = maxiter, epsilon = epsilon)
    squared_normalized_residuals = result["squared.normalized.residuals"]
    squared_normalized_robust_distances = result["squared.normalized.robust.distances"]
    n = length(squared_normalized_robust_distances)
    scplot = scatter(
        squared_normalized_robust_distances,
        squared_normalized_residuals,
        legend = false,
        series_annotations = text.(1:n, :bottom),
        tickfont = font(10),
        guidefont = font(10),
        labelfont = font(10),
    )
    title!("Billor & Chatterjee & Hadi Plot")
    xlabel!("Squared Normalized Robust Distances")
    ylabel!("Squared Normalized Residuals")
end

end # end of module