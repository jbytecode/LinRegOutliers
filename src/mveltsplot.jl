"""
    mveltsplot(setting; alpha = 0.05, showplot = true)

Generate MVE - LTS plot for visual detecting of regression outliers.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `alpha::Float64`: Probability for quantiles of Chi-Squared statistic.
- `showplot::Bool`: Whether a plot is shown or only return statistics.

# References
Van Aelst, Stefan, and Peter Rousseeuw. "Minimum volume ellipsoid." Wiley 
Interdisciplinary Reviews: Computational Statistics 1.1 (2009): 71-82.
"""
function mveltsplot(setting::RegressionSetting; alpha=0.05, showplot=true)
    ltsresult = lts(setting)
    mveresult = mve(setting.data)

    n, p = size(setting.data)
    indices = collect(1:n)

    chidist = Chisq(p)
    chicrit = sqrt(quantile(chidist, 1.0 - alpha))

    scaledresiduals = ltsresult["scaled.residuals"]
    robdistances = sqrt.(mveresult["squared.mahalanobis"])
    

    regularpoints = filter(i -> abs(scaledresiduals[i]) < 2.5 && robdistances[i] < chicrit, indices)
    outliers = filter(i -> scaledresiduals[i] > 2.5 && robdistances[i] < chicrit, indices)
    leverage = filter(i -> abs(scaledresiduals[i]) < 2.5 && robdistances[i] > chicrit, indices)
    outlierandleveragepoints = filter(i -> abs(scaledresiduals[i]) > 2.5 && robdistances[i] > chicrit, indices)

    scplot = nothing
    if showplot
        scplot = scatter(robdistances, 
            scaledresiduals, 
            legend=false, 
            series_annotations=text.(1:n, :bottom),
            tickfont=font(10), guidefont=font(10), labelfont=font(10)
            )
        title!("MVE - LTS Plot")
        xlabel!("Robust distances")
        ylabel!("Scaled residuals")
        hline!([-2.5, 2.5])
        vline!([chicrit])
    end

    result = Dict()
    result["plot"] = scplot
    result["robust.distances"] = robdistances
    result["scaled.residuals"] = scaledresiduals
    result["chi.squared"] = chicrit^2
    result["regular.points"] = regularpoints
    result["outlier.points"] = outliers
    result["leverage.points"] = leverage 
    result["outlier.and.leverage.points"] = outlierandleveragepoints
    return result
end