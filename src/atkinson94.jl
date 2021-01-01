"""
        atkinson94(setting, iters, crit)

Runs the Atkinson94 algorithm to find out outliers using LMS method.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `iters::Int`: Number of random samples.
- `crit::Float64`: Critical value for residuals

# Description 
The algorithm randomly selects initial basic subsets and performs a very robust method, e.g `lms`
to enlarge the basic subset. In each iteration of forward search, the best objective value and parameter 
estimates are stored. These values are also used in Atkinson's Stalactite Plot for a visual investigation of 
outliers. See `atkinsonstalactiteplot`.

# Output
- `["optimum_index"]`: The iteration number in which the minimum objective is obtained
- `["residuals_matrix"]`: Matrix of residuals obtained in each iteration
- `["outliers"]`: Array of indices of detected outliers
- `["objective"]`: Minimum objective value
- `["coef"]`: Estimated regression coefficients
- `["crit"]`: Critical value given by the user.


# Examples
```julia-repl
julia> reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
julia> atkinson94(reg)
Dict{Any,Any} with 6 entries:
  "optimum_index"    => 10
  "residuals_matrix" => [0.0286208 0.0620609 … 0.0796249 0.0; 0.0397778 0.120547 … 0.118437 0.0397778; … ; 1.21133 1.80846 … 0.690327 4.14366; 1.61977 0.971592 … 0.616204 3.58098]
  "outliers"         => [1, 3, 4, 21]
  "objective"        => 0.799134
  "coef"             => [-38.3133, 0.745659, 0.432794, 0.0104587]
  "crit"             => 3.0

```
# References
Atkinson, Anthony C. "Fast very robust methods for the detection of multiple outliers."
Journal of the American Statistical Association 89.428 (1994): 1329-1339.
"""
function atkinson94(setting::RegressionSetting; iters=nothing, crit=3.0)
    X, y = @extractRegressionSetting setting
    return atkinson94(X, y, iters=iters, crit=crit)
end

function atkinson94(X::Array{Float64,2}, y::Array{Float64,1}; iters=nothing, crit=3.0)
    n, p = size(X)

    # the median index
    h = Int(floor((n + p + 1) / 2.0))

    if iters === nothing
        iters = minimum([500 * p, 3000])
    end

    bestobjective = Inf
    bestparameters = zeros(Float64, p)
    bestres = zeros(Float64, n, n)
    bestindex = 0
    indices = collect(1:n)

    # store all the sigma values across all iterations
    sigmas = zeros(Float64, iters, n)

    for iter in 1:iters
        m_subset_indices = sample(indices, p, replace=false)

        # stores the n - p sigma values of a forward run
        sigma_tilde = zeros(Float64, n)

        # stores the (n - p) * n residuals during the forward run
        studentized_residuals = zeros(Float64, n, n)
        copy_parameters = false

        @inbounds for m = p:n
            olsreg = ols(X[m_subset_indices,:], y[m_subset_indices])
            betas = coef(olsreg)
            e = (y .- X * betas)
            r = e.^2

            # sigma is the median of the residuals
            sigma_tilde[m] = sqrt(sort(r)[h])
            XXinv = pinv(X[m_subset_indices, :]'X[m_subset_indices, :])

            # scale the residual according to whether it belongs to m_subset_indices or not
            @inbounds for index in 1:n
                if index in m_subset_indices
                    h_i = X[index, :]' * XXinv * X[index, :]
                    studentized_residuals[m, index] = abs(e[index]) / (sigma_tilde[m] * sqrt(abs(1 - h_i)))
                else
                    d_i = X[index, :]' * XXinv * X[index, :]
                    studentized_residuals[m, index] = abs(e[index]) / (sigma_tilde[m] * sqrt(1 + d_i))
                    r[index] = (e[index]^2) / (1 + d_i)
                end
            end

            # increase the size of subset by 1
            ordered_residue_indices = sortperm(r)
            if m != n
                m_subset_indices = ordered_residue_indices[1:m + 1]
            end

            # if sigma was lowest, record the parameters
            if sigma_tilde[m] < bestobjective
                bestobjective = sigma_tilde[m]
                bestparameters = betas
                bestindex = m
                copy_parameters = true
        end
        end
#         sigmas[iter, :] = sigma_tilde

        # copy the entire n^2 residuals for the forward search to generate stalactite plot
        if copy_parameters
            bestres = copy(studentized_residuals)
        end
    end
#     sigma_bar = mean(sigmas, dims=1)
#     for m = p:n
#         bestres[m, :] = bestres[m, :] .* sigma_bar[m]
#     end
    d = Dict()
#     d["sigma_bar"] = sigma_bar
#     d["sigmas"] = sigmas[:, p:end]
    d["coef"] = bestparameters
    d["objective"] = bestobjective
    d["optimum_index"] = bestindex
    d["residuals_matrix"] = bestres[p:end, :]
    d["crit"] = crit
    d["outliers"] = filter(i -> bestres[bestindex, i] > crit, 1:n)
    return d
end
