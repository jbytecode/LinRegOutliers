module DeepestRegression

export deepestregression

import ..Basis:
	RegressionSetting, @extractRegressionSetting, designMatrix, responseVector

using mrfDepth_jll: mrfDepth_jll


"""
	deepestregression(setting; maxit = 1000)

Estimate Deepest Regression paramaters.


# Arguments
- `setting::RegressionSetting`: RegressionSetting object with a formula and dataset.
- `maxit`: Maximum number of iterations

# Description
Estimates Deepest Regression Estimator coefficients.

# References
Van Aelst S., Rousseeuw P.J., Hubert M., Struyf A. (2002). The
deepest regression method. Journal of Multivariate Analysis,
81, 138-166.


# Output
- `betas`: Vector of regression coefficients estimated.
"""
function deepestregression(setting::RegressionSetting; maxit::Int = 10000)
	X = designMatrix(setting)
	y = responseVector(setting)
	if all(x -> isone(x), X[:, 1])
		X = X[:, 2:end]
	end
	return deepestregression(X, y, maxit = maxit)
end

function deepestregression(X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; maxit::Int = 10000)::AbstractVector{Float64}
	drdata = hcat(X, y)
	n, p = size(drdata)
	n = Int32(n)
	p = Int32(p)
	betas = zeros(Float64, p)
	maxit = Int32(maxit)
	iter = Int32(1)
	MDEPAPPR = Int32(p)
	ccall((:sweepmedres_, mrfDepth_jll.libmrfDepth),
		Cint,
		(Ref{Float64},           # X
			Ref{Int32},          # n
			Ref{Int32},          # np
			Ref{Float64},        # betas
			Ref{Cint},           # maxit 
			Ref{Cint},           # iter
			Ref{Cint},            # MDEPAPPR
		), drdata, n, p, betas, maxit, iter, MDEPAPPR)

	return vcat(betas[end], betas[1:(end-1)])
end


end # end of module DeepestRegression
