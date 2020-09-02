module LinRegOutliers

# using GLM
# using DataFrames
# using Distributions
# using Clustering
# using StatsBase
# using LinearAlgebra
# using Plots
# using Optim

import GLM: @formula, lm, FormulaTerm, ModelFrame, ModelMatrix, predict, coef, residuals
import DataFrames: DataFrame
import Distributions: TDist, Chisq, Normal, std, cov, median
import Clustering: Hclust, hclust, cutree
import StatsBase: quantile, standardize, ZScoreTransform, mean, sample
import LinearAlgebra: inv, det, diag, eigen
import Plots: scatter, title!, xlabel!, ylabel!, hline!, vline!
import Optim: optimize, NelderMead
import Combinatorics: combinations

include("basis.jl")
include("data.jl")
include("diagnostics.jl")
include("hs93.jl")
include("ks89.jl")
include("smr98.jl")
include("lms.jl")
include("lts.jl")
include("mve.jl")
include("mveltsplot.jl")
include("bch.jl")
include("py95.jl")
include("satman2013.jl")
include("satman2015.jl")
include("lad.jl")
include("lta.jl")
include("hadi1992.jl")
include("dataimage.jl")

# Essentials from other packages
export @formula, DataFrame
export mean, quantile

# Basics 
export RegressionSetting
export createRegressionSetting
export designMatrix
export responseVector
export applyColumns
export find_minimum_nonzero

# Data
export phones, hbk

# Diagnostics 
export dffit
export hatmatrix
export studentizedResiduals
export adjustedResiduals
export jacknifedS
export cooks

# Algorithms
export hs93, hs93initialset, hs93basicsubset
export ks89
export smr98, asm2000
export lms
export lts 
export mve
export mveltsplot
export bch, bchplot
export py95, py95SuspectedObservations
export satman2013
export satman2015, dominates
export lad
export lta
export hadi1992
export dataimage

end # module
