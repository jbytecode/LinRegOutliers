module LinRegOutliers

# using GLM
# using DataFrames
# using Distributions
# using Clustering
# using StatsBase
# using LinearAlgebra
# using Plots
# using Optim

# import functions from corresponding packages
import GLM: @formula, lm, FormulaTerm, ModelFrame, ModelMatrix, predict, coef, residuals
import DataFrames: DataFrame
import Distributions: TDist, Chisq, Normal, std, cov, median
import Clustering: Hclust, hclust, cutree
import StatsBase: quantile, standardize, ZScoreTransform, mean, sample
import LinearAlgebra: inv, det, diag, eigen, norm
import Plots: scatter, title!, xlabel!, ylabel!, hline!, vline!, RGB, plot, font, text
import Optim: optimize, NelderMead
import Combinatorics: combinations

# Basis
include("basis.jl")

# Predefined datasets used in outlier detection literature
include("data.jl")

# Regression diagnostics
include("diagnostics.jl")

# Ordinary least squares type and functions
# for fast regression tasks in outlier detection algorithms
include("ols.jl")

# Hadi & Simonoff (1993) algorithm
include("hs93.jl")

# Kianifard & Swallow (1989) algorithm
include("ks89.jl")

# Sebert et. al. (1998) algorithm
include("smr98.jl")

# Rousseeuw's Least Median of Squares estimator
include("lms.jl")

# Rousseeuw's Least Trimmed Squares estimator
include("lts.jl")

# Minimum Volume Ellipsoid estimator
# for robust covariance matrix
include("mve.jl")

# MVE - LTS plot for visual detection of regression outliers
include("mveltsplot.jl")

# Billor & Chatterjee & Hadi Algorithm for detecting outliers
include("bch.jl")

# Pena & Yohai (1995) algorithm
include("py95.jl")

# Satman (2013) algorithm
include("satman2013.jl")

# Satman (2015) algorithm
include("satman2015.jl")

# Least Absolute Deviations estimator
include("lad.jl")

# Least Trimmed Absolute Deviations estimator
include("lta.jl")

# Hadi (1992) detecting outliers in multivariate data
# This algorithm is not directly related with the regression
# however, detecting bad leverage points is same as detecting
# outliers in X-space
include("hadi1992.jl")

# Gray-scale images of distance matrices
include("dataimage.jl")

# Compact genetic algorithm
include("cga.jl")

# Genetic Algorithm
include("ga.jl")

# Modified and original Satman (2012) algorithms
include("gwlts.jl")

# RANSAC Algorithm
include("ransac.jl")

# All-in-one
include("summary.jl")

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
export phones, hbk, stackloss, weightloss, hs93randomdata, woodgravity

# Diagnostics
export dffit
export hatmatrix
export studentizedResiduals
export adjustedResiduals
export jacknifedS
export cooks
export mahalanobisSquaredMatrix

# Ordinary least squares
export OLS, ols, wls, residuals, predict

# Algorithms
export hs93, hs93initialset, hs93basicsubset
export ks89
export smr98, asm2000
export lms
export lts
export mve, mcd
export mveltsplot
export bch, bchplot
export py95, py95SuspectedObservations
export satman2013
export satman2015, dominates
export lad
export lta
export hadi1992
export dataimage
export gwcga, galts, ga, cga, RealChromosome
export detectOutliers
export ransac


end # module
