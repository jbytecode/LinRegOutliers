module LinRegOutliers


# import functions from corresponding packages
import StatsModels: @formula, FormulaTerm, ModelFrame, ModelMatrix
import DataFrames: DataFrame
import Distributions: TDist, Chisq, std, cov, median
import Clustering: Hclust, hclust, cutree
import StatsBase: quantile, standardize, ZScoreTransform, mean, sample
import LinearAlgebra: inv, pinv, det, diag, eigen, norm, rank, qr, Diagonal, UpperTriangular, ColumnNorm
import Plots: scatter, title!, xlabel!, ylabel!, hline!, vline!, RGB, plot, font, text
import Combinatorics: combinations
import JuMP
import GLPK

# Basis
include("basis.jl")
import .Basis: RegressionSetting, createRegressionSetting, @extractRegressionSetting, applyColumns, find_minimum_nonzero, designMatrix, responseVector
export  RegressionSetting
export  createRegressionSetting
export  designMatrix
export  responseVector
export  applyColumns
export  find_minimum_nonzero
export  @extractRegressionSetting


# Predefined datasets used in outlier detection literature
include("data.jl")

# Ordinary least squares type and functions
# for fast regression tasks in outlier detection algorithms
include("ols.jl")
import .OrdinaryLeastSquares: OLS, ols, wls, residuals, predict, coef

# Regression diagnostics
include("diagnostics.jl")
import .Diagnostics: dffit, dfbeta, hatmatrix, studentizedResiduals, adjustedResiduals, jacknifedS, cooks, mahalanobisSquaredMatrix, covratio, hadimeasure


# Hadi & Simonoff (1993) algorithm
include("hs93.jl")

# Kianifard & Swallow (1989) algorithm
include("ks89.jl")

# Sebert et. al. (1998) algorithm
include("smr98.jl")
import .SMR98: majona, smr98

# Rousseeuw's Least Median of Squares estimator
include("lms.jl")

# Rousseeuw's Least Trimmed Squares estimator
include("lts.jl")
import .LTS: lts 


# asm (2000) algorithm
include("asm2000.jl")
import .ASM2000: asm2000


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
import .Satman2013: satman2013

# Satman (2015) algorithm
include("satman2015.jl")
import .Satman2015: satman2015, dominates 


# Least Absolute Deviations estimator
include("lad.jl")

# Least Trimmed Absolute Deviations estimator
include("lta.jl")

# Hadi (1992) detecting outliers in multivariate data
# This algorithm is not directly related with the regression
# however, detecting bad leverage points is same as detecting
# outliers in X-space
include("hadi1992.jl")

# Hadi (1994) algorithm
include("hadi1994.jl")

# Gray-scale images of distance matrices
include("dataimage.jl")

# Compact genetic algorithm
include("cga.jl")

# Genetic Algorithm
include("ga.jl")
import .GA: ga, RealChromosome 

# Modified and original Satman (2012) algorithms
include("gwlts.jl")
import .GALTS: gwcga, galts 

# RANSAC Algorithm
include("ransac.jl")


# CCF formulation and heuristic; Barratt, Angeris, and Boyd (2020)
include("ccf.jl")

# Atkinson94 Algorithm
include("atkinson94.jl")
include("atkinsonstalactiteplot.jl")

# BACON 2000 Algorithm
include("bacon.jl")

# Imon 2005 Algorithm
include("imon2005.jl")

#  Chatterjee & Mächler (1997) Algorithm
include("cm97.jl")

# All-in-one
include("summary.jl")

# Essentials from other packages
export @formula, DataFrame
export mean, quantile



# Data
export phones, hbk, stackloss
export weightloss, hs93randomdata, woodgravity
export hills, softdrinkdelivery

# Diagnostics
export dffit, dfbeta
export hatmatrix
export studentizedResiduals
export adjustedResiduals
export jacknifedS
export cooks
export mahalanobisSquaredMatrix
export covratio
export hadimeasure

# Ordinary least squares
export OLS, ols, wls, residuals, predict, coef

# Algorithms
export hs93, hs93initialset, hs93basicsubset
export ks89
export smr98
export asm2000
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
export hadi1994
export dataimage
export gwcga, galts, ga, cga, RealChromosome
export detectOutliers
export ransac
export ccf
export imon2005
export atkinson94, atkinsonstalactiteplot, generate_stalactite_plot
export bacon
export cm97

end # module
