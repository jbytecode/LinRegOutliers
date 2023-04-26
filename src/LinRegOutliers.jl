module LinRegOutliers

using Requires

# After the module is loaded, we check if Plots is installed and loaded.
# If Plots is installed and loaded, we load the corresponding modules.
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
  
        import .Plots: RGBX

        include("mveltsplot.jl")
        include("dataimage.jl")
        include("bchplot.jl")

        import .MVELTSPlot: mveltsplot
        import .DataImage: dataimage
        import .BCHPlot: bchplot
        
        export mveltsplot, dataimage, bchplot, RGBX

    end
end

# Basis
include("basis.jl")
import .Basis:
    RegressionSetting,
    createRegressionSetting,
    @extractRegressionSetting,
    applyColumns,
    find_minimum_nonzero,
    designMatrix,
    responseVector
export RegressionSetting
export createRegressionSetting
export designMatrix
export responseVector
export applyColumns
export find_minimum_nonzero
export @extractRegressionSetting



# Hooke-Jeeves algorithm
include("hj.jl")
# The function hj() is not exported.

# Genetic Algorithm
include("ga.jl")
import .GA: ga, RealChromosome


# Predefined datasets used in outlier detection literature
include("data.jl")
import .DataSets: phones, hbk, stackloss
import .DataSets: weightloss, hs93randomdata, woodgravity
import .DataSets: hills, softdrinkdelivery, animals

# Ordinary least squares type and functions
# for fast regression tasks in outlier detection algorithms
include("ols.jl")
import .OrdinaryLeastSquares: OLS, ols, wls, residuals, predict, coef

# Regression diagnostics
include("diagnostics.jl")
import .Diagnostics:
    dffit,
    dffits,
    dfbeta,
    dfbetas,
    hatmatrix,
    studentizedResiduals,
    adjustedResiduals,
    jacknifedS,
    cooks,
    cooksoutliers,
    mahalanobisSquaredMatrix,
    covratio,
    hadimeasure,
    diagnose


# Hadi & Simonoff (1993) algorithm
include("hs93.jl")
import .HS93: hs93, hs93initialset, hs93basicsubset

# Kianifard & Swallow (1989) algorithm
include("ks89.jl")
import .KS89: ks89

# Sebert et. al. (1998) algorithm
include("smr98.jl")
import .SMR98: majona, smr98

# Rousseeuw's Least Median of Squares estimator
include("lms.jl")
import .LMS: lms


# Rousseeuw's Least Trimmed Squares estimator
include("lts.jl")
import .LTS: lts


# asm (2000) algorithm
include("asm2000.jl")
import .ASM2000: asm2000


# Minimum Volume Ellipsoid estimator
# for robust covariance matrix
include("mve.jl")
import .MVE: mve, mcd

# Moved into grahhics.jl
# MVE - LTS plot for visual detection of regression outliers
#include("mveltsplot.jl")
#import .MVELTSPlot: mveltsplot

# Billor & Chatterjee & Hadi Algorithm for detecting outliers
include("bch.jl")
import .BCH: bch

# Pena & Yohai (1995) algorithm
include("py95.jl")
import .PY95: py95, py95SuspectedObservations

# Satman (2013) algorithm
include("satman2013.jl")
import .Satman2013: satman2013

# Satman (2015) algorithm
include("satman2015.jl")
import .Satman2015: satman2015, dominates


# Least Absolute Deviations estimator
include("lad.jl")
import .LAD: lad

# Quantile Regression Estimator
include("quantileregression.jl")
import .QuantileRegression: quantileregression

# Least Trimmed Absolute Deviations estimator
include("lta.jl")
import .LTA: lta



# Hadi (1992) detecting outliers in multivariate data
# This algorithm is not directly related with the regression
# however, detecting bad leverage points is same as detecting
# outliers in X-space
include("hadi1992.jl")
import .Hadi92: hadi1992

# Hadi (1994) algorithm
include("hadi1994.jl")
import .Hadi94: hadi1994


# Moved into graphics.jl
# Gray-scale images of distance matrices
#include("dataimage.jl")
#import .DataImage: dataimage


# Modified and original Satman (2012) algorithms
include("gwlts.jl")
import .GALTS: galts

# RANSAC Algorithm
include("ransac.jl")
import .Ransac: ransac


# CCF formulation and heuristic; Barratt, Angeris, and Boyd (2020)
include("ccf.jl")
import .CCF: ccf


# Atkinson94 Algorithm
include("atkinson94.jl")
import .Atkinson94: atkinson94

include("atkinsonstalactiteplot.jl")
import .AtkinsonPlot: atkinsonstalactiteplot, generate_stalactite_plot


# BACON 2000 Algorithm
include("bacon.jl")
import .Bacon: bacon

# Imon 2005 Algorithm
include("imon2005.jl")
import .Imon2005: imon2005

#  Chatterjee & Mächler (1997) Algorithm
include("cm97.jl")
import .CM97: cm97

# Theil-Sen Estimator
include("theilsen.jl")
import .TheilSen: theilsen

# All-in-one
include("summary.jl")
import .Summary: detectOutliers


# Essentials from other packages
import StatsModels: @formula
import Distributions: mean, quantile
import DataFrames: DataFrame
export @formula, DataFrame
export mean, quantile



# Data
export phones, hbk, stackloss
export weightloss, hs93randomdata, woodgravity
export hills, softdrinkdelivery, animals

# Diagnostics
export dffit, dffits, dfbeta, dfbetas
export hatmatrix
export studentizedResiduals
export adjustedResiduals
export jacknifedS
export cooks
export cooksoutliers
export mahalanobisSquaredMatrix
export covratio
export hadimeasure
export diagnose


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
export bch
export py95, py95SuspectedObservations
export satman2013
export satman2015, dominates
export lad
export quantileregression
export lta
export hadi1992
export hadi1994
export galts, ga, RealChromosome
export detectOutliers
export ransac
export ccf
export imon2005
export atkinson94, atkinsonstalactiteplot, generate_stalactite_plot
export bacon
export cm97
export theilsen


# Snoop-Precompile 
include("precompile/precompile.jl")


end # module
