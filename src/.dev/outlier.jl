@info "Loading StatsModels"
# using StatsModels
import StatsModels: @formula, FormulaTerm, ModelFrame, ModelMatrix

@info "Loading DataFrames"
# using DataFrames
import DataFrames: DataFrame

@info "Loading Distributions"
# using Distributions
import Distributions: TDist, Chisq, Normal, std, cov, median

@info "Loading Clustering"
# using Clustering
import Clustering: Hclust, hclust, cutree

@info "Loading StatsBase"
# using StatsBase
import StatsBase: quantile, standardize, ZScoreTransform, mean, sample

@info "Loading LinearAlgebra"
# using LinearAlgebra
import LinearAlgebra: inv, pinv, det, diag, eigen, norm, Diagonal

@info "Loading Plots"
# using Plots
import Plots: scatter, title!, xlabel!, ylabel!, hline!, vline!, RGB, plot, font, text

@info "Loading Optim"
# using Optim
import Optim: optimize, NelderMead

@info "Loading Combinatorics"
import Combinatorics: combinations

@info "Loading source"
include("../basis.jl")
include("../data.jl")
include("../diagnostics.jl")
include("../ols.jl")
include("../hs93.jl")
include("../ks89.jl")
include("../smr98.jl")
include("../asm2000.jl")
include("../lms.jl")
include("../lts.jl")
include("../mve.jl")
include("../mveltsplot.jl")
include("../bch.jl")
include("../py95.jl")
include("../satman2013.jl")
include("../satman2015.jl")
include("../lad.jl")
include("../lta.jl")
include("../hadi1992.jl")
include("../dataimage.jl")
include("../cga.jl")
include("../ga.jl")
include("../gwlts.jl")
include("../summary.jl")
include("../ransac.jl")
include("../atkinson94.jl")
include("../imon2005.jl")

println("ready")

