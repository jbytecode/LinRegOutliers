using Test
using DataFrames
using Random
using LinearAlgebra

using LinRegOutliers
import Plots: RGBX

include("testbasis.jl")
include("testols.jl")
include("testdiagnostics.jl")
include("tesths93.jl")
include("testks89.jl")
include("testsmr98.jl")
include("testlms.jl")
include("testlts.jl")
include("testasm2000.jl")
include("testmvemcd.jl")
include("testbch.jl")
include("testpy95.jl")
include("testlad.jl")
include("testquantileregression.jl")
include("testga.jl")
include("testccf.jl")
include("testsatman2013.jl")
include("testsatman2015.jl")
include("testlta.jl")
include("testhadi92.jl")
include("testhadi94.jl")
include("testsatman2012.jl")
include("testransac.jl")
include("testatkinson94.jl")
include("testimon2005.jl")
include("testcm97.jl")
include("testbacon2000.jl")
include("testdataimage.jl")
include("testtheilsen.jl")
include("testsummary.jl")
include("testdeepestregression.jl")
include("testrobhatreg.jl")