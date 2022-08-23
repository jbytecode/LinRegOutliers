using Test
using DataFrames
using Random
using LinearAlgebra

using LinRegOutliers

#include("testdiagnostics.jl")
#include("testbasis.jl")
#include("testols.jl")
#include("tesths93.jl")
#include("testks89.jl")
#include("testsmr98.jl")
#include("testlms.jl")
#include("testlts.jl")
#include("testasm2000.jl")
#include("testmvemcd.jl")
#include("testbch.jl")
#include("testpy95.jl")
#include("testlad.jl")
#include("testga.jl")
#include("testccf.jl")
#include("testsatman2013.jl")
#include("testsatman2015.jl")
#include("testlta.jl")
#include("testhadi92.jl)
#include("testhadi94.jl)
include("testsatman2012.jl")
include("testransac.jl")





@testset "Atkinson 1994 - Algorithm" begin
    df = stackloss
    reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
    result = atkinson94(reg)
    @test result["outliers"] == [1, 3, 4, 21]
end


@testset "Imon 2005 - Algorithm" begin
    Random.seed!(12345)
    reg = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)
    result = imon2005(reg)
    @test result["outliers"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
end

@testset "BACON 2000 - Algorithm" begin
    df = stackloss
    reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
    result = bacon(reg, m=12)["outliers"]
    @test result == [1, 3, 4, 21]
end

@testset "Chatterjee & Mächler (1997) cm97 with stackloss" begin
    df2 = stackloss
    reg = createRegressionSetting(@formula(stackloss ~ airflow + watertemp + acidcond), stackloss)
    regresult = cm97(reg)
    betas = regresult["betas"]
    betas_in_article = [-37.00, 0.84, 0.63, -0.11]
    @test isapprox(betas, betas_in_article, atol=0.01)
end

