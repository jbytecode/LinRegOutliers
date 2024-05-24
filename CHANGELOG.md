# v0.11.4 (Upcoming Release)

- More explicit return types, drop `Dict` with `Dict{String, Any}` or `Dict{String, Vector}`
- Add `Julia v.1.10` to GitHub actions
- Reduce memory allocations in `lts()` and `lms()`
- Reduce memory allocations in `hadi92()` and `hadi94()`
- Reduce memory allocations in `hs93()`

# v0.11.3 

- Minor fixes

# v0.11.2

- Optional eps and maxiter parameters in iterateCSteps() in LTS.
- Replace `ols(X[indices, :], y[indices])` with `X[...] \ y[...]` in highly computational ones such like LTS, LMS, and LTA. 

# v0.11.1

- Concrete types of X and y changed to AbstractMatrix{Float64} and AbstractVector{Float64}
- Change function signatures from ::Function to ::F where {F <: Function}
- Increase test coverage

# v0.11.0

- Deepest Regression Estimator added.


# v0.10.2 

- mahalanobisSquaredBetweenPairs() return Union{Nothing, Matrix} depending on the determinant of the covariance matrix
- mahalanobisSquaredMatrix() returns Union{Nothing, Matrix} depending on the determinant of the covariance matrix
- import in DataImages fixed.
- Array{Float64, 1} is replaced by Vector{Float64}.
- Array{Float64, 2} is replaced by Matrix{Float64}.
- Use of try/catch reduced, many tries were depending on singularities.


# v0.10.1 

- Update compatibility with Clustering and StatsModels


# v0.10.0

- LMS returns betas rather than coefs
- PrecompileTools integration for faster loading of package
- Replace RGB{} to RGBX{} in plots

# v0.9.5 

- Adopt to SnoopPrecompile for better first-time-use experiment
  


# v0.9.4

- Convergency check for hs93()
- Add earlystop option for inexact version of lta().
- Remove cga() and cga based experimental algorithm.
- Fix deadlock in bacon.

# v0.9.3

- ga() is now faster.
- Implement dfbetas().
- Separate implementations of dffit() and dffits()
- Implement diagnose()
- Replace GLPK with HiGHS. When n > 10000, HiGHS based lad() is now approximately 6x faster.
- asm2000(), imon2005(), ks89(), bacon(), and smr98() now return regression coefficients calculated using the clean set of observations.
- lts() has a new optional parameter called earlystop and it is true by default. If the objective function does not change in a predefined number of iterations, the search is stopped.
- py95() returns vector of estimated regression coefficients using the clean observations.


# v0.9.2

- Increase test coverage 
- Add ols(setting) and wls(setting, weights = w) type method call where setting is a regression setting 
- Implement cooksoutliers() method for determining potential regression outliers using a cutoff value.
- Update documentation
- Implement Theil-Sen estimator for multiple regression


# v0.9.1

- Fix bchplot dependencies
- Update README with new instructions
- Satman2015 now returns more verbose output


# v0.9.0

- Add exact argument for LAD. If exact is true then the linear programming based exact solution is found. Otherwise, a GA based search is performed to yield approximate solutions. 
- Remove dependency of Plots.jl. If Plots.jl is installed and loaded manually, the functionality that uses Plot is autmatically loaded by Requires.jl. Affected functions are `dataimage`, `mveltsplot`, and `bchplot`.


# v0.8.19

- Update Satman(2013) algorithm


# v0.8.18 

- Add docs for Satman's (modified) GA based LTS estimation (2012)



# v0.8.17

- Remove dependency of StatsBase


# v0.8.16

- Quantile Regression implemented


# v0.8.15 

- Modularize dataimage()
- Grouped tests
- x === nothing style decisions replaces by isnothing(x)
  

# v0.8.14

- Update documentation
- Modularization  


# v0.8.13

- Removed some unused variables 
- Refactor code
- Update docs system
  
# v0.8.12

- Dependency entries updated in Project.toml
  

# v0.8.11

- asyncmap replaced with map in lta
- JuMP version increased
  

# v0.8.10

- Julia compatibality level is now 1.7
- Update JuMP and GLPK


# v0.8.9

- LAD (Least Absolute Deviations) is now exact and uses a linear programming based model
- Dependencies for JuMP and GLPK are added 
- Dependency for Optim removed

