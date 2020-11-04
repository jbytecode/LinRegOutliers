[![Build Status](https://travis-ci.org/jbytecode/LinRegOutliers.svg?branch=master)](https://travis-ci.org/jbytecode/LinRegOutliers)

# LinRegOutliers

A Julia package for outlier detection in linear regression.

## Implemented Methods
- Ordinary Least Squares, Weighted Least Squares, Basic diagnostics
- Hadi & Simonoff (1993)
- Kianifard & Swallow (1989)
- Sebert & Montgomery & Rollier (1998)
- Least Median of Squares 
- Least Trimmed Squares 
- Minimum Volume Ellipsoid (MVE)
- MVE & LTS Plot 
- Billor & Chatterjee & Hadi (2006)
- Pena & Yohai (1995)
- Satman (2013)
- Satman (2015)
- Setan & Halim & Mohd (2000)
- Least Absolute Deviations (LAD)
- Least Trimmed Absolute Deviations (LTA)
- Hadi (1992)
- Marchette & Solka (2003) Data Images
- Satman's GA based LTS estimation (2012)
- Fischler & Bolles (1981) RANSAC Algorithm
- Minimum Covariance Determinant Estimator
- Imon (2005) Algorithm
- Barratt & Angeris & Boyd (2020) CCF algorithm
- Atkinson (1994) Forward Search Algorithm
- BACON Algorithm (Billor & Hadi & Velleman (2000))
- Hadi (1994) Algorithm
- Chatterjee & Mächler (1997)
- Summary


## Unimplemented Methods
- Depth based estimators (Regression depth, deepest regression, etc.)
- Theil & Sen estimator for mutliple regression


## Installation
```julia
julia> ]
(@v1.5) pkg> add LinRegOutliers
```

or

```julia
julia> using Pgk
julia> Pkg.add("LinRegOutliers")
```

then

```julia
julia> using LinRegOutliers
```

to make all the stuff be ready!


## Examples
We provide some examples [here](https://github.com/jbytecode/LinRegOutliers/blob/master/examples.md).
 
## Documentation
Please check out the reference manual [here](https://jbytecode.github.io/LinRegOutliers/docs/build/).

## News
- We implemented algorithm(X, y) style calls for all of the algorithms where X is the design matrix and y is the response vector. 
- We implemented ~25 outlier detection algorithms which covers a high percentage of the literature.


## Want to have contributions?
You are probably the right contributor

- If you have statistics background
- If you like Julia

However, the second condition is more important because an outlier detection algorithm is just an algorithm. Reading the implemented methods is enough to implement new ones. Please follow the issues. [Here is the a bunch of first shot introductions for new comers](https://github.com/jbytecode/LinRegOutliers/issues/3). Welcome and thank you in advance!


## Contact & Communication
- Please use issues for a new feature request or bug reports
- We are in #linregoutliers channel on [Julia Slack](http://julialang.slack.com/) for any discussion requires online chatting. 
