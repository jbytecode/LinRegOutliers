[![Build Status](https://travis-ci.org/jbytecode/LinRegOutliers.svg?branch=master)](https://travis-ci.org/jbytecode/LinRegOutliers) [![DOI](https://joss.theoj.org/papers/10.21105/joss.02892/status.svg)](https://doi.org/10.21105/joss.02892)
[![Doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbytecode.github.io/LinRegOutliers/dev/)
[![codecov](https://codecov.io/gh/jbytecode/LinRegOutliers/branch/master/graph/badge.svg?token=DM4XXML78A)](https://codecov.io/gh/jbytecode/LinRegOutliers)

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
- Quantile Regression Parameter Estimation (quantileregression)
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
- Depth based estimators (Regression depth, deepest regression, etc.) See [#13](https://github.com/jbytecode/LinRegOutliers/issues/13) for the related issue.
- Pena & Yohai (1999). See [#25](https://github.com/jbytecode/LinRegOutliers/issues/25) for the related issue.
- Theil & Sen estimator for multiple regression


## Installation

```LinRegOutliers``` can be installed using the ```Julia``` REPL.  

```julia
julia> ]
(@v1.8) pkg> add LinRegOutliers
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
Please check out the reference manual [here](https://jbytecode.github.io/LinRegOutliers/).

## News
- We implemented ~25 outlier detection algorithms which covers a high percentage of the literature.
- Visit the [CHANGELOG.md](https://github.com/jbytecode/LinRegOutliers/blob/master/CHANGELOG.md) for the log of latest changes.

## Contributions
You are probably the right contributor

- If you have statistics background
- If you like Julia

However, the second condition is more important because an outlier detection algorithm is just an algorithm. Reading the implemented methods is enough to implement new ones. Please follow the issues. [Here is the a bunch of first shot introductions for new comers](https://github.com/jbytecode/LinRegOutliers/issues/3). Welcome and thank you in advance!


## Citation
Please refer our original paper if you use the package in your research using

```
Satman et al., (2021). LinRegOutliers: A Julia package for detecting outliers in linear regression. Journal of Open Source Software, 6(57), 2892, https://doi.org/10.21105/joss.02892
```

or the bibtex entry

```
@article{Satman2021,
  doi = {10.21105/joss.02892},
  url = {https://doi.org/10.21105/joss.02892},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {57},
  pages = {2892},
  author = {Mehmet Hakan Satman and Shreesh Adiga and Guillermo Angeris and Emre Akadal},
  title = {LinRegOutliers: A Julia package for detecting outliers in linear regression},
  journal = {Journal of Open Source Software}
}
```


## Contact & Communication
- Please use issues for a new feature request or bug reports.
- We are in #linregoutliers channel on [Julia Slack](http://julialang.slack.com/) for any discussion requires online chatting. 
