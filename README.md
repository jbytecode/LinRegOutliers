[![Build Status](https://travis-ci.org/jbytecode/LinRegOutliers.svg?branch=master)](https://travis-ci.org/jbytecode/LinRegOutliers)

# LinRegOutliers

A Julia package for outlier detection in linear regression.

## Implemented Methods
- Basic diagnostics
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
- Satman (2012) (fast CGA modified)
- Summary

## Example

```julia
julia> using LinRegOutliers
julia> # Regression setting for Hawkins & Bradu & Kass data
julia> reg = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)
julia> smr98(reg)
14-element Array{Int64,1}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
```


```julia
julia> py95(reg)["outliers"]
14-element Array{Int64,1}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14

```

