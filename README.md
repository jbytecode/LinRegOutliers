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


```julia
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> lms(reg)
Dict{Any,Any} with 6 entries:
  "stdres"    => [2.42593, 1.62705, 0.550525, 0.584612, 0.155943, -0.272726, -0.608843, -1.03751, -0.448118, -0.228929  …  93.4182, 96.9692, 112.552, 127.209, 147.419, 174.108…
  "S"         => 1.08048
  "outliers"  => [14, 15, 16, 17, 18, 19, 20, 21]
  "objective" => 0.43276
  "coef"      => [-56.3796, 1.16317]
  "crit"      => 2.5 
```


```julia
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);

julia> lts(reg)
Dict{Any,Any} with 6 entries:
  "betas"            => [-56.5219, 1.16488]
  "S"                => 1.10918
  "hsubset"          => [11, 10, 5, 6, 23, 12, 13, 9, 24, 7, 3, 4, 8]
  "outliers"         => [14, 15, 16, 17, 18, 19, 20, 21]
  "scaled.residuals" => [2.41447, 1.63472, 0.584504, 0.61617, 0.197052, -0.222066, -0.551027, -0.970146, -0.397538, -0.185558  …  91.0312, 94.4889, 109.667, 123.943, 143.629, …
  "objective"        => 3.43133
```  

![detectOutliersImage](https://github.com/jbytecode/jbytecode/blob/master/images/detectoutliers.png)
