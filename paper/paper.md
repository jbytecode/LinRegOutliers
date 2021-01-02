---
title: 'LinRegOutliers: A Julia package for detecting outliers in linear regression'
tags:
  - Julia
  - linear regression
  - outlier detection
  - robust statistics
authors:
  - name: Mehmet Hakan Satman
    orcid: 0000-0002-9402-1982
    affiliation: 1
  - name: Shreesh Adiga
    orcid: 0000-0002-1818-6961
    affiliation: 2
  - name: Guillermo Angeris
    orcid: 0000-0002-4950-3990
    affiliation: 3
  - name: Emre Akadal
    orcid: 0000-0001-6817-0127 
    affiliation: 4
affiliations:
 - name: Department of Econometrics, Istanbul University, Istanbul, Turkey
   index: 1
 - name: Department of Electronics and Communication Engineering, RV College of Engineering, Bengaluru, India
   index: 2
 - name: Department of Electrical Engineering, Stanford University, Stanford, California, USA
   index: 3
 - name: Department of Informatics, Istanbul University, Istanbul, Turkey
   index: 4

date: 26 November 2020
bibliography: paper.bib
---

# Summary

`LinRegOutliers` is a Julia package that implements a number of outlier detection algorithms for linear regression. The package also implements robust covariance matrix estimation and graphing functions which can be used to visualize the regression residuals and distances between observations, with many possible metrics (*e.g.*, the Euclidean or Mahalanobis distances with either given or estimated covariance matrices). Our package covers a significant portion of the literature on model fitting with outliers and allows users to quickly try many different methods with reasonable default settings, while also providing a good starting framework for researchers who may want to extend the package with novel methods.


# State of the field
In linear regression, we are given a number of data points (say, $n$) where each data point is represented by a vector $x_i$, with $p$ entries, and a dependent variable that corresponds to each of these data points, represented by the scalar $y_i$, for $i=1, 2, \dots, n$. We then seek to find the linear model which best describes the data (up to some error term, $\epsilon_i$):
$$
y_i = \beta_1 (x_{i})_1+ \dots + \beta_{p} (x_i)_p +  \epsilon_i,
$$
where $\beta_1, \dots, \beta_p$ are the $p$ unknown parameters. We will assume that the $\epsilon_i$ are independent and identically-distributed (i.i.d.) error terms with zero mean. Note that, if $(x_i)_1 = 1$ for all $i=1, \dots, n$, this is equivalent to having an intercept term given by $\beta_1$.

We can write this more conveniently by letting $X$ be the *design matrix* of size $n\times p$, whose $i$th row is given by the vectors $x_i$ (where $(x_i)_1=1$ if the model has an intercept), while $y$ is an $n$-vector of observations, whose entries are $y_i$, and similarly for $\epsilon$:
$$
y = X\beta + \epsilon.
$$
The usual approach to finding an estimate for $\beta$, which we call $\hat \beta$, is the Ordinary Least Squares (OLS) estimator given by $\hat{\beta} = (X^TX)^{-1}X^Ty$, which is efficient and has good statistical properties when the error terms are all of roughly the same magnitude (*i.e.*, there are no outliers). On the other hand, the OLS estimator is very sensitive to outliers: even if a single  observation lies far from the regression hyperplane, OLS will often fail to find a good estimate for the parameters, $\beta$.

To solve this problem, a number of methods have been developed in the literature. These methods can be roughly placed in one or more of the five following categories: diagnostics, direct methods, robust methods, and multivariate methods. *Diagnostics* are methods which attempt to find points that significantly affect the fit of a model (often, such points can be labeled as outliers). Diagnostics can then be used to initialize *direct methods*, which fit a (usually non-robust) model to a subset of points suspected to be clear of outliers; remaining points which are not outliers with respect to this fit are continually added to this subset until all points not in the subset are deemed outliers. *Robust methods*, on the other hand, find a best-fit model by approximately minimizing a loss function that is resistant to outliers. Some of the proposed methods are also *multivariate methods*, which can accommodate fitting models that have multiple dependent variables for every data point. *Visual methods* generally work on the principle of visualizing the statistics obtained from these mentioned methods.

# Statement of need 

In practice, many of the proposed methods have reasonable performance and yield similar results for most datasets, but sometimes differ widely in specific circumstances by means of masking and swamping ratios. Additionally, some of the methods are relatively complicated and, if canonical implementations are available, they are often out of date or only found in specific languages of the author's choice, making it difficult for researchers to compare the performance of these algorithms on their datasets.

We have reimplemented many of the algorithms available in the literature in Julia [@julia], an open-source, high performance programming language designed primarily for scientific computing. Our package, `LinRegOutliers`, is a comprehensive and simple-to-use Julia package that includes many of the algorithms in the literature for detecting outliers in linear regression. The implemented `Julia` methods for diagnostics, direct methods, robust methods, multivariate methods, and visual diagnostics are shown in Table 1, Table 2, Table 3, Table 4, and Table 5, respectively. 
 

| Algorithm(s)                    | Reference      | Method                         |
| :------------------------------ | :------------- | :----------------------------- |
| Hadi Measure                    | [@hadimeasure] | `hadimeasure`                  |
| Covariance Ratio, DFBETA, DFFIT | [@diagnostics] | `covration`, `dfbeta`, `dffit` |
| Mahalanobis Distances           | [@mahalanobis] | `mahalanobisSquaredMatrix`     |
| Cook Distances                  | [@cooks]       | `cooks`                        |

Table: Regression Diagnostics

| Algorithm   | Reference     | Method       |
| :---------- | :------------ | :----------- |
| Ransac      | [@ransac]     | `ransac`     |
| KS-89       | [@ks89]       | `ks89`       |
| HS-93       | [@hs93]       | `hs93`       |
| Atkinson-94 | [@atkinson94] | `atkinson94` |
| PY-95       | [@py95]       | `py95`       |
| CM-97       | [@cm97]       | `cm97`       |
| SMR-98      | [@smr98]      | `smr98`      |
| ASM-2000    | [@asm2000]    | `asm2000`    |
| BACON       | [@bacon]      | `bacon`      |
| Imon-2005   | [@imon2005]   | `imon2005`   |
| bch         | [@bch]        | `bch`        |

Table: Direct Methods


| Algorithm                         | Reference     | Method       |
| :-------------------------------- | :------------ | :----------- |
| Least Absolute Deviations         | [@lad]        | `lad`        |
| Least Absolute Trimmed Deviations | [@lta]        | `lta`        |
| Least Median of Squares           | [@lms]        | `lms`        |
| Least Trimmed Squares             | [@lts]        | `lts`        |
| ga-lts                            | [@galts]      | `galts`      |
| Satman-2013                       | [@satman2013] | `satman2013` |
| Satman-2015                       | [@satman2015] | `satman2015` |
| CCF                               | [@ccf]        | `ccf`        |

Table: Robust Methods


| Algorithm                      | Reference   | Method     |
| :----------------------------- | :---------- | :--------- |
| Hadi-1992                      | [@hadi1992] | `hadi1992` |
| Hadi-1994                      | [@hadi1994] | `hadi1994` |
| Minimum Volume Ellipsoid       | [@mve]      | `mve`      |
| Minimum Covariance Determinant | [@mcd]      | `mcd`      |

Table: Multivariate Methods


| Algorithm       | Reference     | Method                   |
| :-------------- | :------------ | :----------------------- |
| BCH Plot        | [@bch]        | `bchplot`                |
| MVE-LTS Plot    | [@mve]        | `mveltsplot`             |
| Data Images     | [@dataimage]  | `dataimage`              |
| Stalactite Plot | [@atkinson94] | `atkinsonstalactiteplot` |

Table: Visual Methods

# Installation and basic usage

`LinRegOutliers` can be downloaded and installed using the Julia package manager by typing

```julia
julia> using Pkg
julia> Pkg.add("LinRegOutliers")
```

in the Julia console. The regression methods follow a uniform call convention. For instance, a user can type

```julia
julia> setting = createRegressionSetting(@formula(calls ~ year), phones);
julia> smr98(setting)
Dict{String,Array{Int64,1}} with 1 entry:
  "outliers" => [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
```

or

```julia
julia> X = hcat(ones(24), phones[:, "year"]);
julia> y = phones[:, "calls"];
julia> smr98(X, y)
Dict{String,Array{Int64,1}} with 1 entry:
  "outliers" => [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
```

to apply *smr98* [@smr98] on the Telephone dataset [@lms], where $X$ is the design matrix with ones in its first column. In this case, observations 15 to 24 are reported as outliers by the method. Some methods may also return additional information specific to the method which is passed back in a ```Dict``` object. For example, the *ccf* function returns a ```Dict``` object containing *betas*, *outliers*, *lambdas*, and *residuals*:

```julia
julia> ccf(X, y)
Dict{Any,Any} with 4 entries:
  "betas"     => [-63.4816, 1.30406]
  "outliers"  => [15, 16, 17, 18, 19, 20]
  "lambdas"   => [1.0, 1.0, 1.0, 1.0, 1.0, ...
  "residuals" => [-2.67878, -1.67473, -0.37067, -0.266613, …
```

Indices of outliers can be accessed using standard ```Dict``` operations like

```julia
julia> result = ccf(X, y)
julia> result["outliers"]
6-element Array{Int64,1}:
 15
 16
 17
 18
 19
 20
```
 
# Acknowledgements

Guillermo Angeris is supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-1656518. 




# References
