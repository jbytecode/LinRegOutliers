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
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Guillermo Angeris
    orcid: 0000-0000-0000-0000 
    affiliation: 3
  - name: Emre Akadal
    orcid: 0000-0000-0000-0000 
    affiliation: 4
affiliations:
 - name: Department of Econometrics, Istanbul University, Istanbul, Turkey
   index: 1
 - name: AFFILIATION FOR SHREESH ADIGA
   index: 2
 - name: AFFILIATION FOR GUILLERMO ANGERIS
   index: 3
 - name: AFFILIATION FOR EMRE AKADAL
   index: 4

date: 26 November 2020
bibliography: paper.bib
---

# Summary

*LinRegOutliers* is a *Julia* package that implements many outlier detection algorithms in linear regression. The implemented algorithms can be categorized as regression diagnostics, direct methods, robust methods, multivariate methods, and graphical methods. 
Diagnostics are generally used in initial stages of direct methods. Since the design matrix of a regression model is multivariate data, the package also contains some robust covariance matrix estimators and outlier detection methods developed for multivariate data. Graphical methods are implemented for visualizing the regression residuals and the distances between observations using Euclidean or Mahalanobis distances generally calculated using a robust covariance matrix. The functions in the package have the same calling conventions. The package covers most of the literature on this topic and provides a good basis for those who want to implement novel methods.


# State of the field
Suppose the linear regression model is

$$
y = X \beta + \epsilon
$$

where $y$ is the vector of dependent variable, $X$ is the $n \times p$ design matrix, $\beta$ is the vector
of unknown parameters, $n$ is the number of observations, $p$ is the number of parameters, and $\epsilon$ is the vector of error-term with zero mean and constant 
variance. The Ordinary Least Squares (OLS) estimator $\hat{\beta} = (X'X)^{-1}X'y$ is not resistant to outliers even if a single 
observation lies far from the regression hyper-plane. There are many direct and robust methods for detecting outliers in linear regression in the literature. Robust methods are based on estimating robust regression parameters and observations with higher absolute residuals are labelled as outliers whereas direct methods are generally based on enlarging a clean subset of observations by iterations since a termination criterion is met. 
Although outlier detection in multivariate data is another topic, robust covariance estimators and some multivarite outlier detection methods are used for detecting influential observations in design matrix of regression models.
    
# Statement of need 

Julia is a high performance programming language designed primarily for scientific computing. *LinRegOutliers* is a Julia package that covers the literature on this topic well. The package implements *ks89* [@ks89], *hadi1992* [@hadi1992], *hs93* [@hs93], *atkinson94* [@atkinson94], *cm97* [@cm97], *asm2000* [@asm2000], *bacon* [@bacon], *dataimage* [@dataimage], *bch* [@bch], *ccf* [@ccf]






# References
