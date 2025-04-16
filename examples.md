# Some examples


## A brief introduction

Suppose the linear regression model is 

*y = Xβ + ε*

where *y* is the vector of dependent variable, *X* is the *n ˟ p* matrix of design, *ε* is i.i.d error term with zero mean, *n* is the number of observations, and *p* is the number of regression parameters. 

When a single regressor exists in the model, it can be basically written as 

*y = β₀ + β₁ x +  ε*

where *β₀* and *β₁* are unknown intercept and slope parameters. In `R` and `Julia` we can represent this model in a similar form. Specifically, in Julia, the simple model can be expressed using the `@formula` macro as

```julia
@formula(y ~ x)
```

where ```~``` operator separates the dependent and independent variables. When the model includes more than one regressors, the model can similarly be expressed as

```julia
@formula(y ~ x1 + x2 + x3)
```

```LinRegOutliers``` follows this convention for expressing linear models. 

_________________

## Sebert & Montgomery & Rollier (1998) Algorithm
Sebert & Montgometry & Rollier (smr98) algorithm starts with an ordinary least squares estimation for a given model and data. Residuals and fitted responses are calculated using the estimated model. A hierarchical clustering analysis is applied using standardized residuals and standardized fitted responses. The tree structure of clusters are cut using a threshold, e.g Majona criterion, as suggested by the authors. It is expected that the subtrees with relatively small number of observations are declared to be clusters of outliers.

Hawkings & Bradu & Kass dataset has 4 variables and 75 observations. The observations 1-14 are known to be outliers. In the example below, we create an regression setting using the formula ```y ~ x1 + x2 + x3``` and ```hbk``` dataset. ```smr98``` is directly applied on this setting.  

```julia
julia> using LinRegOutliers
julia> # Regression setting for Hawkins & Bradu & Kass data
julia> reg = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)
julia> smr98(reg)
Dict{String,Array{Int64,1}} with 1 entry:
  "outliers" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
```

The Julia method ```smr98()``` returns a ```Dict``` object with produced output. In this case, the single output is indices of detected outlying observations. 
The algorithm successfully detects the outliers.

________________________
## Peña and Yohai (1995) 

Peña and Yohai (```py1995```) algorithm starts by constructing an influence matrix using results of an ordinary least squares estimate for a given model and data. In the second stage, the eigen structure of the influence matrix is examined for detecting subset of potential outliers of data. 

Here is an example of ```py95``` method applied on the ```hbk``` data. The method returns a ```Dict```  object with keys ```outliers``` and ```suspected.sets```. An usual researcher may directly focus on the ```outliers``` indices. The method reports the observations 1-14 are outliers.

```julia
julia> py95(reg)
Dict{Any,Any} with 2 entries:
  "outliers"       => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  "suspected.sets" => Set([[14, 13], [43, 54, 24, 38, 22], Int64[], [58, 66, 32, 28, 65, 36], [62], [73, 2…
```


________________________________

## Least Trimmed Squares Regression
Least Trimmed Squares (LTS) is a robust regression estimator with high break-down point. LTS searches for the parameter estimates that minimize sum of the *h* smallest squared residuals where *h* is a constant larger than *n/2*.    

Phone data is a regression data with a single regressor variable. The independent and dependent variables are ```year``` and ```calls``` and have 24 observations. Observations 14-21 are said to be outliers. 

Since LTS is a robust method, the parameter estimates are of interest. However, we provide indices of outliers in results for diagnostic purposes only. 

The method ```lts``` also reports standard deviation of estimate, scaled residuals and LTS objective function as well. 

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



```julia
using Plots
x = phones[:,"year"]
y = phones[:,"calls"]
f(x) = -56.5219 +  1.16488x 
scatter(x, y, label=false, title="Phone Data")
px = [x[1], x[end]]
py = map(f, px)
plot!(px, py, label=false, color=:red, width=2) 
```


<img src="https://github.com/jbytecode/jbytecode/blob/master/images/ltsandphonedata.png" alt="dataimages" width="500"/>

Figure 1 - Phone Data and estimated LTS line



_________________

## Data Images

The method ```dataimage``` implements the Data Image algorithm and serves a visual tool as an outlier detection algorithm for multivariate data only. The algorithm generates a color matrix with each single cell represents a proper distance between observations. Since 

```dataimage(data, distance = :euclidean)```

defines color using the Euclidean distance, whereas

```dataimage(data, distance = :mahalabobis)```

uses Mahalanobis distances for determining color values. The default distance metric is Euclidean distance. 

In the example below, the distances between observations are calculated and drawn using corresponding colors. Since the method is for multivariate data, only the design matrix is used. In other terms, the response vector is omitted. 

```julia
julia> # Matrix of independent variables of Hawkins & Bradu & Kass data
julia> data = hcat(hbk.x1, hbk.x2, hbk.x3);
julia> dataimage(data)
``` 


<img src="https://github.com/jbytecode/jbytecode/blob/master/images/dataimages.png" alt="dataimages" width="500"/>

Figure 2 - Data Image of Design Matrix of ```hbk``` Data


_________________________
## Atkinson's Stalactite Plot

Atkinson's Stalactite Plot serves a visual method for detecting outliers in linear regression. Despite it shares the same calling convention with the other methods, the method ```atkinsonstalactiteplot``` generates a text based plot. The method performs a robust regression estimator many times and residuals higher than some threshold are labelled using ```+``` and ```*```. After many iterations, the observations with many labels are considered as suspected or outlying observations.  

```julia
julia> using LinRegOutliers
julia> reg = createRegressionSetting(@formula(calls ~ year), phones);
julia> atkinsonstalactiteplot(reg)
m           1         2
   123456789012345678901234
 2              ********   
 3              ********   
 4 +            ********   
 5 +            ********   
 6 +            ********   
 7 +            ********   
 8 +            ********   
 9 +            ********+  
10 +            ********+  
11 +            ********   
12              ********   
13 +            ********   
14              ********   
15              ********   
16              ********   
17              ********   
18               *******+  
19               ****** +++
20               ****** +++
21               ****** +++
22               ****** +++
23               +++*** +++
24                  ++* +++
   123456789012345678901234
            1         2

```

The output above can be considered as an evidence that the observations 14-21 are suspected. Observations 1, 22, 23, 24 are also labelled as ```+``` in some iterations. However, the frequency of labels of these observations are relatively small. 


____________________
## Other algorithms
```LinRegOutliers``` implements more than 20 outlier detection methods in linear regression and covers a big proportion of the classical literature in this subject. The documentation of the package includes the referenced citations. Any researcher can follow the details of algorithms using these information. 

_____________________
## Other calling conventions
The calling convention 

```julia
julia> setting = createRegressionSetting(@formula(...), data)
julia> method(setting) 
```

is the preferred way of calling implemented methods in ```LinRegOutliers```, we multiple dispatch the methods using the syntax

```julia
julia> method(X, y) 
```

where *X* is the design matrix and *y* is the response vector. This calling convention may be more suitable for those who iteratively calls the methods possibly in a simulation study or other kinds of researching stuff. 

For example, we can perform ```hs93``` on the Phones data using 

```julia
julia> hs93(reg)
Dict{Any,Any} with 3 entries:
  "outliers" => [14, 15, 16, 17, 18, 19, 20, 21]
  "t"        => -3.59263
  "d"        => [2.04474, 1.14495, -0.0633255, 0.0632934, -0.354349, -0.766818, -1.06862, -1.47638, -0.710…
```

as well as 

```julia
julia> X = hcat(ones(24), phones[:, "year"]);

julia> y = phones[:, "calls"];

julia> hs93(X, y)
Dict{Any,Any} with 3 entries:
  "outliers" => [14, 15, 16, 17, 18, 19, 20, 21]
  "t"        => -3.59263
  "d"        => [2.04474, 1.14495, -0.0633255, 0.0632934, -0.354349, -0.766818, -1.06862, -1.47638, -0.710…
```

__________________________
## Multiple Methods in a single shot!

We also provide ```detectOutliers``` method for data scientist for performing many methods and presenting the summarized results. 

The method can be called using default arguments only by feeding a regression setting object:

```julia
julia> detectOutliers(aSettingObject)
```

The method generates a console output:

<img src="https://github.com/jbytecode/jbytecode/blob/master/images/detectoutliers.png" alt="dataimages" width="500"/>
