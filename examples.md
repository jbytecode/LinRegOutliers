# Some examples

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

julia> lts(reg)
Dict{Any,Any} with 6 entries:
  "betas"            => [-56.5219, 1.16488]
  "S"                => 1.10918
  "hsubset"          => [11, 10, 5, 6, 23, 12, 13, 9, 24, 7, 3, 4, 8]
  "outliers"         => [14, 15, 16, 17, 18, 19, 20, 21]
  "scaled.residuals" => [2.41447, 1.63472, 0.584504, 0.61617, 0.197052, -0.222066, -0.551027, -0.970146, -0.397538, -0.185558  …  91.0312, 94.4889, 109.667, 123.943, 143.629, …
  "objective"        => 3.43133
```  



<img src="https://github.com/jbytecode/jbytecode/blob/master/images/detectoutliers.png" alt="alt text" width="500"/>




```julia
julia> # Matrix of independent variables of Hawkins & Bradu & Kass data
julia> data = hcat(hbk.x1, hbk.x2, hbk.x3);
julia> dataimage(data)
``` 


<img src="https://github.com/jbytecode/jbytecode/blob/master/images/dataimages.png" alt="alt text" width="500"/>

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
