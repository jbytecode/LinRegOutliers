using LinRegOutliers

import LinRegOutliers.GA: ga


const e = exp(1)

function f(x)
    return (x[1] - pi)^2 + (x[2] - e)^2
end 


@time result = ga(100, 
2, f, 
[-500.0, -500.0], 
[500.0, 500.0], 
1.0, 
0.1, 
2, 
10000)

println(result[1])