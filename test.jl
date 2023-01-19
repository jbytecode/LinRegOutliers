using LinRegOutliers



#sett = createRegressionSetting(@formula(calls ~ year), phones)
sett = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)


#n = 30
#x1 = rand(n)
#x2 = rand(n)
#e = randn(n)
#y = 5 .+ 5 .* x1 + 5 .* x2 + e 
#y[end] = y[end] * 10
#mydata = DataFrame("x1" => x1, "x2" => x2, "y" => y)
#sett = createRegressionSetting(@formula(y ~ x1 + x2), mydata)

result = diagnose(sett)