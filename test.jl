using LinRegOutliers



#sett = createRegressionSetting(@formula(calls ~ year), phones)
#sett = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)

n = 1000
betas = [5.0 for i in 1:10]
x = rand(n, 10)
e = randn(n)
y = x * betas + e


result = lts(x, y)