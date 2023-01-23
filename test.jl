using LinRegOutliers



#sett = createRegressionSetting(@formula(calls ~ year), phones)
#sett = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)

n = 1000
p = 25
betas = [5.0 for i in 1:p]
x = rand(n, p)
e = randn(n)
y = x * betas + e


result = lts(x, y)