using LinRegOutliers



#sett = createRegressionSetting(@formula(calls ~ year), phones)
#sett = createRegressionSetting(@formula(y ~ x1 + x2 + x3), hbk)

n = 1000
p = 25
betas = [5.0 for i in 1:p]
x = rand(n, p)
e = randn(n)
y = x * betas + e
range = collect(701:1000)
no = length(range)

y[range] .= rand(no)
for i in 1:25
    x[range, i] .= rand(no)
end
result = lts(x, y)