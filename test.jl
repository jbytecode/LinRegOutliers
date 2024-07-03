using LinRegOutliers
using BenchmarkTools




sett = createRegressionSetting(@formula(calls ~ year), phones)

@btime lta($sett)