using LinRegOutliers



sett = createRegressionSetting(@formula(calls ~ year), phones)

result = dfbetas(sett)