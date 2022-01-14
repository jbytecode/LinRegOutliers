using LinRegOutliers 

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 20.0]

one = ones(Float64, length(x))

X = hcat(one, x)


result = lad2(X, y)