using JuMP, Cbc
m = Model(solver=CbcSolver())

@variable(m, x[1:2])
@objective(m, Min, (x[1]-3)^2 + (x[2]-4)^2)
@constraint(m, (x[1]-1)^2 + (x[2]+1)^2 <= 1)

solve(m)