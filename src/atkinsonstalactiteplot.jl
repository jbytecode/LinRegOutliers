module AtkinsonPlot 


export atkinsonstalactiteplot
export generate_stalactite_plot

import ..Basis: RegressionSetting, @extractRegressionSetting, designMatrix, responseVector, applyColumns
import ..Atkinson94: atkinson94

"""
        atkinsonstalactiteplot(setting, iters, crit)

Runs the atkinson94 algorithm and additionally plots the Stalactite text plot as described in the paper. Works for data with less than 
100 points at the moment (since screen width of 80-120 is the standard).
See `atkinson94` for details.

# Arguments
- `setting::RegressionSetting`: A regression setting object.
- `iters::Int`: Number of random samples.
- `crit::Float64`: Critical value for residuals

# References
Atkinson, Anthony C. "Fast very robust methods for the detection of multiple outliers."
Journal of the American Statistical Association 89.428 (1994): 1329-1339.
"""
function atkinsonstalactiteplot(setting::RegressionSetting; iters=nothing, crit=3.0)::Nothing
    X, y = @extractRegressionSetting setting
    return atkinsonstalactiteplot(X, y, iters=iters, crit=crit)
end

function atkinsonstalactiteplot(X::Array{Float64,2}, y::Array{Float64,1}; iters=nothing, crit=3.0)::Nothing
    n, p = size(X)
    output = atkinson94(X, y, iters=iters, crit=crit)
    residuals_matrix = output["residuals_matrix"]
    generate_stalactite_plot(residuals_matrix, n, p, crit)
end

function generate_stalactite_plot(residuals_matrix::Array{Float64,2}, n::Int64, p::Int64, crit)::Nothing
    print("m  ")
    print_tens_row(n)
    println()
    print("   ")
    print_ones_row(n)
    println()

    for m = p:n
        if m <= 9
            print(" $m ")
        else
            print("$m ")
        end
        for i = 1:n
            print(stalactite_char_value(residuals_matrix[m - p + 1, i], crit))
        end
        println()
    end
    print("   ")
    print_ones_row(n)
    println()
    print("   ")
    print_tens_row(n)
    println()
end

function print_tens_row(n::Int64)::Nothing
    for i = 1:Int(floor(n / 10))
        for i = 1:9
            print(" ")
        end
        print(i)
    end
end


function print_ones_row(n::Int64)::Nothing
    for i = 1:n
        print(i % 10)
    end
end

function stalactite_char_value(x, crit)::Char
    if x >= crit
        return '*'
    elseif x >= crit - 1
        return '+'
    else
        return ' '
    end
end


end # end of module AtkinsonPlot 