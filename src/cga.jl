module CGA 

export cga 


"""
Generates a binary array of values using a probability vector.
Each single element of the probability vector is the probability of bit having 
the value of 1. When the probability vector is [1, 1, 1, ..., 1] then the sampled
vector is [1.0, 1.0, 1.0, ..., 1.0] whereas it is [0.0, 0.0, 0.0, ..., 0.0] when the probability vector
is a vector of zeros. The CGA (compact genetic algorithms) search is started using the 
probability vector of [0.5, 0.5, 0.5, ..., 0.5] which produces random vectors of either
zeros or ones.
 
# Examples
```jldoctest
julia> sample([1, 1, 1, 1, 1])
5-element Array{Bool,1}:
 1
 1
 1
 1
 1
julia> cgasample(ones(10) * 0.5)
10-element Array{Bool,1}:
 1
 1
 1
 1
 0
 0
 0
 1
 1
 0
```
"""
function cgasample(probvector::Array{Float64,1})::Array{Bool}
    len = length(probvector)
    ch = Array{Bool}(undef, len)
    rands = rand(Float64, len)
    @inbounds for i in eachindex(ch)
        ch[i] = rands[i] < probvector[i]
    end
    return ch
end


"""
Performs a CGA (Compact Genetic Algorithm) search for minimization of an objective function.
In the example below, the objective function is to minimize sum of bits of a binary vector.
The search method results the optimum vector of [0, 0, ..., 0] where the objective function is zero.
# Examples
```jldoctest
julia> function f(x)
           return sum(x)
       end
f (generic function with 1 method)
julia> cga(chsize = 10, costfunction = f, popsize = 100)
10-element Array{Bool,1}:
 0
 0
 0
 0
 0
 0
 0
 0
 0
 0
```
"""
function cga(;chsize::Int, costfunction::Function, popsize::Int)::Array{Bool,1}
    probvector = ones(Float64, chsize) * 0.5
    mutation = 1.0 / convert(Float64, popsize)
    while !(all(x -> (x <= mutation) || (x >= 1.0 - mutation), probvector))
        ch1 = cgasample(probvector)
        ch2 = cgasample(probvector)
        cost1 = costfunction(ch1)
        cost2 = costfunction(ch2)
        winner = ch1
        loser = ch2
        if (cost2 < cost1) 
            winner = ch2
            loser = ch1
        end
        for i in 1:chsize	
		if winner[i] != loser[i]
			if winner[i]
				probvector[i] += mutation
        			else
				probvector[i] -= mutation
                	end
		end
        end
    end
    return cgasample(probvector)
end


end # end of module CGA 