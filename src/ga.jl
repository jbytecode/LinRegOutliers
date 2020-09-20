mutable struct RealChromosome
    genes::Array{Float64,1}
    cost::Float64
end

function Base.:*(c1::RealChromosome, num::Float64)::RealChromosome
    RealChromosome(c1.genes * num, Inf64)
end

function Base.:*(num::Float64, c1::RealChromosome)::RealChromosome
    RealChromosome(c1.genes * num, Inf64)
end

function Base.:+(c1::RealChromosome, c2::RealChromosome)::RealChromosome
    RealChromosome(c1.genes .+ c2.genes, Inf64)
end

function Base.:-(c1::RealChromosome, c2::RealChromosome)::RealChromosome
    RealChromosome(c1.genes .- c2.genes, Inf64)
end

function LinearCrossover(c1::RealChromosome, c2::RealChromosome)::Tuple{RealChromosome,RealChromosome,RealChromosome}
    offspring1 = 0.5 * c1 + 0.5 * c2
    offspring2 = 1.5 * c1 - 0.5 * c2
    offspring3 = 1.5 * c2 - 0.5 * c1
    return (offspring1, offspring2, offspring3)
end

function Mutate(c::RealChromosome, prob::Float64)::RealChromosome
    genes = copy(c.genes)
    for i in 1:length(genes)
        if rand(1)[1] < prob
            genes[i] += randn(1)[1] 
        end
    end
    return RealChromosome(genes, Inf64)
end

function TournamentSelection(pop::Array{RealChromosome,1})::Tuple{RealChromosome,RealChromosome}
    n = length(pop)
    indices = sample(collect(1:n), 4, replace=false)
    lucky1 = nothing
    lucky2 = nothing
    if pop[indices[1]].cost < pop[indices[2]].cost 
        lucky1 = pop[indices[1]]
    else
        lucky1 = pop[indices[2]]
    end
    if pop[indices[3]].cost < pop[indices[4]].cost 
        lucky2 = pop[indices[3]]
    else
        lucky2 = pop[indices[4]]
    end
    return (lucky1, lucky2)
end


