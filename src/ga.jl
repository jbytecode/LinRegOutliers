module GA

import Distributions: sample

abstract type Chromosome end

mutable struct RealChromosome <: Chromosome
    genes::Array{Float64,1}
    cost::Float64
end

RealChromosome() = RealChromosome(Float64[], Inf64)

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

function Base.isless(c1::RealChromosome, c2::RealChromosome)::Bool
    return c1.cost < c2.cost
end

function LinearCrossover(
    c1::RealChromosome,
    c2::RealChromosome,
)::Tuple{RealChromosome,RealChromosome,RealChromosome}
    offspring1 = 0.5 * c1 + 0.5 * c2
    offspring2 = 1.5 * c1 - 0.5 * c2
    offspring3 = 1.5 * c2 - 0.5 * c1
    return (offspring1, offspring2, offspring3)
end

function ArithmeticCrossOver(
    c1::RealChromosome,
    c2::RealChromosome,
)::Tuple{RealChromosome,RealChromosome}
    alpha = rand()
    return alpha * c1 + (1.0 - alpha) * c2
end

function Mutate(c::RealChromosome, prob::Float64)::RealChromosome
    genes = copy(c.genes)
    @inbounds for i in eachindex(genes)
        if rand() < prob
            genes[i] += randn()
        end
    end
    return RealChromosome(genes, Inf64)
end

function TournamentSelection(
    pop::Array{RealChromosome,1},
)::Tuple{RealChromosome,RealChromosome}
    n = length(pop)
    indices = sample(1:n, 4, replace = false)
    lucky1 :: Chromosome = RealChromosome()
    lucky2 :: Chromosome = RealChromosome()
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


function createPopulation(
    popsize::Int,
    chsize::Int,
    mins::Array{Float64,1},
    maxs::Array{Float64,1},
)::Array{RealChromosome,1}
    pop = Array{RealChromosome,1}(undef, popsize)
    @inbounds for i = 1:popsize
        c = RealChromosome(mins .+ rand(chsize) .* (maxs - mins), Inf64)
        pop[i] = c
    end
    return pop
end

function Evaluate(pop::Array{RealChromosome,1}, fcost::Function)::Array{RealChromosome,1}
    @inbounds for i in eachindex(pop)
        pop[i].cost = fcost(pop[i].genes)
    end
    return pop
end

function Generation(
    pop::Array{RealChromosome,1},
    fcost::Function,
    elitism::Int,
    pcross::Float64,
    pmutate::Float64,
)::Array{RealChromosome,1}
    popsize = length(pop)
    newpop = Array{Chromosome, 1}(undef, popsize)
    pop = sort(Evaluate(pop, fcost))
    csize :: Int = 0
    @inbounds for i = 1:elitism
        csize += 1
        newpop[csize] = pop[i]
    end

    winner1 = RealChromosome()
    winner2 = RealChromosome()

    @inbounds while csize < popsize
        parent1, parent2 = TournamentSelection(pop)
        if rand() < pcross
            offspring1, offspring2, offspring3 = LinearCrossover(parent1, parent2)
            offpop = sort(Evaluate([offspring1, offspring2, offspring3], fcost))
            winner1 = Mutate(offpop[1], pmutate)
            winner2 = Mutate(offpop[2], pmutate)
            Evaluate(RealChromosome[winner1, winner2], fcost)
        else
            winner1, winner2 = parent1, parent2
        end
        if (csize + 1 <= popsize)
            csize += 1
            newpop[csize] =  winner1
        end
        if (csize + 1 <= popsize)
            csize += 1
            newpop[csize] =  winner2
        end
    end
    return sort(newpop)
end


function ga(
    popsize::Int,
    chsize::Int,
    fcost::Function,
    mins::Array{Float64,1},
    maxs::Array{Float64,1},
    pcross::Float64,
    pmutate::Float64,
    elitisim::Int,
    iterations::Int,
)::Array{RealChromosome,1}
    pop = createPopulation(popsize, chsize, mins, maxs)
    for _ = 1:iterations
        pop = Generation(pop, fcost, elitisim, pcross, pmutate)
    end
    return pop
end

end # end of module GA 
