module HookeJeeves

export hj

function mutate(par, p, d)
    newpar = copy(par)
    newpar[p] += d
    return newpar
end

function hj(
    f::FType,
    par::AbstractVector{Float64};
    maxiter = 1000,
    startstep = 5.0,
    endstep = 0.0001,
) where {FType<:Function}
    p = length(par)
    currentstep = startstep
    iter::Int64 = 0
    while iter < maxiter
        fold = f(par)
        fnow = fold
        for currentp = 1:p
            mutateleft = mutate(par, currentp, -currentstep)
            fleft = f(mutateleft)
            mutateright = mutate(par, currentp, currentstep)
            fright = f(mutateright)
            if fleft < fold
                par = mutateleft
                fnow = fleft
            elseif fright < fold
                par = mutateright
                fnow = fright
            end
        end
        if fold <= fnow
            currentstep /= 2
        end
        if currentstep < endstep
            break
        end
        iter += 1
    end

    return Dict("par" => par, "iter" => iter, "step" => currentstep)
end

end # end of module
