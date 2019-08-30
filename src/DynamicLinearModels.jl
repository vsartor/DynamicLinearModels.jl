module DynamicLinearModels

export dlm_dimension, simulate

using Distributions
using LinearAlgebra


"""
    dlm_dimension(F, G[, V, W])

Computes the dimension of a Dynamic Linear Model based on the observational
and evolutional matrices. The error covariance matrices and observations may
be passed as well so that dimensions can be checked.
"""
function dlm_dimension(F::Matrix{Float64},
                       G::Matrix{Float64},
                       V::Union{Symmetric{Float64}, Nothing} = nothing,
                       W::Union{Symmetric{Float64}, Nothing} = nothing)

    n = size(F, 1)
    p = size(F, 2)

    if size(G, 1) != size(G, 2)
        throw(DimensionMismatch("G must be a square matrix."))
    end

    if size(G) != (p, p)
        throw(DimensionMismatch("Dimension of G dos not match that of F."))
    end

    if !isnothing(V)
        # Only need to check one dimensions since Symmetric guarantees square
        if size(V, 1) != n
            throw(DimensionMismatch("Dimensions of V are a mismatch."))
        end
    end

    if !isnothing(W)
        # Only need to check one dimensions since Symmetric guarantees square
        if size(W, 1) != p
            throw(DimensionMismatch("Dimensions of W are a mismatch."))
        end
    end

    return n, p
end


"""
    simulate(F, G, V, W, θ₀, T[, nreps])

Simulates a Dynamic Linear Model specified by the quadruple (F, G, V, W)
with a starting state of θ₁ <- N(G θ₀, W), with T observations. A parameters
nreps may be passed indicating the number of replicates to be generated.

Note that the parametrizations being considered in this package is such that
    y[t] = F * y[t-1] + ϵ
and not the notation from West and Harrison (1996) where
    y[t] = F' * y[t-1] + ϵ.
"""
function simulate(F::Matrix{Float64},
                  G::Matrix{Float64},
                  V::Symmetric{Float64},
                  W::Symmetric{Float64},
                  θ₀::Vector{Float64},
                  T::Integer,
                  nreps::Integer = 1)
    # TODO: Allow V and W to be lists

    # TODO: Create a new method:
    #         Allow for a discount factor to be passed (it is computed online
    #         with an ongoing filtering routine, starting with a smart prior on
    #         θ₀).

    n, p = dlm_dimension(F, G, V, W)

    θ = Array{Float64}(undef, T, p)
    y = Array{Float64}(undef, T, n * nreps)

    ω = MultivariateNormal(zeros(p), W)
    ϵ = MultivariateNormal(zeros(n), V)

    θ[1,:] = G * θ₀ + rand(ω)
    y[1,:] = F * θ[1,:] + rand(ϵ)
    for t = 2:T
        θ[t,:] = G * θ[t-1,:] + rand(ω)
        y[t,:] = F * θ[t,:] + rand(ϵ)
    end

    return θ, y
end

end # module
