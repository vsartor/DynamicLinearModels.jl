module DynamicLinearModels

export dlm_dimension, simulate

using Distributions
using LinearAlgebra


"""
    dlm_dimension(F, G[, V, W, Y])

Computes the dimension of a Dynamic Linear Model based on the observational
and evolutional matrices. The error covariance matrices and observations may
be passed as well so that dimensions can be checked.
"""
function dlm_dimension(F::Matrix{Float64},
                       G::Matrix{Float64},
                       V::Union{Symmetric{Float64}, Nothing} = nothing,
                       W::Union{Symmetric{Float64}, Nothing} = nothing,
                       Y::Union{Vector{Vector{Float64}}, Nothing} = nothing)

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

    if !isnothing(Y)
        for t = 1:size(Y,1)
            if size(Y[t],1) != n
                throw(DimensionMismatch("Observations with wrong dimension."))
            end
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

    θ = Vector{Vector{Float64}}(undef, T)
    y = Vector{Vector{Float64}}(undef, T)

    ω = MultivariateNormal(zeros(p), W)
    ϵ = MultivariateNormal(zeros(n), V)

    θ[1] = G * θ₀ + rand(ω)
    y[1] = F * θ[1] + rand(ϵ)
    for t = 2:T
        θ[t] = G * θ[t-1] + rand(ω)
        y[t] = F * θ[t] + rand(ϵ)
    end

    return θ, y
end


"""
    filter(Y, F, G, V, W[, m₀, C₀])

Dynamic Linear Model simple filtering routine for the case where the covariance
matrices are known and constants. If the parameters for the prior multivariate
normal distribution is not given, smart values that have little effect on the
result are chosen.
"""
function filter(Y::Vector{Vector{Float64}},
                F::Matrix{Float64},
                G::Matrix{Float64},
                V::Symmetric{Float64},
                W::Symmetric{Float64},
                m₀::Union{Vector{Float64}, Nothing} = nothing,
                C₀::Union{Symmetric{Float64}, Nothing} = nothing)
    #TODO: Create new methods:
    #        - One that uses discount factors.
    #        - One that uses discount factors and uses stochastic variance.

    n, p = dlm_dimension(F, G, V, W, Y)
    T = size(Y, 1)

    if isnothing(m₀)
        m₀ = zeros(p)
    end

    if isnothing(C₀)
        magic_sdev = maximum(abs.(Y[1] - F * m₀))
        C₀ = Symmetric(Diagonal(repeat([magic_sdev^2], p)))
    end

    a = Vector{Vector{Float64}}(undef, T)
    m = Vector{Vector{Float64}}(undef, T)
    R = Vector{Symmetric{Float64}}(undef, T)
    C = Vector{Symmetric{Float64}}(undef, T)

    a[1] = G * m₀
    R[1] = Symmetric(G * C₀ * G') + W
    f = F * a[1]
    Q = Symmetric(F * R[1] * F') + V
    A = R[1] * F' * inv(Q)
    m[1] = a[1] + A * (Y[1] - f)
    C[1] = R[1] - Symmetric(A * Q * A')

    for t = 2:T
        a[t] = G * m[t-1]
        R[t] = Symmetric(G * C[t-1] * G') + W
        f = F * a[t]
        Q = Symmetric(F * R[t] * F') + V
        A = R[t] * F' * inv(Q)
        m[t] = a[t] + A * (Y[t] - f)
        C[t] = R[t] - Symmetric(A * Q * A')
    end

    return a, R, m, C
end


function smoother(F::Matrix{Float64},
                  G::Matrix{Float64},
                  a::Vector{Vector{Float64}},
                  R::Vector{Symmetric{Float64}},
                  m::Vector{Vector{Float64}},
                  C::Vector{Symmetric{Float64}})
    n, p = dlm_dimension(F, G)
    T = size(R, 1)

    s = Vector{Vector{Float64}}(undef, T)
    S = Vector{Symmetric{Float64}}(undef, T)

    s[T] = m[T]
    S[T] = C[T]

    for t = T-1:-1:1
        B = C[t] * G' * inv(R[t+1])
        s[t] = m[t] + B * (s[t+1] - a[t+1])
        S[t] = C[t] - Symmetric(B * (R[t+1] - S[t+1]) * B')
    end

    return s, S
end

end # module
