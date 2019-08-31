module DynamicLinearModels

export dlm_dimension, simulate

using Distributions
using LinearAlgebra


"""
    CovMat

Simple alias for Symmetric Dense matrices.
"""
const CovMat{RT <: Real} = Symmetric{RT, Matrix{RT}}


"""
    collect(x, index)

Utility (possibly temporary) function that facilitates fetching the time_series
for one of the DLM objects.
"""
function collect(x::Vector{<: Union{Vector{RT}, CovMat{RT}}},
                 index::Integer)::Vector{RT} where RT <: Real
    if typeof(x) == Vector{RT}
        return [x[t][index] for t = 1:size(x,1)]
    else
        return [x[t][index,index] for t = 1:size(x,1)]
    end
end


"""
    dlm_dimension(F, G[, V, W, Y])

Internal utility function that computes the dimension of a Dynamic Linear Model
based on the observational and evolutional matrices. The error covariance
matrices and observations may be passed as well so that dimensions can be
checked.
"""
function dlm_dimension(F::Matrix{RT},
                       G::Matrix{RT};
                       V::Union{CovMat{RT}, Nothing} = nothing,
                       W::Union{CovMat{RT}, Nothing} = nothing,
                       Y::Union{Vector{Vector{RT}}, Nothing} = nothing) where RT <: Real

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
function simulate(F::Matrix{RT},
                  G::Matrix{RT},
                  V::CovMat{RT},
                  W::CovMat{RT},
                  θ₀::Vector{RT},
                  T::Integer,
                  nreps::Integer = 1) where RT <: Real

    # TODO: Allow V and W to be lists

    # TODO: Create a new method:
    #         Allow for a discount factor to be passed (it is computed online
    #         with an ongoing filtering routine, starting with a smart prior on
    #         θ₀).

    n, p = dlm_dimension(F, G, V=V, W=W)

    θ = Vector{Vector{RT}}(undef, T)
    y = Vector{Vector{RT}}(undef, T)

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
    dlm_set_prior(Y, F, m₀, C₀)

Internal utility function for computing a smart prior that's not informative
but at the same time doesn't lead to computational or visualization problems.
"""
function dlm_set_prior(Y::Vector{Vector{RT}},
                       F::Matrix{RT},
                       m₀::Union{Vector{RT}, Nothing},
                       C₀::Union{CovMat{RT}, Nothing}) where RT <: Real

     p = size(F, 2)

     if isnothing(m₀)
         m₀ = zeros(p)
     end

     if isnothing(C₀)
         magic_sdev = maximum(abs.(Y[1] - F * m₀))
         C₀ = Symmetric(Diagonal(repeat([magic_sdev^2], p)))
     end

     return m₀, C₀
end


"""
    filter(Y, F, G, V, W[, m₀, C₀])

Filtering routine for the simplest Dynamic Linear Model case where covariance
matrices are known and constants. If the parameters for the prior multivariate
normal distribution is not given, smart values that have little effect on the
result are chosen.
"""
function filter(Y::Vector{Vector{RT}},
                F::Matrix{RT},
                G::Matrix{RT},
                V::CovMat{RT},
                W::CovMat{RT},
                m₀::Union{Vector{RT}, Nothing} = nothing,
                C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    #TODO: Create new methods:
    #        - One that uses discount factors and uses stochastic variance.

    n, p = dlm_dimension(F, G, V=V, W=W, Y=Y)
    T = size(Y, 1)

    m₀, C₀ = dlm_set_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

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


"""
    filter(Y, F, G, V, δ[, m₀, C₀])

Filtering routine for a discount factor Dynamic Linear Model where observational
covariance is known and constants. If the parameters for the prior multivariate
normal distribution is not given, smart values that have little effect on the
result are chosen.
"""
function filter(Y::Vector{Vector{RT}},
                F::Matrix{RT},
                G::Matrix{RT},
                V::CovMat{RT},
                δ::RT,
                m₀::Union{Vector{RT}, Nothing} = nothing,
                C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    n, p = dlm_dimension(F, G, V=V, Y=Y)
    T = size(Y, 1)

    m₀, C₀ = dlm_set_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

    a[1] = G * m₀
    R[1] = Symmetric(G * C₀ * G') / δ
    f = F * a[1]
    Q = Symmetric(F * R[1] * F') + V
    A = R[1] * F' * inv(Q)
    m[1] = a[1] + A * (Y[1] - f)
    C[1] = R[1] - Symmetric(A * Q * A')

    for t = 2:T
        a[t] = G * m[t-1]
        R[t] = Symmetric(G * C[t-1] * G') / δ
        f = F * a[t]
        Q = Symmetric(F * R[t] * F') + V
        A = R[t] * F' * inv(Q)
        m[t] = a[t] + A * (Y[t] - f)
        C[t] = R[t] - Symmetric(A * Q * A')
    end

    return a, R, m, C
end


"""
    smoother(F, G, a, R, m, C)

Smoothing routine for the simplest Dynamic Linear Model case.
"""
function smoother(F::Matrix{RT},
                  G::Matrix{RT},
                  a::Vector{Vector{RT}},
                  R::Vector{CovMat{RT}},
                  m::Vector{Vector{RT}},
                  C::Vector{CovMat{RT}}) where RT <: Real

    n, p = dlm_dimension(F, G)
    T = size(R, 1)

    s = similar(m, T)
    S = similar(C, T)

    s[T] = m[T]
    S[T] = C[T]

    for t = T-1:-1:1
        B::Matrix{RT} = C[t] * G' * inv(R[t+1])
        s[t] = m[t] + B * (s[t+1] - a[t+1])
        S[t] = C[t] - Symmetric(B * (R[t+1] - S[t+1]) * B')
    end

    return s, S
end


"""
    forecast(F, G, V, W, s, S, h)

Forecast routine for the simplest Dynamic Linear Model with a horizon `h`.
"""
function forecast(F::Matrix{RT},
                  G::Matrix{RT},
                  V::CovMat{RT},
                  W::CovMat{RT},
                  m::Vector{Vector{RT}},
                  C::Vector{CovMat{RT}},
                  h::Integer) where RT <: Real

    f = similar(m, h)
    Q = similar(C, h)

    a = G * m[end]
    R = Symmetric(G * C[end] * G') + W
    f[1] = G * a
    Q[1] = Symmetric(F * R * F') + V

    for t in 2:h
        a = G * a
        R = Symmetric(G * R * G') + W
        f[t] = G * a
        Q[t] = Symmetric(F * R * F') + V
    end

    return f, Q
end


"""
    forecast(F, G, V, δ, s, S, h)

Forecast routine for a discount factor Dynamic Linear Model with a horizon `h`.
"""
function forecast(F::Matrix{RT},
                  G::Matrix{RT},
                  V::CovMat{RT},
                  δ::RT,
                  m::Vector{Vector{RT}},
                  C::Vector{CovMat{RT}},
                  h::Integer) where RT <: Real
    W = C[end] / δ

    return forecast(F, G, V, W, m, C, h)
end

end # module
