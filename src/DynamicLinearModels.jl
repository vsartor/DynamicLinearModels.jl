module DynamicLinearModels

export DLMPlot
export extract
export simulate
export compute_prior
export check_dimensions
export kfilter
export ksmoother
export evolutional_covariances
export fitted
export forecast
export estimate


using Distributions
using LinearAlgebra
using RecipesBase


"""
    CovMat

Alias for symmetric, real valued, dense matrices.
"""
const CovMat{RT <: Real} = Symmetric{RT, Matrix{RT}}


"""
    DLMPlot

Indicator type for plotting recipe.
"""
struct DLMPlot end


"""
    plot(::DLMPlot, Y, f, Q[, fh, Qh; factor = 1.64, index = 1])

Recipe for easily plotting the results obtained by the package routines, where
`Y` is the vector of observations, `f` and `Q` are results from `fitted`, and
`fh` and `Qh` are the results from `forecast`. Factor implies the credibility
or the credibility intervals interval, e.g. a factor of 1.64 implies a
credibility of 90%. Index indicates which observational index is to be plotted.
"""
@recipe function plot(::DLMPlot,
                      Y::Vector{Vector{RT}},
                      f::Vector{Vector{RT}},
                      Q::Vector{CovMat{RT}},
                      fh::Union{Vector{Vector{RT}}, Nothing} = nothing,
                      Qh::Union{Vector{CovMat{RT}}, Nothing} = nothing;
                      factor = 1.64,
                      index = 1,
                      nreps = 1) where RT <: Real

    n = size(f[1], 1)
    T = size(Y, 1)

    if isnothing(fh) != isnothing(Qh)
        throw(ArgumentError("Missingness of fh and Qh must match."))
    end

    x::Vector{UnitRange{Int}} = []
    y::Vector{Vector{RT}} = []
    st = Matrix{Symbol}(undef, 1, 0)
    lt = Matrix{Symbol}(undef, 1, 0)
    co = Matrix{Symbol}(undef, 1, 0)
    lb = Matrix{String}(undef, 1, 0)

    for i = 1:nreps
        local_index = n * (i-1) + index
        push!(x, 1:T)
        push!(y, extract(Y, local_index))
        st = hcat(st, :scatter)
        lt = hcat(lt, :auto)
        co = hcat(co, :grey)
        lb = hcat(lb, i == 1 ? "Observations" : "")
    end

    yhat = extract(f, index)
    Qhat = extract(Q, index)

    push!(x, 1:T, 1:T, 1:T)
    push!(y,
          yhat,
          yhat + factor * sqrt.(Qhat),
          yhat - factor * sqrt.(Qhat))
    st = hcat(st, :line, :line, :line)
    lt = hcat(lt, :solid, :dot, :dot)
    co = hcat(co, :orange, :orange, :orange)
    lb = hcat(lb, "Fitted", "", "")

    if !isnothing(fh)
        h = size(fh, 1)
        fend = f[end][index]
        Qend = Q[end][index,index]
        yfor = extract(fh, index)
        Qfor = extract(Qh, index)

        push!(x, T:T+h, T:T+h, T:T+h)
        push!(y,
              vcat(fend, yfor),
              vcat(fend, yfor) + factor * sqrt.(vcat(Qend, Qfor)),
              vcat(fend, yfor) - factor * sqrt.(vcat(Qend, Qfor)))
        st = hcat(st, :line, :line, :line)
        lt = hcat(lt, :solid, :dot, :dot)
        co = hcat(co, :skyblue, :skyblue, :skyblue)
        lb = hcat(lb, "Forecast", "", "")
    end

    seriestype := st
    linestyle := lt
    color --> co
    label --> lb
    (x, y)
end


"""
    extract(x, index)

Utility function that facilitates fetching the time series as a vector for one
the objects used in the package.

If `x` is a `Vector{Vector{RT}}` it returns `[x[t][index] for t = 1:T]`;
If `x` is a `Vector{CovMat{RT}}` it returns `[x[t][index,index] for t = 1:T]`.
"""
function extract(x::Vector{<: Union{Vector{RT}, CovMat{RT}}},
                 index::Union{<: Integer, BitVector}) where RT <: Real

    if typeof(x) == Vector{Vector{RT}}
        return [x[t][index] for t = 1:size(x,1)]
    else
        return [x[t][index,index] for t = 1:size(x,1)]
    end
end


"""
    check_dimensions(F, G[; V, W, Y])

Utility function that checks for dimension mistmatches in the given arguments
and returns the dimensions for the observations and for the state-space.
"""
function check_dimensions(F::Matrix{RT},
                          G::Matrix{RT};
                          V::Union{CovMat{RT}, Nothing} = nothing,
                          W::Union{CovMat{RT}, Nothing} = nothing,
                          Y::Union{Vector{Vector{RT}}, Nothing} = nothing,
                          nreps::Integer = 1) where RT <: Real

    n = size(F, 1)
    p = size(F, 2)

    if size(G, 1) != size(G, 2)
        throw(DimensionMismatch("G must be a square matrix."))
    end

    if size(G) != (p, p)
        throw(DimensionMismatch("Dimension of G dos not match that of F."))
    end

    if !isnothing(V)
        # Only need to check one dimension since Symmetric guarantees square
        if size(V, 1) != n
            throw(DimensionMismatch("Dimensions of V are a mismatch."))
        end
    end

    if !isnothing(W)
        # Only need to check one dimension since Symmetric guarantees square
        if size(W, 1) != p
            throw(DimensionMismatch("Dimensions of W are a mismatch."))
        end
    end

    if !isnothing(Y)
        for t = 1:size(Y, 1)
            if size(Y[t], 1) != n * nreps
                throw(DimensionMismatch("Dimension of Y incompatible with F."))
            end
        end
    end

    return n, p
end


"""
    simulate(F, G, V, W, θ₀, T[, nreps])

Simulates a time-series Dynamic Linear Model specified by the quadruple
(`F`, `G`, `V`, `W`) with a starting state of `θ₁ <- Nₚ(G θ₀, W)`, with `T`
observations. A parameter `nreps` may be passed indicating the number of
replicates to be generated. Returns the generated `θ` and `y`.

Note that the parametrizations being considered in this package is such that
    `y[t] = F * y[t-1] + ϵ`
and not the notation from West and Harrison (1996) where
    `y[t] = F' * y[t-1] + ϵ`.
"""
function simulate(F::Matrix{RT},
                  G::Matrix{RT},
                  V::CovMat{RT},
                  W::CovMat{RT},
                  θ₀::Vector{RT},
                  T::Integer,
                  nreps::Integer = 1) where RT <: Real

    n, p = check_dimensions(F, G, V=V, W=W)

    θ = Vector{Vector{RT}}(undef, T)
    y = Vector{Vector{RT}}(undef, T)

    ω = MultivariateNormal(zeros(p), W)
    ϵ = MultivariateNormal(zeros(n), V)

    θ[1] = G * θ₀ + rand(ω)
    y[1] = repeat(F * θ[1], nreps) + reshape(rand(ϵ, nreps), nreps * n)
    for t = 2:T
        θ[t] = G * θ[t-1] + rand(ω)
        y[t] = repeat(F * θ[t], nreps) + reshape(rand(ϵ, nreps), nreps * n)
    end

    return θ, y
end


"""
    compute_prior(Y, F, m₀, C₀)

Utility function for computing a smart prior that's not informative and at the
same does not lead to numerical or visualization issues. Only takes effect if
one of `m₀` is `C₀` is `nothing`, otherwise it just returns `m₀` and `C₀`.
"""
function compute_prior(Y::Vector{Vector{RT}},
                       F::Matrix{RT},
                       m₀::Union{Vector{RT}, Nothing},
                       C₀::Union{CovMat{RT}, Nothing}) where RT <: Real

     p = size(F, 2)

     if isnothing(m₀)
         m₀ = zeros(p)
     end

     if isnothing(C₀)
         magic_sdev = maximum(abs.(Y[1] - F * m₀))
         C₀ = Symmetric(diagm(repeat([magic_sdev^2], p)))
     end

     return m₀, C₀
end


"""
    exclude_low_weights(Y, η, ϵ)

Internal function which returns a version of the original `Y` and `η` variables,
containing only the observations with weights above `ϵ`.
"""
@inline function exclude_low_weights(Y::Vector{Vector{RT}},
                                     η::Vector{RT},
                                     ϵ::RT) where RT <: Real

    good_mask = η .> ϵ
    if all(good_mask)
        return Y, η
    end

    return extract(Y, good_mask), η[good_mask]
end


"""
    exclude_low_weights(Y, η, ϵ)

Internal function which returns a version of the original `Y` and `η` variables,
containing only the observations with weights above `ϵ`.
"""
@inline function exclude_low_weights(Y::Vector{Vector{RT}},
                                     η::Matrix{RT},
                                     ϵ::RT) where RT <: Real

    good_mask = η .> ϵ
    if all(good_mask)
        return Y, η
    end

    return extract(Y, good_mask), η[:,good_mask]
end


"""
    kfilter_core(y, F, G, V, W, a, R)

Internal function which actually performs the computation step.
"""
@inline function kfilter_core(y::Vector{RT},
                              F::Matrix{RT},
                              G::Matrix{RT},
                              V::CovMat{RT},
                              W::CovMat{RT},
                              m::Vector{RT},
                              C::CovMat{RT}) where RT <: Real

    a = G * m
    R = Symmetric(G * C * G') + W
    f = F * a
    Q = Symmetric(F * R * F') + V
    A = R * F' * inv(Q)
    m = a + A * (y - f)
    C = R - Symmetric(A * Q * A')

    return a, R, m, C
end


"""
    kfilter_core(y, F, G, V, W, a, R)

Internal function which actually performs the computation step.
"""
@inline function kfilter_core(y::Vector{RT},
                              F::Matrix{RT},
                              G::Matrix{RT},
                              V::CovMat{RT},
                              δ::RT,
                              m::Vector{RT},
                              C::CovMat{RT}) where RT <: Real

    a = G * m
    R = Symmetric(G * C * G') / δ
    f = F * a
    Q = Symmetric(F * R * F') + V
    A = R * F' * inv(Q)
    m = a + A * (y - f)
    C = R - Symmetric(A * Q * A')

    return a, R, m, C
end


"""
    kfilter(Y, F, G, V, W[, m₀, C₀])

Filtering routine for the Dynamic Linear Model (`F`, `G`) where the
observational and evolutional covariance matrices `V` and `W` are known and
constant. `Y` is a vector of observations, containing all observations Y[1],
..., Y[T].  Prior parameters `m₀` and `C₀` may be omitted, in which case
`compute_prior` kicks in to assign a prior.

Returns one-step ahead prior means and covariances `a` and `R`, and online
means and covariances `m` and `C`.
"""
function kfilter(Y::Vector{Vector{RT}},
                 F::Matrix{RT},
                 G::Matrix{RT},
                 V::CovMat{RT},
                 W::CovMat{RT},
                 m₀::Union{Vector{RT}, Nothing} = nothing,
                 C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    n, p = check_dimensions(F, G, V=V, W=W, Y=Y)
    T = size(Y, 1)

    m₀, C₀ = compute_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

    a[1], R[1], m[1], C[1] = kfilter_core(Y[1], F, G, V, W, m₀, C₀)
    for t = 2:T
        a[t], R[t], m[t], C[t] = kfilter_core(Y[t], F, G, V, W, m[t-1], C[t-1])
    end

    return a, R, m, C
end


"""
    kfilter(Y, F, G, V, δ[, m₀, C₀])

Filtering routine for a discount factor Dynamic Linear Model (`F`, `G`) where
the observational covariance matrix `V` is known and constant and evolutional
covariance matrices `W[1], ..., W[T]` are indirectly modelled through a discount
factor `δ`. See West & Harrison (1996) for further information of the discount
factor apporach. `Y` is a vector of observations, containing all observations
Y[1], ..., Y[T].  Prior parameters `m₀` and `C₀` may be omitted, in which case
`compute_prior` kicks in to assign a prior.

Returns one-step ahead prior means and covariances `a` and `R`, and online
means and covariances `m` and `C`.
"""
function kfilter(Y::Vector{Vector{RT}},
                 F::Matrix{RT},
                 G::Matrix{RT},
                 V::CovMat{RT},
                 δ::RT,
                 m₀::Union{Vector{RT}, Nothing} = nothing,
                 C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    n, p = check_dimensions(F, G, V=V, Y=Y)
    T = size(Y, 1)

    m₀, C₀ = compute_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

    a[1], R[1], m[1], C[1] = kfilter_core(Y[1], F, G, V, δ, m₀, C₀)
    for t = 2:T
        a[t], R[t], m[t], C[t] = kfilter_core(Y[t], F, G, V, δ, m[t-1], C[t-1])
    end

    return a, R, m, C
end


"""
    kfilter(Y, F, G, V, δ[, m₀, C₀])

Filtering routine for a discount factor Dynamic Linear Model (`F`, `G`) where
the observational covariance matrices `V[1], …, V[T]` are known and evolutional
covariance matrices `W[1], ..., W[T]` are indirectly modelled through a discount
factor `δ`. See West & Harrison (1996) for further information of the discount
factor apporach. `Y` is a vector of observations, containing all observations
Y[1], ..., Y[T].  Prior parameters `m₀` and `C₀` may be omitted, in which case
`compute_prior` kicks in to assign a prior.

Returns one-step ahead prior means and covariances `a` and `R`, and online
means and covariances `m` and `C`.
"""
function kfilter(Y::Vector{Vector{RT}},
                 F::Matrix{RT},
                 G::Matrix{RT},
                 V::Vector{CovMat{RT}},
                 δ::RT,
                 m₀::Union{Vector{RT}, Nothing} = nothing,
                 C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    n, p = check_dimensions(F, G, V=V, Y=Y)
    T = size(Y, 1)

    m₀, C₀ = compute_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

    a[1], R[1], m[1], C[1] = kfilter_core(Y[1], F, G, V[1], δ, m₀, C₀)
    for t = 2:T
        a[t], R[t], m[t], C[t] = kfilter_core(Y[t], F, G, V[t], δ, m[t-1], C[t-1])
    end

    return a, R, m, C
end


"""
    kfilter(Y, F, G, V, η, δ[, m₀, C₀])

Filtering routine for a discount factor Dynamic Linear Model (`F`, `G`) where
the observational covariance matrix `V` is known and constants and evolutional
covariance matrices `W[1], ..., W[T]` are indirectly modelled through a discount
factor `δ` and observations have replications. `Y` is a vector of observations,
where at each time point there are `nreps` replicates, each with a weight
`η[i]`.  Prior parameters `m₀` and `C₀` may be omitted, in which case
`compute_prior` kicks in to assign a prior.

Returns one-step ahead prior means and covariances `a` and `R`, and online
means and covariances `m` and `C`.
"""
function kfilter(Y::Vector{Vector{RT}},
                 F::Matrix{RT},
                 G::Matrix{RT},
                 V::CovMat{RT},
                 η::Vector{RT},
                 δ::RT,
                 m₀::Union{Vector{RT}, Nothing} = nothing,
                 C₀::Union{CovMat{RT}, Nothing} = nothing;
                 exclude_under::Union{RT, Nothing} = nothing) where RT <: Real

    if !isnothing(exclude_under)
        Y, η = exclude_low_weights(Y, η, exclude_under)
    end

    nreps = size(η, 1)
    n, p = check_dimensions(F, G, V=V, Y=Y, nreps=nreps)
    T = size(Y, 1)

    η /= sum(η)

    F = repeat(F, nreps)
    V = Symmetric(kron(diagm(1 ./ η), V))

    m₀, C₀ = compute_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

    a[1], R[1], m[1], C[1] = kfilter_core(Y[1], F, G, V, δ, m₀, C₀)
    for t = 2:T
        a[t], R[t], m[t], C[t] = kfilter_core(Y[t], F, G, V, δ, m[t-1], C[t-1])
    end

    return a, R, m, C
end


"""
    kfilter(Y, F, G, V, η, δ[, m₀, C₀])

Filtering routine for a discount factor Dynamic Linear Model (`F`, `G`) where
the observational covariance matrix `V` is known and constants and evolutional
covariance matrices `W[1], ..., W[T]` are indirectly modelled through a discount
factor `δ` and observations have replications. `Y` is a vector of observations,
where at each time point there are `nreps` replicates, each with dynamic weights
`η[t,i]`.  Prior parameters `m₀` and `C₀` may be omitted, in which case
`compute_prior` kicks in to assign a prior.

Returns one-step ahead prior means and covariances `a` and `R`, and online
means and covariances `m` and `C`.
"""
function kfilter(Y::Vector{Vector{RT}},
                 F::Matrix{RT},
                 G::Matrix{RT},
                 V::CovMat{RT},
                 η::Matrix{RT},
                 δ::RT,
                 m₀::Union{Vector{RT}, Nothing} = nothing,
                 C₀::Union{CovMat{RT}, Nothing} = nothing;
                 exclude_under::Union{RT, Nothing} = nothing) where RT <: Real

    if !isnothing(exclude_under)
        Y, η = exclude_low_weights(Y, η, exclude_under)
    end

    nreps = size(η, 1)
    n, p = check_dimensions(F, G, V=V, Y=Y, nreps=nreps)
    T = size(Y, 1)

    for t = 1:T
        η[t,:] /= sum(η[t,:])
    end

    F = repeat(F, nreps)

    m₀, C₀ = compute_prior(Y, F, m₀, C₀)

    a = Vector{Vector{RT}}(undef, T)
    m = Vector{Vector{RT}}(undef, T)
    R = Vector{CovMat{RT}}(undef, T)
    C = Vector{CovMat{RT}}(undef, T)

    VV = Symmetric(kron(diagm(1 ./ η[1,:]), V))
    a[1], R[1], m[1], C[1] = kfilter_core(Y[1], F, G, VV, δ, m₀, C₀)
    for t = 2:T
        VV = Symmetric(kron(diagm(1 ./ η[t,:]), V))
        a[t], R[t], m[t], C[t] = kfilter_core(Y[t], F, G, VV, δ, m[t-1], C[t-1])
    end

    return a, R, m, C
end


"""
    ksmoother(G, a, R, m, C)

Filtering routine for a Dynamic Linear Model ( ⋅, `G`), where `a` and `R` are
the filtered one-step ahead prior means and covariances, and `m` and `C` are
the filtered online means and covariances.

Returns the posterior means and covariances `s` and `S`.
"""
function ksmoother(G::Matrix{RT},
                   a::Vector{Vector{RT}},
                   R::Vector{CovMat{RT}},
                   m::Vector{Vector{RT}},
                   C::Vector{CovMat{RT}}) where RT <: Real

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
    evolutional_covariances(Y, F, G, V, δ[, m₀, C₀])

Compute the implied values of the evolutional covariances W[1], ..., W[T] when
considering a discount factor approach, and returns them.
"""
function evolutional_covariances(Y::Vector{Vector{RT}},
                                 F::Matrix{RT},
                                 G::Matrix{RT},
                                 V::CovMat{RT},
                                 δ::RT,
                                 m₀::Union{Vector{RT}, Nothing} = nothing,
                                 C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    T = size(Y, 1)
    m, C = compute_prior(Y, F, m₀, C₀)

    W = Vector{CovMat{RT}}(undef, T)

    for t = 1:T
        _, _, m, C = kfilter_core(Y[t], F, G, V, δ, m, C)
        W[t] = Symmetric(G * C * G') * ((1. - δ) / δ)
    end

    return W
end


"""
    evolutional_covariances(Y, F, G, V, η, δ[, m₀, C₀])

Compute the implied values of the evolutional covariances W[1], ..., W[T] when
considering a discount factor approach, and weighted replicates.
"""
function evolutional_covariances(Y::Vector{Vector{RT}},
                                 F::Matrix{RT},
                                 G::Matrix{RT},
                                 V::CovMat{RT},
                                 η::Vector{RT},
                                 δ::RT,
                                 m₀::Union{Vector{RT}, Nothing} = nothing,
                                 C₀::Union{CovMat{RT}, Nothing} = nothing) where RT <: Real

    nreps = size(η, 1)
    F = repeat(F, nreps)
    V = Symmetric(kron(diagm(1 ./ η), V))
    return evolutional_covariances(Y, F, G, V, δ, m₀, C₀)
end


"""
    fitted(F, V, m, C)

Computes the fitted values for the data, for a model with observational matrix
`F`, evolutional covariance matrix `V`, and state means `m` and `C`. Note that
this can be done with the one-step ahead priors, online parameters or, more
appropriately, smoother results.

Returns observational means and covariances `f` and `Q`.
"""
function fitted(F::Matrix{RT},
                V::CovMat{RT},
                m::Vector{Vector{RT}},
                C::Vector{CovMat{RT}}) where RT <: Real

    T = size(m, 1)

    f = Vector{Vector{RT}}(undef, T)
    Q = Vector{CovMat{RT}}(undef, T)

    for t = 1:T
        f[t] = F * m[t]
        Q[t] = Symmetric(F * C[t] * F') + V
    end

    return f, Q
end


"""
    forecast(F, G, V, W, μ, Σ, h)

Filtering routine for the Dynamic Linear Model (`F`, `G`) where the
observational and evolutional covariance matrices `V` and `W` are known and
constant. `μ` and `Σ` are the mean and covariance matrix for the last state
given the most recent information, and `h` is the forecasting window.

Returns observational means and covariances `f` and `Q`.
"""
function forecast(F::Matrix{RT},
                  G::Matrix{RT},
                  V::CovMat{RT},
                  W::CovMat{RT},
                  μ::Vector{RT},
                  Σ::CovMat{RT},
                  h::Integer) where RT <: Real

    f = Vector{Vector{RT}}(undef, h)
    Q = Vector{CovMat{RT}}(undef, h)

    a, R = μ, Σ
    for t in 1:h
        a = G * a
        R = Symmetric(G * R * G') + W
        f[t] = F * a
        Q[t] = Symmetric(F * R * F') + V
    end

    return f, Q
end


"""
    forecast(F, G, V, δ, μ, Σ, h)

Filtering routine for the Dynamic Linear Model (`F`, `G`) where the
observational covariance matrix `V` is known and constants, and evolutional
covariance matrices W[1], ..., W[T] are indirectly modeled through a discount
factor `δ`. `μ` and `Σ` are the mean and covariance matrix for the last state
given the most recent information, and `h` is the forecasting window.

Returns observational means and covariances `f` and `Q`.
"""
function forecast(F::Matrix{RT},
                  G::Matrix{RT},
                  V::CovMat{RT},
                  δ::RT,
                  μ::Vector{RT},
                  Σ::CovMat{RT},
                  h::Integer) where RT <: Real

    return forecast(F, G, V, Σ / δ, μ, Σ, h)
end


"""
    estimate(Y, F, G, δ[, m₀, C₀; maxit, ϵ])

Obtains maximum a posteriori estimates for the states and observational
covariance matrix for a Dynamic Linear Model (`F`, `G`), considering a discount
factor `δ` for the evolutional covariance matrices. Prior parameters `m₀` and
`C₀` may be omitted, in which case `compute_prior` kicks in to assign a prior.
Parameters `maxit` and `ϵ` control the maximum number of iterations and the
numerical precision for early convergence for the Coordinate Descent algorithm,
respectively.

Returns point estimates for the states `θ` and point estimate for the
covariance matrix `V`. Also returns the number of iterations until convergence.
If negative, it means the algorithm stopped from reaching the maximum number
of iterations.
"""
function estimate(Y::Vector{Vector{RT}},
                  F::Matrix{RT},
                  G::Matrix{RT},
                  δ::RT,
                  m₀::Union{Vector{RT}, Nothing} = nothing,
                  C₀::Union{CovMat{RT}, Nothing} = nothing;
                  maxit::Integer = 50,
                  ϵ::RT = 1e-8) where RT <: Real

    n, p = check_dimensions(F, G, Y=Y)
    T = size(Y, 1)

    # Initialize values
    ϕ = ones(1)
    θ = [Vector{RT}(undef, p) for t = 1:T]

    it_count = zero(maxit)

    # Coordinate descent algorithm: Iterate on conditional maximums
    for it = 1:maxit
        prev_θ = copy(θ)

        # Conditional maximum for the states is the mean from the normal
        # distributions resulting from Kalman smoothing
        a, R, m, C = kfilter(Y, F, G, Symmetric(diagm(ϕ)), δ, m₀, C₀)
        θ, _ = ksmoother(G, a, R, m, C)

        # Conditional maximum for variance comes from the Gamma distribution
        # which is InverseGamma(T - 1, sum((y - θ)^2)) for which the mode is
        # given by the sum((y - F θ)^2) / T.
        ϕ = zero(ϕ)
        for t = 1:T
            ϕ += (Y[t] - F * θ[t]) .^ 2 / T
        end

        # Check for early convergence condition
        it_count = it
        if sum([sum((θ[t] - prev_θ[t]) .^ 2) for t = 1:T]) < p * T * ϵ^2
            break
        end
    end

    return θ, Symmetric(diagm(ϕ)), it_count < maxit ? it_count : -1
end


"""
    estimate(Y, F, G, η, δ[, m₀, C₀; maxit, ϵ])

Obtains maximum a posteriori estimates for the states and observational
covariance matrix for a Dynamic Linear Model (`F`, `G`), considering a discount
factor `δ` for the evolutional covariance matrices, and weighted replicates.
Prior parameters `m₀` and `C₀` may be omitted, in which case `compute_prior`
kicks in to assign a prior. Parameters `maxit` and `ϵ` control the maximum
number of iterations and the numerical precision for early convergence for the
Coordinate Descent algorithm, respectively.

Returns point estimates for the states `θ` and point estimate for the
covariance matrix `V`. Also returns the number of iterations until convergence.
If negative, it means the algorithm stopped from reaching the maximum number
of iterations.
"""
function estimate(Y::Vector{Vector{RT}},
                  F::Matrix{RT},
                  G::Matrix{RT},
                  η::Vector{RT},
                  δ::RT,
                  m₀::Union{Vector{RT}, Nothing} = nothing,
                  C₀::Union{CovMat{RT}, Nothing} = nothing;
                  maxit::Integer = 50,
                  ϵ::RT = 1e-8,
                  exclude_under::Union{RT, Nothing} = nothing) where RT <: Real

    if !isnothing(exclude_under)
        Y, η = exclude_low_weights(Y, η, exclude_under)
    end

    nreps = size(η, 1)
    n, p = check_dimensions(F, G, Y=Y, nreps=nreps)
    T = size(Y, 1)
    FF = repeat(F, nreps)
    η /= sum(η)

    index_map = [(n * (l - 1) + 1):(n * (l - 1) + n) for l = 1:nreps]

    # Initialize values
    ϕ = ones(1)
    θ = [Vector{RT}(undef, p) for t = 1:T]

    it_count = zero(maxit)

    # Coordinate descent algorithm: Iterate on conditional maximums
    for it = 1:maxit
        prev_θ = copy(θ)

        # Conditional maximum for the states is the mean from the normal
        # distributions resulting from Kalman smoothing
        V = Symmetric(kron(diagm(1 ./ η), diagm(ϕ)))
        a, R, m, C = kfilter(Y, FF, G, V, δ, m₀, C₀)
        θ, _ = ksmoother(G, a, R, m, C)

        # Conditional maximum for variance comes from the Gamma distribution
        ϕ = zero(ϕ)
        for t = 1:T
            for i = 1:nreps
                ϕ += η[i] * (Y[t][index_map[i]] - F * θ[t]) .^ 2 / T
            end
        end

        # Check for early convergence condition
        it_count = it
        if sum([sum((θ[t] - prev_θ[t]) .^ 2) for t = 1:T]) < p * T * ϵ^2
            break
        end
    end

    return θ, Symmetric(diagm(ϕ)), it_count < maxit ? it_count : -1
end


"""
    estimate(Y, F, G, η, δ[, m₀, C₀; maxit, ϵ])

Obtains maximum a posteriori estimates for the states and observational
covariance matrix for a Dynamic Linear Model (`F`, `G`), considering a discount
factor `δ` for the evolutional covariance matrices, and dynamically weighted
replicates. Prior parameters `m₀` and `C₀` may be omitted, in which case
`compute_prior` kicks in to assign a prior. Parameters `maxit` and `ϵ` control
the maximum number of iterations and the numerical precision for early
convergence for the Coordinate Descent algorithm, respectively.

Returns point estimates for the states `θ` and point estimate for the
covariance matrix `V`. Also returns the number of iterations until convergence.
If negative, it means the algorithm stopped from reaching the maximum number
of iterations.
"""
function estimate(Y::Vector{Vector{RT}},
                  F::Matrix{RT},
                  G::Matrix{RT},
                  η::Matrix{RT},
                  δ::RT,
                  m₀::Union{Vector{RT}, Nothing} = nothing,
                  C₀::Union{CovMat{RT}, Nothing} = nothing;
                  maxit::Integer = 50,
                  ϵ::RT = 1e-8,
                  exclude_under::Union{RT, Nothing} = nothing) where RT <: Real

    if !isnothing(exclude_under)
        Y, η = exclude_low_weights(Y, η, exclude_under)
    end

    nreps = size(η, 1)
    n, p = check_dimensions(F, G, Y=Y, nreps=nreps)
    T = size(Y, 1)
    FF = repeat(F, nreps)
    η /= sum(η)

    index_map = [(n * (l - 1) + 1):(n * (l - 1) + n) for l = 1:nreps]

    # Initialize values
    ϕ = ones(1)
    θ = [Vector{RT}(undef, p) for t = 1:T]

    it_count = zero(maxit)

    # Coordinate descent algorithm: Iterate on conditional maximums
    for it = 1:maxit
        prev_θ = copy(θ)

        # Conditional maximum for the states is the mean from the normal
        # distributions resulting from Kalman smoothing
        V = [Symmetric(kron(diagm(1 ./ η[t,:]), diagm(ϕ))) for t = 1:T]
        a, R, m, C = kfilter(Y, FF, G, V, δ, m₀, C₀)
        θ, _ = ksmoother(G, a, R, m, C)

        # Conditional maximum for variance comes from the Gamma distribution
        ϕ = zero(ϕ)
        for t = 1:T
            for i = 1:nreps
                ϕ += η[i] * (Y[t][index_map[i]] - F * θ[t]) .^ 2 / T
            end
        end

        # Check for early convergence condition
        it_count = it
        if sum([sum((θ[t] - prev_θ[t]) .^ 2) for t = 1:T]) < p * T * ϵ^2
            break
        end
    end

    return θ, Symmetric(diagm(ϕ)), it_count < maxit ? it_count : -1
end

end # module
