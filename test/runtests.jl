using DynamicLinearModels
using LinearAlgebra
using Test

@testset "DynamicLinearModels" begin

@testset "Dimension Checking" begin
    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = reshape([1.], 1, 1)
    W = [1. 0.; 0. 0.2]

    @test_throws TypeError check_dimensions(F, G, V=V, W=W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0.; 0. 0.2])

    @test check_dimensions(F, G, V=V, W=W) == (1, 2)
    @test check_dimensions(F, G) == (1, 2)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0. 0.; 0. 2. 0.; 0. 0. 0.2])

    @test_throws DimensionMismatch check_dimensions(F, G, V=V, W=W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric([1. 0.; 0. 1.])
    W = Symmetric([1. 0.; 0. 2.])

    @test_throws DimensionMismatch check_dimensions(F, G, V=V, W=W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1. 1.; 0. 1. 1.; 0. 0. 0.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0.; 0. 2.])

    @test_throws DimensionMismatch check_dimensions(F, G, V=V, W=W)

    G = [1. 1.; 0. 1.]
    y = reshape([1., 1., 1., 1.], 1, 4)

    @test_throws TypeError check_dimensions(F, G, V=V, W=W, Y=y)
    @test_throws TypeError check_dimensions(F, G, Y=y)

    y = [[1.], [1.], [1.], [1.]]

    @test check_dimensions(F, G, V=V, W=W, Y=y) == (1, 2)
    @test check_dimensions(F, G, Y=y) == (1,2)
end


@testset "Prior Computation" begin
    y_a = [[1.]]
    F_a = hcat(1., 0.)
    m_a = ones(2)
    C_a = Symmetric(ones(2, 2))

    @test compute_prior(y_a, F_a, nothing, nothing) == (zeros(2), I(2))
    @test compute_prior(y_a, F_a, m_a, nothing) == (m_a, zeros(2, 2))
    @test compute_prior(y_a, F_a, nothing, C_a) == (zeros(2), ones(2, 2))
    @test compute_prior(y_a, F_a, m_a, C_a) == (m_a, C_a)

    # Make sure only Symmetric is allowed
    @test_throws MethodError compute_prior(y_a, F_a, nothing, ones(2, 2))

    y_b = [[2., 3.]]
    F_b = [1. 0. .5; 0. 1. .5]
    m_b = [1., 1., .5]
    C_b = Symmetric(ones(3, 3))

    @test compute_prior(y_b, F_b, nothing, nothing) == (zeros(3), 9 * I(3))
    @test compute_prior(y_b, F_b, m_b, nothing) == (m_b, 3.0625 * I(3))
    @test compute_prior(y_b, F_b, nothing, C_b) == (zeros(3), ones(3, 3))
    @test compute_prior(y_b, F_b, m_b, C_b) == (m_b, C_b)
end


@testset "Estimation: Known V and W" begin
    y = [[1.], [2.], [3.]]
    F = hcat(1., 0.)
    G = [1. 1.; 0. 1.]
    V = Symmetric(hcat(1.))
    W = Symmetric(diagm([1., 1.]))

    ## Filter tests

    a, R, m, C = kfilter(y, F, G, V, W)

    @test a[1] ≈ zeros(2)
    @test R[1] ≈ [3. 1.; 1. 2.]
    @test m[end] ≈ [2 + 900/999, 882/999]

    ## Smoother tests

    s, S = ksmoother(G, a, R, m, C)

    @test s[1] ≈ [954/999, 783/999]
    # Should be a copy, not approximate on purpose
    @test C[end] == S[end]

    ## Fitted tests

    f, Q = fitted(F, V, s, S)

    @test f[1] ≈ [954 / 999]
    @test Q[1] ≈ hcat(1 + 549/999)

    ## Forecast tests

    f, Q = forecast(F, G, V, W, s[end], C[end], 2)

    @test f[2] ≈ [4 + 2/3]
    @test Q[2] ≈ hcat(14 + 1/3)
end

end
