using DynamicLinearModels
using LinearAlgebra
using Test

@testset "Dimension Checking" begin
    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = reshape([1.], 1, 1)
    W = [1. 0.; 0. 0.2]

    @test_throws MethodError dlm_dimension(F, G, V, W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0.; 0. 0.2])

    @test dlm_dimension(F, G, V, W) == (1, 2)
    @test dlm_dimension(F, G) == (1, 2)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0. 0.; 0. 2. 0.; 0. 0. 0.2])

    @test_throws DimensionMismatch dlm_dimension(F, G, V, W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric([1. 0.; 0. 1.])
    W = Symmetric([1. 0.; 0. 2.])

    @test_throws DimensionMismatch dlm_dimension(F, G, V, W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1. 1.; 0. 1. 1.; 0. 0. 0.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0.; 0. 2.])

    @test_throws DimensionMismatch dlm_dimension(F, G, V, W)
end
