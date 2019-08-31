using DynamicLinearModels
using LinearAlgebra
using Test

DLM = DynamicLinearModels

@testset "Dimension Checking" begin
    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = reshape([1.], 1, 1)
    W = [1. 0.; 0. 0.2]

    @test_throws TypeError DLM.dlm_dimension(F, G, V=V, W=W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0.; 0. 0.2])

    @test DLM.dlm_dimension(F, G, V=V, W=W) == (1, 2)
    @test DLM.dlm_dimension(F, G) == (1, 2)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0. 0.; 0. 2. 0.; 0. 0. 0.2])

    @test_throws DimensionMismatch DLM.dlm_dimension(F, G, V=V, W=W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1.; 0. 1.]
    V = Symmetric([1. 0.; 0. 1.])
    W = Symmetric([1. 0.; 0. 2.])

    @test_throws DimensionMismatch DLM.dlm_dimension(F, G, V=V, W=W)

    F = reshape([1., 0.], 1, 2)
    G = [1. 1. 1.; 0. 1. 1.; 0. 0. 0.]
    V = Symmetric(reshape([1.], 1, 1))
    W = Symmetric([1. 0.; 0. 2.])

    @test_throws DimensionMismatch DLM.dlm_dimension(F, G, V=V, W=W)

    G = [1. 1.; 0. 1.]
    y = reshape([1., 1., 1., 1.], 1, 4)

    @test_throws TypeError DLM.dlm_dimension(F, G, V=V, W=W, Y=y)
    @test_throws TypeError DLM.dlm_dimension(F, G, Y=y)

    y = [[1.], [1.], [1.], [1.]]

    @test DLM.dlm_dimension(F, G, V=V, W=W, Y=y) == (1, 2)
    @test DLM.dlm_dimension(F, G, Y=y) == (1,2)
end
