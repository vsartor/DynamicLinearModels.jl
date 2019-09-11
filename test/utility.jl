
@testset "Utility Functions" begin
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

        # Cover exception for when G is not square
        F = reshape([1., 0.], 1, 2)
        G = [1. 1. 1.; 0. 1. 0.]
        V = Symmetric(reshape([1.], 1, 1))
        W = Symmetric([1. 0.; 0. 0.2])
        @test_throws DimensionMismatch check_dimensions(F, G, V=V, W=W)

        G = [1. 1.; 0. 1.]
        y = reshape([1., 1., 1., 1.], 4, 1)

        @test check_dimensions(F, G, V=V, W=W, Y=y) == (1, 2)

        y = [[1.], [1.], [1.], [1.]]

        @test_throws TypeError check_dimensions(F, G, V=V, W=W, Y=y) == (1, 2)
        @test_throws TypeError check_dimensions(F, G, Y=y) == (1,2)
    end


    @testset "Prior Computation" begin
        y_a = hcat(1.)
        F_a = hcat(1., 0.)
        m_a = ones(2)
        C_a = Symmetric(ones(2, 2))

        @test compute_prior(y_a, F_a, nothing, nothing) == (zeros(2), I(2))
        @test compute_prior(y_a, F_a, m_a, nothing) == (m_a, zeros(2, 2))
        @test compute_prior(y_a, F_a, nothing, C_a) == (zeros(2), ones(2, 2))
        @test compute_prior(y_a, F_a, m_a, C_a) == (m_a, C_a)

        # Make sure only Symmetric is allowed
        @test_throws MethodError compute_prior(y_a, F_a, nothing, ones(2, 2))

        y_b = hcat(2., 3.)
        F_b = [1. 0. .5; 0. 1. .5]
        m_b = [1., 1., .5]
        C_b = Symmetric(ones(3, 3))

        @test compute_prior(y_b, F_b, nothing, nothing) == (zeros(3), 9 * I(3))
        @test compute_prior(y_b, F_b, m_b, nothing) == (m_b, 3.0625 * I(3))
        @test compute_prior(y_b, F_b, nothing, C_b) == (zeros(3), ones(3, 3))
        @test compute_prior(y_b, F_b, m_b, C_b) == (m_b, C_b)
    end


    @testset "Plot Recipe" begin
        F = reshape([1., 0.], 1, 2)
        G = [1. 1.; 0. 1.]
        V = Symmetric(reshape([12.], 1, 1))
        W = Symmetric([1. 0.; 0. 0.2])
        θ₀ = zeros(2)
        T = 36
        θ, y = simulate(F, G, V, W, θ₀, T)
        a, R, m, C = kfilter(y, F, G, V, W)
        s, S = ksmoother(G, a, R, m, C)
        f, Q = fitted(F, V, s, S)
        fh, Qh = forecast(F, G, V, W, m[end,:], C[end], 10)

        # When applying the recipe, this function is required to check validity
        # of the recipe keys. To avoid importing Plots, extend it to always return
        # true.
        function RecipesBase.is_key_supported(x)
            return true
        end

        apply_recipe = RecipesBase.apply_recipe

        @test size(apply_recipe(Dict{Symbol,Any}(), DLMPlot(), y, f, Q
                               )[1].plotattributes[:color]) == (1,4)
        @test size(apply_recipe(Dict{Symbol,Any}(), DLMPlot(), y, f, Q, fh, Qh
                               )[1].plotattributes[:color]) == (1,7)

        @test_throws ArgumentError apply_recipe(Dict{Symbol,Any}(), DLMPlot(), y, f, Q, nothing, Qh)
        @test_throws ArgumentError apply_recipe(Dict{Symbol,Any}(), DLMPlot(), y, f, Q, fh, nothing)
    end
end
