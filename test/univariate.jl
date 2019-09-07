
@testset "Univariate" begin
    @testset "Known V, Known W" begin
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
        # Should be a copy, not using '≈' on purpose
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


    @testset "Known V, Unknown W" begin
        y = [[1.], [2.], [3.]]
        F = hcat(1., 0.)
        G = [1. 1.; 0. 1.]
        V = Symmetric(hcat(1.))

        ## Filter tests

        a, R, m, C = kfilter(y, F, G, V, 0.5)

        @test a[1] ≈ zeros(2)
        @test R[1] ≈ [4. 2.; 2. 2.]
        @test m[end] ≈ [2.947368421052632, 0.9282296650717702]

        ## Smoother tests

        s, S = ksmoother(G, a, R, m, C)

        @test s[1] ≈ [0.9 + 45/990, 0.6290271132376396]
        # Should be a copy, not approximate on purpose
        @test C[end] == S[end]

        ## Fitted tests

        f, Q = fitted(F, V, s, S)

        @test f[1] ≈ [0.9 + 45/990]
        @test Q[1] ≈ hcat(1.5 + 81/990)

        ## Forecast tests

        f, Q = forecast(F, G, V, 0.5, s[end], C[end], 2)

        @test f[2] ≈ [4.803827751196172]
        @test Q[2] ≈ hcat(12.023923444976077)

        ## evolutional_covariances

        W = evolutional_covariances(y, F, G, V, 0.5)

        @test W[1] ≈ [2.8 1.6; 1.6 1.2]
        @test W[end][1,1] ≈ 2.258373205741627
        @test W[end][1,2] ≈ 0.9952153110047848
        @test W[end][2,2] ≈ 0.5741626794258374
    end


    @testset "Unknown V, Unknown W" begin
        y = [[1.1], [2.1], [2.9], [4.05]]
        F = hcat(1., 0.)
        G = [1. 1.; 0. 1.]

        θ, Σ, it = estimate(y, F, G, 0.8)
        f, Q = fitted(F, Σ, θ, evolutional_covariances(y, F, G, Σ, 0.8))

        @test f[1] ≈ [1.0941182414662356]
        @test f[end] ≈ [3.9948460619971886]
        @test Q[end] ≈ hcat(0.005579344922273387)
    end
end
