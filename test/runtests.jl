using DynamicLinearModels
using LinearAlgebra
using RecipesBase
using Test

@testset "DynamicLinearModels" begin
    include("utility.jl")
    include("univariate.jl")
    include("weighted.jl")
    include("dynamic.jl")
end
