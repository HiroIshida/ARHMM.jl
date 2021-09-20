#using Revise
using Random
using LinearAlgebra
using ARHMM
using Test 

using StaticArrays
using Distributions

Random.seed!(0)

function data_generation(n, A, prop_list)
    x = SVector{1, Float64}([0.0])
    z = 1
    xs = [x]
    zs = [z]
    for i in 1:n
        cat = Categorical(A[:, z])
        z_next = rand(cat)
        x_next = prop_list[z](x)
        x, z = x_next, z_next
        push!(xs, SVector{1, Float64}(x))
        push!(zs, z)
    end
    return xs, zs
end

@testset "test_randomwalk" begin
    for with_noise in [false, true]
        noise_std = with_noise ? 0.1 : 1e-4
        prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([noise_std^2]), [0.4])
        prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([noise_std^2]), [-0.4])
        prop_list = [prop1, prop2]
        A = [0.85 0.15;
             0.15 0.85]
        xs, zs = data_generation(3000, A, prop_list)
        A_pred_init = [0.5 0.5;
                       0.5 0.5]
        mp = ModelParameters(1, A_pred_init, prop_list)

        z_ests = nothing
        log_likelis = []
        for _ in 1:2 # if more than 2, usually likelihood will be static
            z_ests, zz_ests, log_likeli = compute_hidden_states(mp, xs)
            update_model_parameters!(mp, z_ests, zz_ests, xs)
            push!(log_likelis, log_likeli)
        end
        z_preds = [argmax(z) for z in z_ests]
        @test issorted(log_likelis)
        if ~with_noise 
            # With no-noise case hidden state should match exactly
            @test zs[1:end-1] == z_preds
        end
    end
end
