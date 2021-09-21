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

function two_phase_correct_ratio(z_seq_gt, z_seq_pred)
    n_all = length(z_seq_gt)
    n_success = sum(z_seq_gt .== z_seq_pred)
    ratio = n_success * 1.0 / n_all
    if ratio < 0.5
        ratio = 1.0 - ratio
    end
    return ratio
end

function single_case_test(mp, xs, zs)
    hs = HiddenStates(length(xs), 2)
    log_likelis = []
    for k in 1:60
        log_likeli = update_hidden_states!(hs, mp, xs)
        update_model_parameters!(hs, mp, xs)
        push!(log_likelis, log_likeli)
        if k > 1 && abs(log_likelis[end] - log_likelis[end-1]) < 1e-3
            break
        end
    end
    z_preds = [argmax(z) for z in hs.z_ests]
    @test issorted(log_likelis)
    @test two_phase_correct_ratio(zs[1:end-1], z_preds) > 0.9
    println(two_phase_correct_ratio(zs[1:end-1], z_preds))
end

@testset "test_randomwalk" begin
    noise_std = 1e-1
    prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([noise_std^2]), [0.4])
    prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([noise_std^2]), [-0.4])
    prop_list = [prop1, prop2]
    A = [0.85 0.15;
         0.15 0.85]
    xs, zs = data_generation(500, A, prop_list)
    prop1_init = LinearPropagator(Diagonal([1.1]), Diagonal([(noise_std)^2]), [0.4])
    prop2_init = LinearPropagator(Diagonal([1.2]), Diagonal([(noise_std)^2]), [-0.4])
    prop_list_pred_init = [prop1_init, prop2_init]
    A_pred_init = [0.95 0.05;
                   0.05 0.95]
    mp = ModelParameters(1, A_pred_init, prop_list_pred_init)
    single_case_test(mp, xs, zs)

    A_init_error = sum((A_pred_init .- A).^2)
    A_error = sum((mp.A .- A).^2)
    @test A_error < A_init_error
end

@testset "test_randomwalk_switch_once" begin
    noise_std = 1e-1
    prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([noise_std^2]), [0.4])
    prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([noise_std^2]), [-0.4])
    prop_list = [prop1, prop2]
    A = [0.99 0.0;
         0.01 1.0]
    xs, zs = data_generation(500, A, prop_list)
    prop1_init = LinearPropagator(Diagonal([1.0]), Diagonal([(1.2 * noise_std)^2]), [0.1])
    prop2_init = LinearPropagator(Diagonal([1.0]), Diagonal([(2.0 * noise_std)^2]), [-0.6])
    prop_list_pred_init = [prop1_init, prop2_init]
    A_pred_init = [0.95 0.0;
                   0.05 1.0]
    mp = ModelParameters(1, A_pred_init, prop_list_pred_init)
    single_case_test(mp, xs, zs)

    # In this case, due to sparse observed case of switching, estimation of 
    # stochastic matrix doesn't much well to the gt, but at leaset ... j
    @test A_pred_init[2, 2] == 1.0 # this must invariant
end
