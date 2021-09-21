using Revise
using Random
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

Random.seed!(0)
states_list, phases_list = create_dataset(80)

dim = 4
xs_list = [[SVector{dim, Float64}(vcat(seq[t].x, seq[t+1].x)) for t in 1:length(seq)-1] for seq in states_list]
prop1 = LinearPropagator(Diagonal(ones(dim)), Diagonal(ones(dim) * 1.0), zeros(dim))
prop2 = LinearPropagator(Diagonal(ones(dim)), Diagonal(ones(dim) * 1.0), zeros(dim))
prop_list = [prop1, prop2]
A = [0.99 0.0;
     0.01 1.0]
mp = ModelParameters(dim, A, prop_list)

function train!(mp, xs_list, hs_list)
    for i in 1:80
        println(i)
        for (hs, xs) in zip(hs_list, xs_list)
            log_likeli = update_hidden_states!(hs, mp, xs)
        end
        update_model_parameters!(hs_list, mp, xs_list)
    end
end
hs_list = [HiddenStates(length(xs), 2) for xs in xs_list]
@time train!(mp, xs_list, hs_list)

using Plots
index = 37
plot([z[1] for z in hs_list[index].z_ests])
plot!(phases_list[index])
