module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export create_dataset
/include("sample_dataset.jl")

export LinearPropagator, FixedPropagator, transition_prob
include("propagator.jl")

export ModelParameters, emrun!

const Sequence{N} = Vector{SVector{N, Float64}}

mutable struct ModelParameters{N, M}
    A::MMatrix{M, M, Float64}
    prop_list::Vector{Propagator{N}}
    pmf_z1::SVector{M, Float64} #TODO should be Dilechlet
end
function ModelParameters(n_dim, A, prop_list)
    n_phase = size(A)[1]
    pmf_z1 = zeros(n_phase); pmf_z1[1] = 1.0 # because initial phase must be phase 1
    ModelParameters{n_dim, n_phase}(A, prop_list, pmf_z1)
end

function emrun!(mp::ModelParameters{N, M}, seq::Sequence{N}, iter=20) where {N, M}
    z_ests = nothing
    for _ in 1:iter
        z_ests, zz_ests, log_likeli = compute_hidden_states(mp, seq)
        update_model_parameters!(mp, z_ests, zz_ests)
        println(log_likeli)
    end
    return z_ests
end

function update_model_parameters!(mp::ModelParameters{N, M}, z_ests, zz_ests) where {N, M}
    n_seq = length(z_ests) + 1

    # compute new pmf_z1
    pmf_z1_new = z_ests[1] / sum(z_ests[1])

    # compute new A
    A_new = zeros(M, M)
    for t in 1:n_seq - 2
        for i in 1:M
            for j in 1:M
                A_new[i, j] += zz_ests[t][i, j]
            end
        end
    end
    for j in 1:M # normalize
        A_new[:, j] /= sum(A_new[:, j])
    end

    # update
    mp.pmf_z1 = pmf_z1_new
    mp.A = A_new
end

function probs_linear_prop(mp::ModelParameters, x_pre, x)
    gen = (transition_prob(prop, x_pre, x) for prop in mp.prop_list)
end

function compute_hidden_states(mp::ModelParameters{N, M}, seq::Sequence{N}, scaled=true) where {N, M}
    alphas, c_seq = alpha_forward(mp, seq, scaled)
    betas = beta_backward(mp, seq, c_seq)
    z_ests = [a .* b/(scaled ? 1.0 : sum(a.*b)) for (a, b) in zip(alphas, betas)] # γ in PRML
    n_seq = length(seq)

    zz_ests = [MMatrix{M, M, Float64}(undef) for _ in 1:n_seq-2] # ξ in PRML
    for t in 1:n_seq - 2
        # i is index for t, j for (t+1)
        for i in 1:M
            for j in 1:M
                x_t, x_tt = seq[t+1:t+2]
                trans_prob = transition_prob(mp.prop_list[j], x_t, x_tt)
                tmp = alphas[t][i] * mp.A[j, i] * trans_prob * betas[t+1][j]
                zz_ests[t][i, j] = tmp/c_seq[t+2]
            end
        end
    end

    # compute log_likelihood
    log_likeli = sum(log(c) for c in c_seq)

    return z_ests, zz_ests, log_likeli
end

function alpha_forward(mp::ModelParameters{N, M}, seq::Sequence{N}, scaled) where {N, M}
    n_seq = length(seq)
    alphas = [zeros(M) for _ in 1:n_seq-1]
    c_seq = zeros(n_seq)
    c_seq[1] = (scaled ? 1.0 : 1.0)

    x1, x2 = seq[1:2]
    px1 = 1.0 # deterministic x 
    tmp = probs_linear_prop(mp, x1, x2) .* mp.pmf_z1 * px1
    c_seq[2] = (scaled ? sum(tmp) : 1.0)
    alphas[1] = tmp/c_seq[2]

    for t in 2:n_seq - 1
        x_t, x_tp1 = seq[t:t+1]
        for i in 1:M
            integral_term = sum(mp.A[i, j] * alphas[t-1][j] for j in 1:M)
            alphas[t][i] = transition_prob(mp.prop_list[i], x_t, x_tp1) * integral_term
        end
        c_seq[t+1] = (scaled ? sum(alphas[t]) : 1.0)
        alphas[t] /= c_seq[t+1]
    end
    return alphas, c_seq
end

function beta_backward(mp::ModelParameters{N, M}, seq::Sequence{N}, c_seq) where {N, M}
    n_seq = length(seq)
    betas = [zeros(M) for _ in 1:n_seq-1]

    betas[n_seq - 1] = ones(M)
    for t in length(seq)-2:-1:1
        x_tp1 = seq[t+1]
        x_tp2 = seq[t+2]
        for j in 1:M # phase at t
            sum = 0.0
            for i in 1:M # phase at t+1
                sum +=mp.A[i, j] * transition_prob(mp.prop_list[i], x_tp1, x_tp2) * betas[t+1][i]
            end
            betas[t][j] = sum
            betas[t][j] /= c_seq[t+2]
        end
   end
   return betas
end

end # module
