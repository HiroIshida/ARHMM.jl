module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export create_dataset
include("sample_dataset.jl")

export LinearPropagator, FixedPropagator, transition_prob
include("propagator.jl")

export ModelParameters, compute_hidden_states, update_model_parameters!

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

function update_pmf_z1!(mp::ModelParameters{N, M}, z_ests, zz_ests, x_seq) where {N, M}
    mp.pmf_z1 = z_ests[1] / sum(z_ests[1])
end

function update_A!(mp::ModelParameters{N, M}, z_ests, zz_ests, x_seq) where {N, M}
    n_seq = length(z_ests) + 1
    A_new = zeros(M, M)
    for t in 1:n_seq - 2
        for i in 1:M
            for j in 1:M
                # Note that our stochastic matrix is different (transposed) from
                # the one in PRML
                A_new[j, i] += zz_ests[t][i, j]
            end
        end
    end
    for j in 1:M # normalize
        A_new[:, j] /= sum(A_new[:, j])
    end
    mp.A = A_new
end

function update_prop_list!(mp::ModelParameters{N, M}, z_ests, zz_ests, x_seq) where {N, M}
    for i in 1:M
        w_seq = [z_est[i] for z_est in z_ests]
        fit!(mp.prop_list[i], x_seq, w_seq)
    end
end

function update_model_parameters!(mp::ModelParameters{N, M}, z_ests, zz_ests, x_seq) where {N, M}
    update_pmf_z1!(mp, z_ests, zz_ests, x_seq)
    update_A!(mp, z_ests, zz_ests, x_seq)
    update_prop_list!(mp, z_ests, zz_ests, x_seq)
end

function probs_linear_prop(mp::ModelParameters, x_pre, x)
    gen = (transition_prob(prop, x_pre, x) for prop in mp.prop_list)
end

function compute_hidden_states(mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    alphas, c_seq = alpha_forward(mp, seq)
    betas = beta_backward(mp, seq, c_seq)
    z_ests = [a .* b for (a, b) in zip(alphas, betas)] # γ in PRML
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

function alpha_forward(mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    n_seq = length(seq)
    alphas = [zeros(M) for _ in 1:n_seq-1]
    c_seq = zeros(n_seq)
    c_seq[1] = 1.0

    x1, x2 = seq[1:2]
    px1 = 1.0 # deterministic x 
    tmp = probs_linear_prop(mp, x1, x2) .* mp.pmf_z1 * px1
    c_seq[2] = sum(tmp)
    alphas[1] = tmp/c_seq[2]

    for t in 2:n_seq - 1
        x_t, x_tp1 = seq[t:t+1]
        for i in 1:M
            integral_term = sum(mp.A[i, j] * alphas[t-1][j] for j in 1:M)
            alphas[t][i] = transition_prob(mp.prop_list[i], x_t, x_tp1) * integral_term
        end
        c_seq[t+1] = sum(alphas[t])
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
