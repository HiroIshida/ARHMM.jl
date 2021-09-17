module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export LinearPropagator, FixedPropagator, transition_prob
include("propagator.jl")

export HiddenStates, ModelParameters, update_hidden_states!

const Sequence{N} = Vector{SVector{N, Float64}}

mutable struct ModelParameters{N, M}
    A::MMatrix{M, M}
    prop_list::Vector{Propagator{N}}
    pmf_z1::SVector{M, Float64} #TODO should be Dilechlet
end
function ModelParameters(n_dim, A, prop_list)
    n_phase = size(A)[1]
    pmf_z1 = zeros(n_phase); pmf_z1[1] = 1.0 # because initial phase must be phase 1
    ModelParameters{n_dim, n_phase}(A, prop_list, pmf_z1)
end

mutable struct HiddenStates{N, M}
    n_seq::Int
    alpha_seq::Vector{MVector{M, Float64}}
    beta_seq::Vector{MVector{M, Float64}}
    scaling_cache_vec::Vector{Float64}
end

function HiddenStates(sequence::Sequence{N}, n_phase) where N
    n_seq = length(sequence)
    alphas = [MVector{n_phase, Float64}(undef) for _ in 1:n_seq-1]
    betas = [MVector{n_phase, Float64}(undef) for _ in 1:n_seq-1]
    scales = zeros(n_seq)
    HiddenStates{N, n_phase}(n_seq, alphas, betas, scales)
end

#=
function update_model_parameters!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    function gamma(t)
        alpha = hs.alpha_seq[t]
        beta = hs.beta_seq[t]
        alpha .* beta
    end

    function xi(t, i, j)
        alpha = hs.alpha_seq[t]
        beta = hs.beta_seq[t + 1]
        x_pre, x = seq[t:t+1]
        tmp = prob_linear_prop(params.A_list[j], params.Sigma_list[j], x_pre, x)
        ret = alpha[i] * params.A[j, i] * tmp * beta[j] / hs.scaling_cache_vec[t+2]
    end
    
    # compute new pmf_z1
    alpha1, beta1 = hs.alpha_seq[1], hs.beta_seq[1]
    gamma1 = (alpha1 .* beta1)/dot(alpha1, beta1)
    pmf_z1_new = gamma1 / sum(gamma1)

    # compute new A
    println(params.A)
    A_new = zeros(M, M)
    for t in 1:hs.n_seq - 2
        for i in 1:M
            for j in 1:M
                A_new[i, j] += xi(t, i, j)
            end
        end
    end
    for j in 1:M # normalize
        A_new[:, j] /= sum(A_new[:, j])
    end

    # update
    params.pmf_z1 = pmf_z1_new
    params.A = A_new

end
=#

function probs_linear_prop(mp::ModelParameters, x_pre, x)
    gen = (transition_prob(prop, x_pre, x) for prop in mp.prop_list)
end

function update_hidden_states!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq) where {N, M}!
    scaled_alpha_forward!(hs, params, seq)
    scaled_beta_backward!(hs, params, seq)
end

function alpha_forward(mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    n_seq = length(seq)
    alpha_seq = [zeros(M) for _ in 1:n_seq-1]

    x1, x2 = seq[1:2]
    px1 = 1.0 # deterministic x 
    alpha_seq[1] = probs_linear_prop(mp, x1, x2) .* mp.pmf_z1 * px1

    for t in 2:n_seq - 1
        alpha_tm1 = alpha_seq[t-1]
        alpha_t = alpha_seq[t]
        x_t, x_tp1 = seq[t:t+1]
        for i in 1:M
            integral_term = sum(mp.A[i, j] * alpha_tm1[j] for j in 1:M)
            alpha_t[i] = transition_prob(mp.prop_list[i], x_t, x_tp1) * integral_term
        end
    end
    return alpha_seq
end

function beta_backward(mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    n_seq = length(seq)
    beta_seq = [zeros(M) for _ in 1:n_seq-1]

    beta_seq[n_seq - 1] = ones(M)
    for t in length(seq)-2:-1:1
        x_tp1 = seq[t+1]
        x_tp2 = seq[t+2]
        for j in 1:M # phase at t
            sum = 0.0
            for i in 1:M # phase at t+1
                sum +=mp.A[i, j] * transition_prob(mp.prop_list[i], x_tp1, x_tp2) * beta_seq[t+1][i]
            end
            beta_seq[t][j] = sum
        end
   end
   return beta_seq
end

end # module
