module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export HiddenStates, ModelParameters, update_hidden_states!

const Sequence{N} = Vector{SVector{N, Float64}}

mutable struct ModelParameters{N, M}
    A::MMatrix{M, M}
    A_list::SVector{M, Matrix{Float64}} # TODO rename ? F
    Sigma_list::SVector{M, Matrix{Float64}}
    pmf_z1::SVector{M, Float64} #TODO should be Dilechlet
end
function ModelParameters(n_dim, n_phase)
    A = Diagonal([1.0 for _ in 1:n_phase])
    A_list = [Diagonal([1.0 for _ in 1:n_dim]) for _ in 1:n_phase]
    Sigma_list = [Diagonal([1 for _ in 1:n_dim]) for _ in 1:n_phase]
    pmf_z1 = zeros(n_phase); pmf_z1[1] = 1.0 # because initial phase must be phase 1
    ModelParameters{n_dim, n_phase}(A, A_list, Sigma_list, pmf_z1)
end

mutable struct HiddenStates{N, M}
    n_seq::Int
    alpha_cache_vec::Vector{MVector{M, Float64}}
    beta_cache_vec::Vector{MVector{M, Float64}}
    scaling_cache_vec::Vector{Float64}
end

function HiddenStates(sequence::Sequence{N}, n_phase) where N
    n_seq = length(sequence)
    alphas = [MVector{n_phase, Float64}(undef) for _ in 1:n_seq-1]
    betas = [MVector{n_phase, Float64}(undef) for _ in 1:n_seq-1]
    scales = zeros(n_seq)
    HiddenStates{N, n_phase}(n_seq, alphas, betas, scales)
end

function update_model_parameters!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    function gamma(t)
        alpha = hs.alpha_cache_vec[t]
        beta = hs.beta_cache_vec[t]
        alpha .* beta
    end

    function xi(t, i, j)
        alpha = hs.alpha_cache_vec[t]
        beta = hs.beta_cache_vec[t + 1]
        x_pre, x = seq[t:t+1]
        tmp = prob_linear_prop(params.A_list[j], params.Sigma_list[j], x_pre, x)
        alpha[i] * params.A[j, i] * tmp * beta[j] / hs.scaling_cache_vec[t+2]
    end
    
    # compute new pmf_z1
    alpha1, beta1 = hs.alpha_cache_vec[1], hs.beta_cache_vec[1]
    gamma1 = (alpha1 .* beta1)/dot(alpha1, beta1)
    pmf_z1_new = gamma1 / sum(gamma1)

    # compute new A
    A_new = MMatrix{M, M, Float64}(undef)
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

prob_linear_prop(Phi, Sigma, x_pre, x) = pdf(MvNormal(Phi * x_pre, Sigma), x)
function probs_linear_prop(mp::ModelParameters{N, M}, x_pre, x) where {N, M} 
    gen = (prob_linear_prop(mp.A_list[i], mp.Sigma_list[i], x_pre, x) for i in 1:M)
end

function update_hidden_states!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq) where {N, M}!
    alpha_forward!(hs, params, seq)
    beta_backward!(hs, params, seq)
end

function alpha_forward!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq) where {N, M}
    n_seq = length(seq)
    x1, x2 = seq[1:2]
    px1 = 1.0 # TODO this is a special case when observation of x has no uncertainty
    alpha1 = probs_linear_prop(params, x1, x2) .* params.pmf_z1 * px1
    c1 = 1.0
    c2 = sum(probs_linear_prop(params, x1, x2) .* params.pmf_z1)

    alpha_hat_1 = alpha1/c2
    hs.alpha_cache_vec[1] = alpha_hat_1
    hs.scaling_cache_vec[1] = c1
    hs.scaling_cache_vec[2] = c2

    alpha_hat_tm1 = alpha_hat_1

    for t in 2:n_seq-1
        integral_term = (dot(params.A[i, :], alpha_hat_tm1) for i in 1:M)
        xt, xtt = seq[t:t+1]
        qt = probs_linear_prop(params, xt, xtt) .* integral_term
        c_tt = sum(qt)
        alpha_hat = qt/c_tt
        hs.alpha_cache_vec[t] = alpha_hat
        hs.scaling_cache_vec[t+1] = c_tt
    end
end

function beta_backward!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq) where {N, M}
    beta = MVector{M, Float64}([1.0 for _ in 1:M]) # β(n_seq-1)
    beta_hat = beta # c=1.0
    hs.beta_cache_vec[end] = beta
    for t in hs.n_seq-2:-1:1
        xtt, xttt = seq[t+1:t+2]
        beta_hat_new = MVector{M, Float64}([0.0 for _ in 1:M])
        for i in 1:M
            # TODO logic is bit dirty
            for j in 1:M
                prob_prob = prob_linear_prop(params.A_list[j], params.Sigma_list[j], xtt, xttt)
                beta_hat_new[i] += params.A[j, i] * prob_prob * beta_hat[j]
            end
        end
        beta_hat_new /= hs.scaling_cache_vec[t+2]
        hs.beta_cache_vec[t] = beta_hat_new
        beta_hat = beta_hat_new
    end
end

end # module
