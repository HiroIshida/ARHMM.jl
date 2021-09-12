module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export HiddenStates, ModelParameters

const Sequence{N} = Vector{SVector{N, Float64}}

mutable struct ModelParameters{N, M}
    A::MMatrix{M, M}
    A_list::SVector{M, Matrix{Float64}} # TODO rename ? F
    Sigma_list::SVector{M, Matrix{Float64}}
    pmf_z0::SVector{M, Float64} #TODO should be Dilechlet
end
function ModelParameters(n_dim, n_phase)
    A = Diagonal([1.0 for _ in 1:n_phase])
    A_list = [Diagonal([1.0 for _ in 1:n_dim]) for _ in 1:n_phase]
    Sigma_list = [Diagonal([1 for _ in 1:n_dim]) for _ in 1:n_phase]
    pmf_z0 = zeros(n_phase); pmf_z0[1] = 1.0 # because initial phase must be phase 1
    ModelParameters{n_dim, n_phase}(A, A_list, Sigma_list, pmf_z0)
end

mutable struct HiddenStates{N, M}
    n_seq::Int
    alpha_cache_vec::Vector{MVector{M, Float64}}
    beta_cache_vec::Vector{MVector{M, Float64}}
end

function HiddenStates(sequence::Sequence{N}, n_phase) where N
    n_seq = length(sequence)
    alphas = [MVector{n_phase, Float64}(undef) for _ in 1:n_seq]
    betas = [MVector{n_phase, Float64}(undef) for _ in 1:n_seq]
    HiddenStates{N, n_phase}(n_seq, alphas, betas)
end

function prob_propagation(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, x_pre, x) where {N, M}
    gen = (pdf(MvNormal(params.A_list[i]*x_pre, params.Sigma_list[i]), x) for i in 1:M)
    SVector{M, Float64}(gen)
end

function alpha_forward!(hs::HiddenStates{N, M}, params::ModelParameters{N, M}, seq) where {N, M}
    n_seq = length(seq)
    x1, x2 = seq[1:2]
    println(prob_propagation(hs, params, x1, x2))
    println(params.pmf_z0)
    alpha = prob_propagation(hs, params, x1, x2) .* params.pmf_z0
    hs.alpha_cache_vec[1] = alpha
    for t in 2:n_seq-1
        integral_term = (dot(params.A[i, :], alpha) for i in 1:M)
        xt, xtt = seq[t:t+1]
        alpha = prob_propagation(hs, params, xt, xtt) .* integral_term
        hs.alpha_cache_vec[t] = alpha
    end
end

end # module
