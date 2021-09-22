struct Sequence{N}
    data::Matrix{Float64}
    n_seq::Int
end
function Sequence(data::Vector)
    n_seq = length(data)
    n_dim = length(data[1])
    mat = zeros(n_dim, n_seq)
    for i in 1:n_seq
        mat[:, i] = data[i]
    end
    Sequence(mat)
end
function Sequence(data::Matrix) 
    n_dim, n_seq = size(data)
    Sequence{n_dim}(data, n_seq)
end
Base.getindex(seq::Sequence, index::Int) = seq.data[:, index]

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

mutable struct HiddenStates{M}
    n_seq::Int
    z_ests::Vector{MVector{M, Float64}}
    zz_ests::Vector{MMatrix{M, M, Float64}}
    alphas::Vector{MVector{M, Float64}}
    betas::Vector{MVector{M, Float64}}
    c_seq::Vector{Float64}
end
function HiddenStates(n_seq::Int, n_phase::Int)
    M = n_phase
    z_ests = [zeros(M) for _ in 1:n_seq-1]
    zz_ests = [zeros(M, M) for _ in 1:n_seq-2]
    alphas = [MVector{M, Float64}(zeros(M)) for _ in 1:n_seq-1]
    betas = [MVector{M, Float64}(zeros(M)) for _ in 1:n_seq-1]
    c_seq = zeros(n_seq)
    HiddenStates{M}(n_seq, z_ests, zz_ests, alphas, betas, c_seq)
end
