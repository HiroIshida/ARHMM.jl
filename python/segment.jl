using LinearAlgebra
using StaticArrays
using ARHMM
using JSON
using PyCall

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
def dump_pickle(obj, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)
"""
load_pickle = py"load_pickle"
dump_pickle = py"dump_pickle"

function get_sequences(cache_file_name)
    chunk = load_pickle(cache_file_name)
    cmd_seqs_tmp = chunk["cmd_seqs"] # mabe numpy
    cmd_seqs = [Matrix(cmd_seq) for cmd_seq in cmd_seqs_tmp]
    _, n_dim = size(cmd_seqs[1])
    seqs = []
    for cmd_seq in cmd_seqs
        seq = SVector{n_dim, Float64}[]
        n_seq, n_dim = size(cmd_seq)
        for t in 1:n_seq
            push!(seq, cmd_seq[t, :])
        end
        push!(seqs, seq)
    end
    return seqs
end

function create_markov_matrix(n_phase)
    A = Matrix(Diagonal([0.99 for _ in 1:n_phase]))
    for i in 1:n_phase-1
        A[i+1, i] = 0.99
    end
    A[n_phase, n_phase] = 1.0
    return A
end

function train_arhmm(seqs, n_phase)
    dim_state = length(seqs[1][1])
    dim_phase = dim_state * 2 # state + state's velocity

    xs_list = Sequence{dim_phase}[]
    for seq in seqs
        xs = Sequence([vcat(seq[t], seq[t+1]) for t in 1:length(seq)-1])
        push!(xs_list, xs)
    end

    prop = LinearPropagator(Diagonal(ones(dim_phase)), Diagonal(ones(dim_phase) * 3.0), zeros(dim_phase))
    prop_list = [deepcopy(prop) for _ in 1:n_phase]

    A = create_markov_matrix(n_phase)
    mp = ModelParameters(dim_phase, A, prop_list)

    hs_list = [HiddenStates(xs.n_seq, n_phase) for xs in xs_list]
    for i in 1:20
        loglikeli = 0.0
        for j in 1:length(hs_list)
            hs = hs_list[j]
            xs = xs_list[j]
            loglikeli += update_hidden_states!(hs, mp, xs)
        end
        update_model_parameters!(hs_list, mp, xs_list)
        println("epoch: ", i, ", loglikeli: ", loglikeli) 
    end
    return hs_list
end

function extract_labels(hs_list, n_phase)
    # converting list of matrix
    mat_list = []
    for hs in hs_list
        n_seq = length(hs.z_ests)
        mat = zeros(n_phase, n_seq)
        for i in 1:n_seq
            mat[:, i] = hs.z_ests[i]
        end
        push!(mat_list, mat')
    end

    phase_seq_list = []
    for hs in hs_list
        n_seq = length(hs.z_ests)
        phase_seq = [argmax(hs.z_ests[i]) for i in 1:n_seq]
        push!(phase_seq_list, phase_seq)
    end
    return phase_seq_list
end

if length(ARGS) == 1
    n_phase = parse(Int, ARGS[1])
else
    n_phaes = 3
end
println("n_phase : ", n_phase)
filename = joinpath(expanduser("~"), ".kyozi/summary_chunk.pickle")
seqs = get_sequences(filename);
hs_list = train_arhmm(seqs, n_phase)
phase_seq_list = extract_labels(hs_list, n_phase)

# dump result
result_filename = joinpath(expanduser("~"), ".kyozi/arhmm_result.pickle")
result_filename_debug_json = joinpath(expanduser("~"), ".kyozi/arhmm_result.json")
dump_pickle(phase_seq_list, result_filename)
open(result_filename_debug_json, "w") do f
    write(f, JSON.json(phase_seq_list, 2))
end
