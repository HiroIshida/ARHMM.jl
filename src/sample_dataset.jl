using LinearAlgebra

mutable struct State
    x
    v
end

abstract type Attractor end

struct SimpleAttractor <: Attractor
    center
    k
    c
    eps
    r_goal_cond
end
Attractor(center) = SimpleAttractor(center, 0.4, 0.4, 0.02, 0.3)

function (attr::SimpleAttractor)(state, dt)
    direction = state.x .- attr.center
    r = norm(direction)
    gravitational_factor = (r > 0.1 ? (attr.k / r^2) : 0.0)
    force = - gravitational_factor * direction - attr.c * state.v + randn(2) * attr.eps
    state_new = State(state.x .+ state.v * dt, state.v .+ force * dt)
    return state_new
end

function is_goal(attr::SimpleAttractor, state)
    norm(state.v) > 0.4 && (return false)
    return norm(state.x - attr.center) < attr.r_goal_cond
end

struct SequentialAttractor
    attractors
end

function (attr::SequentialAttractor)(state, phase, dt)
    sub_attr = attr.attractors[phase]
    state = sub_attr(state, dt) 

    phase_new = phase
    is_last_phase = (length(attr.attractors) == phase)

    if is_goal(sub_attr, state)
        if is_last_phase
            return state, nothing
        end
        phase_new += 1
    end
    return state, phase_new
end

function create_dataset(N)
    states_list = []
    phases_list = []
    for i in 1:N
        attr1 = Attractor([0, 1.])
        attr2 = Attractor([1.5, 1.])
        seqattr = SequentialAttractor([attr1, attr2])

        s = State([0.0, 0.0] + randn(2) * 0.05, [0.2, 0.0] + randn(2) * 0.01)
        phase = 1
        states = [s]
        phases = [phase]
        for t in 1:500
            dt = 0.05
            s, phase = seqattr(s, phase, dt)
            isnothing(phase) && break
            push!(states, s)
        end
        push!(states_list, states)
        push!(phases_list, phases)
    end
    return states_list, phases_list
end
