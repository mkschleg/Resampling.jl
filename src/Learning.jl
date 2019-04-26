"""
    All learning algorithms assume a single value function!!!
"""

using Flux
import LinearAlgebra: dot
import Statistics: mean

tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))

function offpolicy_tdloss(ρ_t::Array{T, 1}, v_t::TrackedArray, c::Array{T, 1}, γ_tp1::Array{T, 1}, ṽ_tp1::Array{T, 1}) where {T<:AbstractFloat}
    target = T.(c .+ γ_tp1.*ṽ_tp1)
    return (T(0.5))*sum(ρ_t.*((v_t .- target).^2)) * (1//length(ρ_t))
end

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    0.5*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))


"""
    LearningUpdate
WIP - Currently LearningUpdate and Optimizer are haphazardly similar....
"""
abstract type LearningUpdate end


"""
    train!(value, opt::Optimizer, lu::LearningUpdate, ρ, s_t, s_tp1, reward, γ, terminal)
# Arguments:
`value::ValueFunction`:
`opt::Optimizer`:
`ρ`: Importance sampling ratios (Array of Floats)
`s_t`: States at time t
`s_tp1`: States at time t + 1
`reward`: cumulant or reward for value function
`γ`: discount factor
`terminal`: Determining termination of the episode (if applicable).
"""
# train!(value::AbstractVFunction, lu::LearningUpdate, ϕ_t, ϕ_tp1, reward, γ, ρ, terminal)

"""
    train!(value::ValueFunction, opt::Optimizer, lu::LearningUpdate, ρ, s_t, s_tp1, reward, γ, terminal, a_t, a_tp1, target_policy)
# Arguments:
`value::ValueFunction`:
`opt::Optimizer`:
`ρ`: Importance sampling ratios (Array of Floats)
`s_t`: States at time t
`s_tp1`: States at time t + 1
`reward`: cumulant or reward for value function
`γ`: discount factor
`terminal`: Determining termination of the episode (if applicable).
`a_t`: Action at time t
`a_tp1`: Action at time t + 1
`target_policy`: Action at time t
"""

# train!(value::AbstractQFunction, lu::LearningUpdate, ϕ_t, ϕ_tp1, reward, γ, ρ, terminal, a_t, a_tp1, target_policy)

update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0) =
    update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal)

mutable struct BatchTD <: LearningUpdate end

function update!(model, opt, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    preds_t = model.(s_t)
    preds_tp1 = model.(s_tp1)
    grads = Flux.gradient(()->offpolicy_tdloss(preds, r, γ, Flux.data.(preds_tp1)), params(model))
    for weights in params(model)
        update!(opt, weights, -grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    dvdt = deriv.(model, s_t)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -apply!(opt, model.W, corr_term*Δ)
end

function update!(model::TabularLayer, opt::Descent, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = corr_term.*δ.*(1.0/length(v_t))
    for (s_idx, s) in enumerate(s_t)
        model.W[s] -= opt.eta*Δ[s_idx]
    end
end


mutable struct WISBatchTD <: LearningUpdate
    batch_td::BatchTD
    WISBatchTD() = new(BatchTD())
end

function update!(model, opt, lu::WISBatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    wis_avg = mean(ρ)
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=(1.0/wis_avg)*corr_term)
end


mutable struct VTrace <: LearningUpdate
    ρ_bar::Float64
    batch_td::BatchTD
    VTrace(ρ_bar) = new(ρ_bar, BatchTD())
end

function update!(model, opt, lu::VTrace, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    update!(model, opt, lu.batch_td, clamp.(ρ, 0.0, lu.ρ_bar), s_t, s_tp1, r, γ, terminal; corr_term=corr_term)
end

"""
    All Q Learning algorithms are defined with multiple outputs, there is no cont action cases yet.
"""

# Q Learning

TabularQLayer(num_actions, dims...) = TabularLayer(num_actions, dims...)

mutable struct BatchSarsa <: LearningUpdate end

function update!(model, opt, lu::BatchSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    preds_t = [model(s_t[idx])[act] for (idx, act) in enumerate(a_t)]
    preds_tp1 = [model(s_tp1[idx])[act] for (idx, act) in enumerate(a_tp1)]
    grads = Flux.gradient(()->offpolicy_tdloss(preds, r, γ, Flux.data.(preds_tp1)), params(model))
    for weights in params(model)
        update!(opt, weights, -grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::BatchSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    q_t = [model(s_t[idx])[act] for (idx, act) in enumerate(action_t)]
    q_tp1 = [model(s_tp1[idx])[act] for (idx, act) in enumerate(a_tp1)]
    dvdt = [deriv(model, s_t[idx]) for (idx, act) in enumerate(action_t)]
    δ = tderror(v_t, r, γ, v_tp1)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -apply!(opt, model.W, corr_term*Δ)
end

function update!(model::TabularLayer, opt::Descent, lu::BatchSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    q_t = [model(CartesianIndex(act, s_t[idx]...)) for (idx, act) in enumerate(a_t)]
    q_tp1 = [model(CartesianIndex(act, s_tp1[idx]...)) for (idx, act) in enumerate(a_tp1)]
    δ = tderror(q_t, r, γ, q_tp1)
    Δ = corr_term.*δ.*(1.0/length(q_t))
    for (s_idx, s) in enumerate(s_t)
        model[a_t[s_idx], s...] -= opt.eta*Δ[s_idx]
    end
end



mutable struct BatchExpectedSarsa <: LearningUpdate end

function update!(model, opt, lu::BatchExpectedSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    preds_t = [model(s_t[idx])[act] for (idx, act) in enumerate(a_t)]
    exp_q = [dot(target_policy, model(s_tp1[idx])) for (idx, act) in enumerate(a_tp1)]
    grads = Flux.gradient(()->offpolicy_tdloss(preds, r, γ, Flux.data.(exp_q)), params(model))
    for weights in params(model)
        update!(opt, weights, -grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::BatchExpectedSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    q_t = [model(s_t[idx])[act] for (idx, act) in enumerate(action_t)]
    exp_q = [dot(target_policy, model(s_tp1[idx])) for (idx, act) in enumerate(a_tp1)]
    dvdt = [deriv(model, s_t[idx]) for (idx, act) in enumerate(action_t)]
    δ = tderror(v_t, r, γ, exp_q)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -apply!(opt, model.W, corr_term*Δ)
end

function update!(model::TabularLayer, opt::Descent, lu::BatchExpectedSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    q_t = [model(CartesianIndex(act, s_t[idx]...)) for (idx, act) in enumerate(a_t)]
    exp_q = [dot(target_policy, (model[:, s...])) for s in s_tp1]
    δ = tderror(q_t, r, γ, exp_q)
    Δ = corr_term.*δ.*(1.0/length(q_t))
    for (s_idx, s) in enumerate(s_t)
        model[a_t[s_idx], s...] -= opt.eta*Δ[s_idx]
    end
end


