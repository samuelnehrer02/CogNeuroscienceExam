using HierarchicalGaussianFiltering
using ActionModels
using StatsPlots
using Plots
using Distributions

agent = premade_agent("hgf_binary_softmax")

set_parameters!(agent, ("xprob", "initial_precision"), 0.9)
inputs = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0];

actions = give_inputs!(agent, inputs)

plot_trajectory(agent, ("u", "input_value"))
plot_trajectory!(agent, ("x", "prediction"))

prior = Dict(("xprob", "volatility") => Normal(1, 0.5))

model = fit_model(agent, prior, inputs, actions, n_iterations = 20)

