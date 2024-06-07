using HierarchicalGaussianFiltering
using ActionModels
using StatsPlots
using Plots
using Distributions



agent = premade_agent("hgf_binary_softmax")
premade_agent("help")

inputs = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0];

actions = give_inputs!(agent, inputs)

plot_trajectory(agent, ("u", "input_value"))
plot_trajectory!(agent, ("xbin", "prediction"))

get_parameters(agent)

prior = Dict(("xprob", "xvol", "coupling_strength") => Normal(1, 1.0))

model = fit_model(agent, prior, inputs, actions, n_iterations = 20)

psplot = plot_predictive_simulation(
    model,
    agent,
    inputs,
    ("xvol", "precision_prediction_error"),
    n_simulations = 3,
    include_data = false
)

model


get_states(agent)

