using Pkg
using DataFrames
using DataFramesMeta
using CSV
using Distributions
using ActionModels
using HierarchicalGaussianFiltering
using Plots
using StatsPlots

df1 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\1_updated_behav.csv", DataFrame)

df1[20:40,:]

@transform!(df1, word_end = ifelse.((df1.Cue .== "Cue 2") .& (df1.condition .== "congruent") .| (df1.Cue .== "Cue 1") .& (df1.condition .== "incongruent"), 0, 1))
@transform!(df1, CatPred = ifelse.(((df1.Cue .== "Cue 1") .& (df1.Prediction .== 1)) .| ((df1.Cue .== "Cue 2") .& (df1.Prediction .== 0)), 1, 0))

df1[4:5,:]

actions = df1[!, :Prediction]
inputs = df1.word_end

actions_bool = map(x -> x == 1 ? true : false, actions)
actions = Vector{Any}(actions_bool)


hgf = premade_hgf("binary_3level");
hgf


agent_parameters = Dict("sigmoid_action_precision" => 5);
agent = premade_agent("hgf_unit_square_sigmoid", hgf, agent_parameters, verbose = false);

actions = give_inputs!(agent, inputs);

get_states(agent)

plot_trajectory(agent, ("u", "input_value"))
plot_trajectory!(agent, ("xbin", "prediction_mean"))

param_priors = Dict(("x2", "evolution_rate") => Normal(-3.0, 0.5));

fitted_model = fit_model(
    agent,
    param_priors,
    inputs,
    actions,
    fixed_parameters = fixed_parameters,
    verbose = true,
    n_iterations = 10,
)
