using ActionModels
using StatsPlots
using Distributions
using Plots
using CSV
using DataFrames
using DataFramesMeta


##### For 1_updated_behav #####
premade_agent("help")

df1 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\1_updated_behav.csv", DataFrame)

@transform!(df1, word_end = ifelse.((df1.Cue .== "Cue 2") .& (df1.Prob_cat .== "Congruent") .| (df1.Cue .== "Cue 1") .& (df1.Prob_cat .== "Incongruent"), 0, 1))

actions = df1[!, :Prediction]
inputs = df1.word_end

actions_bool = map(x -> x == 1 ? true : false, actions)
actions = Vector{Any}(actions_bool)


agent = premade_agent("binary_rescorla_wagner_softmax")

priors = Dict("learning_rate" => Normal(0.5, 0.5))

fitted_model = fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 1000)

plot_parameter_distribution(fitted_model, priors)

##### For 2_updated_behav #####

#df2 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\2_updated_behav.csv", DataFrame)

@transform!(df2, word_end = ifelse.((df2.Cue .== "Cue 2") .& (df2.Prob_cat .== "Congruent") .| (df2.Cue .== "Cue 1") .& (df2.Prob_cat .== "Incongruent"), 0, 1))

#df2 = df2[2:end, :]

actions = df2[!, :Prediction]

inputs = df2.word_end

actions_bool = map(x -> x == 1 ? true : false, actions)
actions = Vector{Any}(actions_bool)

agent = premade_agent("binary_rescorla_wagner_softmax")

priors = Dict("learning_rate" => Normal(0.5, 0.5))

fitted_model = fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 1000)

plot_parameter_distribution(fitted_model, priors)