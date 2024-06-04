using Pkg
using DataFrames
using DataFramesMeta
using CSV
using Distributions
using ActionModels
using HierarchicalGaussianFiltering
using Plots
using StatsPlots

############### For 1_updated_behav ###############
df1 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\1_updated_behav.csv", DataFrame)
df1
@transform!(df1, word_end = ifelse.((df1.Cue .== "Cue 2") .& (df1.Prob_cat .== "Congruent") .| (df1.Cue .== "Cue 1") .& (df1.Prob_cat .== "Incongruent"), 0, 1))

@transform!(df1, Correct = ifelse.(((df1.Cue .== "Cue 2") .& (df1.condition .== "congruent") .& (df1.Prediction .== 0)) .| ((df1.Cue .== "Cue 2") .& (df1.condition .== "incongruent") .& (df1.Prediction .== 1)) .| ((df1.Cue .== "Cue 1") .& (df1.condition .== "congruent") .& (df1.Prediction .== 1)) .| ((df1.Cue .== "Cue 1") .& (df1.condition .== "incongruent") .& (df1.Prediction .== 0)), 1, 0))
df1

#CSV.write(raw"CogNeuroscienceExam\data_behavioral\1_corrected_behav.csv", df1)

filtered_df1 = df1[(df1.Cue .== "Cue 1") .&& (df1.condition .== "congruent") .&& (df1.Prediction .== 1) .&& (df1.Correct .== 0), :]


actions = df1[!, :Prediction]
inputs = df1.word_end

actions_bool = map(x -> x == 1 ? true : false, actions)
actions = Vector{Any}(actions_bool)

agent = premade_agent("hgf_binary_softmax")
get_parameters(agent)

prior = Dict(("xprob", "volatility") => Normal(-4, 0.5))

fitted_model = fit_model(agent, prior, inputs, actions, n_iterations = 20, chains = 4)

plot_parameter_distribution(fitted_model, prior)

############### For 2_updated_behav ###############
df2 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\2_updated_behav.csv", DataFrame)
df2_c = deepcopy(df2[1:2,:])
df2 = df2[2:end, :]

@transform!(df2, word_end = ifelse.((df2.Cue .== "Cue 2") .& (df2.Prob_cat .== "Congruent") .| (df2.Cue .== "Cue 1") .& (df2.Prob_cat .== "Incongruent"), 0, 1))

@transform!(df2, Correct = ifelse.(((df2.Cue .== "Cue 2") .& (df2.condition .== "congruent") .& (df2.Prediction .== 0)) .| ((df2.Cue .== "Cue 2") .& (df2.condition .== "incongruent") .& (df2.Prediction .== 1)) .| ((df2.Cue .== "Cue 1") .& (df2.condition .== "congruent") .& (df2.Prediction .== 1)) .| ((df2.Cue .== "Cue 1") .& (df2.condition .== "incongruent") .& (df2.Prediction .== 0)), 1, 0))

filtered_df2 = df2[(df2.Cue .== "Cue 1") .&& (df2.condition .== "congruent") .&& (df2.Prediction .== 0) .&& (df2.Correct .== 1), :]



actions = df2[!, :Prediction]
inputs = df2.word_end

actions_bool = map(x -> x == 1 ? true : false, actions)
actions = Vector{Any}(actions_bool)

agent = premade_agent("hgf_binary_softmax")

prior = Dict(("xprob", "volatility") => Normal(-4, 0.5))

fitted_model = fit_model(agent, prior, inputs, actions, n_iterations = 20, chains = 4)

plot_parameter_distribution(fitted_model, prior)