using Pkg
using DataFrames
using DataFramesMeta
using CSV
using Distributions
using ActionModels
using HierarchicalGaussianFiltering
using Plots
using StatsPlots

#Pkg.develop(path = raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\ActionModelsTest")


############### For 1_updated_behav ###############
df1 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\1_corrected_behav.csv", DataFrame)

# @transform!(df1, word_end = ifelse.((df1.Cue .== "Cue 2") .& (df1.condition .== "congruent") .| (df1.Cue .== "Cue 1") .& (df1.condittion .== "incongruent"), 0, 1))

# Refinition of actions, depending on believing being in congruent or incongruent condition
@transform!(df1, ActCond = ifelse.((df1.Cue .== "Cue 2" .&& df1.Prediction .== 0) .| ((df1.Cue .== "Cue 1") .&& (df1.Prediction .== 1)), 1, 0))

actions1 = df1[!, :ActCond]
inputs1 = ifelse.(df1.condition .== "congruent", 1, 0)

actions_bool1 = map(x -> x == 1 ? true : false, actions1)
actions1 = Vector{Any}(actions_bool1)

agent1 = premade_agent("hgf_unit_square_sigmoid")
get_parameters(agent1)

prior1 = Dict(("xprob", "volatility") => Normal(-4, 0.5))

fitted_model1 = fit_model(agent1, prior1, inputs1, actions1, n_iterations = 20, chains = 4)

plot_parameter_distribution(fitted_model1, prior1)

get_states(agent1)

psplot = plot_predictive_simulation(
    fitted_model1,
    agent1,
    inputs1,
    ("xvol", "prediction_mean"),
    n_simulations = 3,
    include_data = false
)


    df_prob = CSV.read(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\PlotsAndFigures\TrialProbability.csv", DataFrame)


plot!(
    df_prob.Trial, df_prob.Prob, 
    label = "Probabilities", 
    linewidth = 2, 
    color = :midnightblue,
    xlabel = "Trial",
    ylabel = "P(animal stimulus|animal cue)",
    title = "Probability over Cue-Stimulus Contingency",
    yticks = 0:0.1:1.1,
    ylims = (0.0, 2.0),
    legend = :bottomleft
)


df_prob = CSV.read(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\PlotsAndFigures\TrialProbability.csv", DataFrame)
