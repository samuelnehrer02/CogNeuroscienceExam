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
#Pkg.develop(path = raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\HGFTest")

############### For 1_updated_behav ###############
df1 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\1_corrected_behav.csv", DataFrame)

@transform!(df1, word_end = ifelse.((df1.Cue .== "Cue 2") .& (df1.condition .== "congruent") .| (df1.Cue .== "Cue 1") .& (df1.condition .== "incongruent"), 0, 1))

# Refinition of actions, depending on believing being in congruent or incongruent condition
@transform!(df1, ActCond = ifelse.((df1.Cue .== "Cue 2" .&& df1.Prediction .== 0) .| ((df1.Cue .== "Cue 1") .&& (df1.Prediction .== 1)), 1, 0))

actions1 = df1[!, :ActCond]
inputs1 = ifelse.(df1.condition .== "congruent", 1, 0)

#actions1 = df1[!, :Prediction]
#inputs1 = df1[!, :word_end]

actions_bool1 = map(x -> x == 1 ? true : false, actions1)
actions1 = Vector{Any}(actions_bool1)

hgf_parameters = Dict(
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("x2", "evolution_rate") => -2.5,
    ("x2", "initial_mean") => 0,
    ("x2", "initial_precision") => 1,
    ("x3", "evolution_rate") => -6.0,
    ("x3", "initial_mean") => 1,
    ("x3", "initial_precision") => 1,
    ("x1", "x2", "value_coupling") => 1.0,
    ("x2", "x3", "volatility_coupling") => 1.0,
);

hgf = premade_hgf("binary_3level", hgf_parameters, verbose = false);
agent_parameters = Dict("sigmoid_action_precision" => 5);

agent1 = premade_agent("hgf_unit_square_sigmoid", hgf, agent_parameters, verbose = false);

get_parameters(agent1)

prior1 = Dict(("xvol", "volatility") => Normal(-2, 1.0))

fitted_model1 = fit_model(agent1, prior1, inputs1, actions1, n_iterations = 20, chains = 4)

plot_parameter_distribution(fitted_model1, prior1)

get_states(agent1)

psplot, pwPE_1_xprob = plot_predictive_simulation(
    fitted_model1,
    agent1,
    inputs1,
    ("xprob", "precision_prediction_error"),
    n_simulations = 10,
    include_data = true
)


pwPE_1_xvol
pwPE_1_xprob






psplot

plot(sim_data[:,10], ylim = (-2, 2))


psplot

plot(psplot_xprob)
plot!(psplot_xvol)

sim_data

psplot
plot(simulation_data[:,4])
sim_data

############### For 2_updated_behav ###############
df2 = CSV.read(raw"CogNeuroscienceExam\data_behavioral\2_corrected_behav.csv", DataFrame)

df2 = df2[2:end, :]

# @transform!(df2, word_end = ifelse.((df2.Cue .== "Cue 2") .& (df2.Prob_cat .== "Congruent") .| (df2.Cue .== "Cue 1") .& (df2.Prob_cat .== "Incongruent"), 0, 1))
@transform!(df2, ActCond = ifelse.((df2.Cue .== "Cue 2" .&& df2.Prediction .== 0) .| ((df2.Cue .== "Cue 1") .&& (df2.Prediction .== 1)), 1, 0))

actions2 = df2[!, :ActCond]
inputs2 = ifelse.(df2.condition .== "congruent", 1, 0)

actions_bool2 = map(x -> x == 1 ? true : false, actions2)
actions2 = Vector{Any}(actions_bool2)

agent2 = premade_agent("hgf_unit_square_sigmoid")

prior2 = Dict(("xvol", "volatility") => Normal(-2, 1.0))

fitted_model2 = fit_model(agent2, prior2, inputs2, actions2, n_iterations = 20, chains = 4)

plot_parameter_distribution(fitted_model2, prior2)

plot(fitted_model2)


psplot = plot_predictive_simulation(
    fitted_model2,
    agent2,
    inputs2,
    ("xvol", "precision_prediction_error"),
    n_simulations = 1,
)

psplot, pwPE_2_xprob = plot_predictive_simulation(
    fitted_model2,
    agent2,
    inputs2,
    ("xprob", "precision_prediction_error"),
    n_simulations = 10,
    include_data = true
)


# Saving the PE data
plot(pwPE_1_xprob[:,12])

pwPE_1_xvol
pwPE_1_xvol[!, "average"] = [mean(row) for row in eachrow(pwPE_1_xvol[:, 2:11])]

pwPE_1_xprob
pwPE_1_xprob[!, "average"] = [mean(row) for row in eachrow(pwPE_1_xprob[:, 2:11])]


pwPE_2_xvol
pwPE_2_xvol[!, "average"] = [mean(row) for row in eachrow(pwPE_2_xvol[:, 2:11])]

pwPE_2_xprob
pwPE_2_xprob[!, "average"] = [mean(row) for row in eachrow(pwPE_2_xprob[:, 2:11])]

#CSV.write(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\CogNeuroscienceExam\PEData\pwPE_2_xvol.csv", pwPE_2_xvol)




psplot

get_states(agent2)
