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

prior1 = Dict(("xvol", "volatility") => Normal(-2, 1.0), ("xprob", "volatility") => Normal(-2, 1.0))

fitted_model1 = fit_model(agent1, prior1, inputs1, actions1, n_iterations = 200, chains = 4)

plot_parameter_distribution(fitted_model1, prior1)



fitted_model1

psplot15 = plot_predictive_simulation(
    fitted_model1,
    agent1,
    inputs1,
    ("xvol", "precision_prediction_error"),
    n_simulations = 4,
    include_data = false,
    color = :teal
)


get_states(agent1)

psplot = plot_predictive_simulation(
    fitted_model1,
    agent1,
    inputs1,
    ("xvol", "prediction_mean"),
    n_simulations = 4,
    include_data = false
)


psplo

for i in 1:nrow(input)
    if input[i, :simulation_1] == 1.0
        input[i, :simulation_1] = 1.02
    else
        input[i, :simulation_1] = -0.05
    end
end
actions
for i in 1:nrow(actions)
    if actions[i, :simulation_1] == 1.0
        actions[i, :simulation_1] = 1.05
    else
        actions[i, :simulation_1] = -0.02
    end
end
input
theme(:teal)
plot(input[:,2], seriestype = :scatter, legend = (0.12, 0.3), label = ("Input"))
plot!(actions[:,2], seriestype = :scatter, label = ("Prediction"))

level_one_plot = plot!(bin_prob[:,2], seriestype = :line, ylim = (-0.1, 1.1), label = ("Probability"), size = (1200, 400), title = "input, action, s(μ_2)")
level_two_plot = plot(psplot4, size = (1200, 400), legend = false, title = "μ_2", color = :teal)
level_three_plot = plot(psplot2, size = (1200, 400), ylim = (-4, 4), legend = false, title = "μ_3", color = :blue)

combined_HGF_plot = plot(level_three_plot, level_two_plot, level_one_plot, layout = @layout([c; b; a]), size = (1200, 800))
savefig(PE_plot, raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\PlotsAndFigures\PE_A.png")

level_two_plot



PE_2 = plot(psplot10, size = (1200, 400), ylim = (-0.5, 6), legend = false, title = "pwPE2")
PE_3 = plot(psplot11, size = (1200, 400), ylim = (-0.5, 1.5), legend = false, title = "pwPE3")

PE_plot = plot(PE_2, PE_3, layout = @layout([a; b]), size = (1200, 800), xlabel = "Trial", ylabel = "Precision Prediction Error")






plot(psplot_action, size = (1000, 400))
plot!(psplot_input, size = (1000, 400))

combined_plot = plot(psplot_action, psplot_input, layout = @layout([a; b]), size = (1000, 800))

psplot = plot_predictive_simulation(
    agent1,
    inputs1,
    ("xbin", "prediction_mean"),
    n_simulations = 10
)





level_one_plot
pwPE_1_xvol
pwPE_1_xprob


function sigmoid(x)
    s_x = []

    for i in 1:length(x)
        push!(s_x, 1 / (1 + exp(-x[i])))
    end
    return s_x
end



s_x = sigmoid(xprob[:,2])

plot(s_x)
plot(bin_prob[:,2])



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




############################

agent3 = premade_agent("hgf_unit_square_sigmoid")

prior3 = Dict(("xvol", "volatility") => Normal(-2, 1.0))

inputs3 = inputs1[1:50,:]
actions3 = actions1[1:50,:]


fitted_model3 = fit_model(agent3, prior3, inputs3, actions3, n_iterations = 20, chains = 4)

plot_parameter_distribution(fitted_model3, prior3)




sagent = premade_agent("hgf_unit_square_sigmoid")

iinp = Vector(inputs1[1:50])
inputs1

iinp

give_inputs!(sagent, inputs1[1:51])



get_surprise(sagent)

surprisal = []

for i in 1:length(inputs2)
    give_inputs!(sagent, inputs2[1:i])
    ss = get_surprise(sagent)
    push!(surprisal, ss)
end

surprisal[45:55]

for i in 1:length(inputs1)
    println(i)
end

surpurisal_csv = DataFrame(surprisal = surprisal)
CSV.write(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\CogNeuroscienceExam\PEData\surprisal_B.csv", surpurisal_csv)
