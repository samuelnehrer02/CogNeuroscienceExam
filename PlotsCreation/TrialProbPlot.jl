using CSV
using Plots
using DataFrames


df_prob = CSV.read(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\NeuroScience\PlotsAndFigures\TrialProbability.csv", DataFrame)


# Separate the data by environment
stable_df = filter(row -> row.Environment == "Stable", df_prob)
volatile_df = filter(row -> row.Environment == "Volatile", df_prob)

# Plotting
plot(
    stable_df.Trial, stable_df.Prob, 
    label = "Stable", 
    linewidth = 2, 
    color = :midnightblue,
    xlabel = "Trial",
    ylabel = "P(animal stimulus|animal cue)",
    title = "Probability over Cue-Stimulus Contingency",
    yticks = 0:0.1:1.1,
    ylims = (0, 1.0),
)

ProbPlot = plot!(
    volatile_df.Trial, volatile_df.Prob, 
    label = "Volatile", 
    linewidth = 2, 
    color = :orange
)



