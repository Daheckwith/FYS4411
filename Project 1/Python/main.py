import DataFrame as data
import functions as f

# =============================================================================
# Python Libraries
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

case = bool(eval(input("Choose between Simple Gaussian [0] or Correlated [1]: ")))

if not case:
    Dir = "../Data/Simple Gaussian/"
else:
    Dir = "../Data/Correlated/"


if os.path.isdir("../Figures/"):
    pass
else:
    os.mkdir("../Figures/")

Alpha_list = [0.2, 0.35, 0.6, 0.85]
Steps_list = [0.001, 0.01, 0.1, 0.5, 1]
Dims_list = [1, 2, 3]
Par_list = [1, 10, 100]
MC_list = [2**16, 2**18, 2**19, 2**20]

"""
# Parameters Analysis
# Testing how the steplength (BF) and timestep (IMP) affect the results
# For the brute-force (BF) method and the importance (IMP) method respectively
Dir += "Parameters/"
Par = 100; MC_Cycles = 1048576
file_BF             = data.get_filename(Dir, Importance= False, Thr= 1, Dim= 3, Par= Par, MC_Cycles= MC_Cycles)
file_IMP            = data.get_filename(Dir, Importance= True, Thr= 1, Dim= 3, Par= Par, MC_Cycles= MC_Cycles)
df_BF, time_BF      = data.initiate_df(file_BF)
df_IMP, time_IMP    = data.initiate_df(file_IMP)


plt.close("all")
f.parameter_analysis(df_BF, Steps_list)
f.parameter_analysis(df_IMP, Steps_list)
# Parameters Analysis

# Testing how parrallelization effects the runtime
Dir += "Threads/"
f.parallelization(Dir, Par= 100)
"""

"""
# Energy expectation values for both (BF) and (IMP) as a fucntion of alpha
# For a list of dimensions and particles with 2^21 MC-Cycles. 
Dir1 = Dir + "VMC_BF/"
Dir2 = Dir + "VMC_IMP/"

Dims_list = [1,2,3]; Par_list = [1, 10, 100]; MC_Cycles = 2097152
for Dim in Dims_list:
    print(Dim)
    f.subplots_analytical_numerical(Dir1, Par_list, Importance= False, Dim= Dim, MC_Cycles= MC_Cycles)
    f.subplots_analytical_numerical(Dir2, Par_list, Importance= True, Dim= Dim, MC_Cycles= MC_Cycles)
"""

"""
# Parameters Analysis
# Testing how the number of MC-cycles affect the results 
# For the brute-force (BF) method and the importance (IMP) method respectively
# Steplength (BF) and timestep (IMP) both set to 0.1
Dir += "Parameters/MC-Cycles/"
Par = 100; MC_Cycles = 1000 # Just an arbitrary number for this case only
file_BF             = data.get_filename(Dir, Importance= False, Thr= 1, Dim= 3, Par= Par, MC_Cycles= MC_Cycles)
file_IMP            = data.get_filename(Dir, Importance= True, Thr= 1, Dim= 3, Par= Par, MC_Cycles= MC_Cycles)
df_BF, time_BF      = data.initiate_df(file_BF)
df_IMP, time_IMP    = data.initiate_df(file_IMP)



plt.close("all")
f.parameter_analysis_(df_BF, MC_list)
f.parameter_analysis_(df_IMP, MC_list)
# Parameters Analysis
"""

"""
# Parameters Analysis
# Testing how the number of MC-cycles affect the results for the bootstrapping and bloking variance
# For both the brute-force (BF) method and the importance (IMP) method respectively
Dir += "Parameters/MC-Cycles/"
Par = 100; MC_Cycles = 1000 # Just an arbitrary number for this case only

# Analytical case
# file_BF             = data.get_filename(Dir, Importance= False, BB= True, Thr= 8, Dim= 3, Par= Par, MC_Cycles= MC_Cycles)
file_IMP            = data.get_filename(Dir, Importance= True, BB= True, Thr= 8, Dim= 3, Par= Par, MC_Cycles= MC_Cycles)
# df_BF, time_BF      = data.initiate_df(file_BF)
df_IMP, time_IMP    = data.initiate_df(file_IMP)

# data.df_sequential_alpha_sort(df_BF, MC_list)
data.df_sequential_alpha_sort(df_IMP, MC_list)

# f.parameter_analysis_BB(df_BF, MC_list)
f.parameter_analysis_BB(df_IMP, MC_list)

plt.close("all")
"""

"""
# Gradient Descent
Dir += "GD/"
Dim= 3; Thr_list= [1, 3, 5, 7]
Par= 10; # Par = 10, 50, 100
Energy_min = 1e06
Iterations_max = 0


plt.figure()
for Thr in Thr_list:
    file_GD = data.get_filename(Dir, GD= True, Importance= True, Thr= Thr, Dim= Dim, Par= Par, MC_Cycles= 1000)
    df_GD, time_GD = data.initiate_df(file_GD, learningRate= True)
    plt.plot(df_GD.Alpha, label= r"$\alpha_0= $ " + f"{df_GD.Alpha[0]}")
    tmp_min = df_GD.Energy.min()
    tmp_iterations = df_GD.shape[0]
    
    if (tmp_iterations > Iterations_max):
        Iterations_max = tmp_iterations
        
    if (tmp_min < Energy_min):
        Energy_min  = tmp_min
        idx= df_GD.Energy.idxmin()
        print(df_GD.loc[idx])
        alpha_min   = df_GD.Alpha[idx]
        print(alpha_min)

plt.legend()
# plt.yticks(np.arange(0, 2, step=0.2))  # Set label locations.
plt.text(Iterations_max-10, alpha_min+0.05, r"$\alpha= $" + f"{round(alpha_min,3):.4f}", fontsize=12)
plt.xlabel("Number of Iterations"); plt.ylabel(r"$\alpha$")
f.save_fig(f"GD_{Dim}d_{Par}p")
"""


"""
Dir += "VMC_IMP/"
Par_list = [10, 50, 100]


plt.figure()
for Par in Par_list:
    file= data.get_filename(Dir, Importance= True, BB= True, Thr= 7, Dim= 3, Par= Par, MC_Cycles= 2**20)
    df, time = data.initiate_df(file)
    data.df_alpha_sort(df)
    acc_avg = np.mean(df.AcceptanceRatio)
    plt.plot(df.Alpha, df.Energy/Par, 'o-', label= f"N= {Par}, Avg. Acc: {acc_avg:.2f}")
    # print(Par)
    plt.xlabel(r"$\alpha$"); plt.ylabel(r"$\langle E\rangle/N$")
    
plt.legend()
f.save_fig("Energy_IMP")
"""


# One body density

plt.figure()
Dir = "../Data/Simple Gaussian/OB/"
Dim= 3;  MC_Cycles= 2**21; 
Par= 10; # Par = 10, 50, 100
density_G, min_, max_, numberOfBins = f.onebody_density(Dir, Thr= 3, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles)

Dir = "../Data/Correlated/OB/"
density_C1, min_, max_, numberOfBins = f.onebody_density(Dir, Thr=1, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles, correlated= True, interacting= True)
density_C2, min_, max_, numberOfBins = f.onebody_density(Dir, Thr=2, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles, correlated= True)
# plt.xlabel(r"$ z $"); plt.ylabel(r"$\rho(z)$")
plt.xlabel(r"$ x/y $"); plt.ylabel(r"$\rho$")
plt.legend()
# f.save_fig(f"onebody_{Par}_z.png")
f.save_fig(f"onebody_{Par}_xy.png")
