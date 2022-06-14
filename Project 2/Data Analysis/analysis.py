import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from functions import save_fig, save_sub_fig, get_filename

plt.close("all")

def gradientDescent():
    NumberParticles = 2; Dimension = 2; NumberHidden = 2
    Dir = "../Data/"
    # filename = Dir + f"Data_{NumberParticles}P_{Dimension}D_{NumberHidden}H_timestep05.txt"
    wfmethod = "general"; gdmethod = "momentum"; interaction= True
    filename = get_filename(Dir, wf= wfmethod, gd= gdmethod, interaction= interaction,\
                            P=NumberParticles, D=Dimension, H=NumberHidden)
        
    df = pd.read_fwf(filename, header = 0)
    df.drop(labels = df.columns[0], axis= 1, inplace= True)

    NumberMCcycles_list= [2**14, 2**15, 2**16]
    LearningRate_list= [0.001, 0.005, 0.01, 0.05, 0.1]
    len_MC = NumberMCcycles_list.__len__(); len_eta = LearningRate_list.__len__()

    nrows = len_MC
    ncols= 1
    r= nrows - 1
    fig, axes= plt.subplots(nrows, ncols, figsize= (10, 10), sharex= True)

    for i in range(len_MC):
        NumberMCcycles = NumberMCcycles_list[i]
        df_MC = pd.DataFrame(df.where(df.MCcycles == NumberMCcycles))
        df_MC.dropna(inplace = True)
        for j in range(len_eta):
            eta = LearningRate_list[j]
            df_eta = pd.DataFrame(df_MC.where(df_MC.Eta == eta))
            df_eta.dropna(inplace = True)
            # plt.errorbar(df_eta.reset_index().index, df_eta.Energy, df_eta.BlockingVar, fmt= "o-", label= f"MC-cycles= {NumberMCcycles}")
            # plt.plot(df_eta.reset_index().index, df_eta.BlockingVar, ".", label= f"Eta: {eta} converged after {df_eta.shape[0]} iterations")
            label= f"Eta: {eta} converged after {df_eta.shape[0]} iterations"
            axes[r].plot(df_eta.reset_index().index, df_eta.Energy, "o", label= label)
            
        
        axes[r].set_title(f"Number of MC-cycels: 2^{np.log2(NumberMCcycles):g}")
        axes[r].axhline(2.0, label = "Analytical energy", color='k', ls='--', alpha=0.8)
        axes[r].legend(loc= 'best')
        r -= 1

    for ax in axes.flat:
        ax.set(xlabel= r"Number of iterations", ylabel= r"$\langle E\rangle$")
    for ax in axes.flat:
        ax.label_outer()



    if not interaction:
        save_sub_fig(fig, f"gd_{wfmethod}WF_{gdmethod}GD")
    else:
        save_sub_fig(fig, f"gd_{wfmethod}WF_{gdmethod}GD_interaction")
    
# gradientDescent()


NumberParticles = 1; Dimension = 2;
Dir = "../Data/"
# filename = Dir + f"Data_{NumberParticles}P_{Dimension}D_{NumberHidden}H_timestep05.txt"
wfmethod = "general"; gdmethod = "adam"; interaction= True

NumberHidden_list = [1, 2, 4, 8, 16]
# NumberHidden_list = [2]
len_h  = NumberHidden_list.__len__()

std_arr = np.zeros((2, len_h))
mean_arr = std_arr.copy()
for i in range(len_h):
    NumberHidden = NumberHidden_list[i]
    
    filename = get_filename(Dir, wf= wfmethod, gd= gdmethod, interaction= interaction,\
                            P=NumberParticles, D=Dimension, H=NumberHidden)

    df = pd.read_fwf(filename, header = 0)
    df.drop(labels = df.columns[0], axis= 1, inplace= True)
    
    std_arr[0][i] = df.QuantVar.std()
    mean_arr[0][i] = df.QuantVar.mean()
    std_arr[1][i] = df.BlockingVar.std()
    mean_arr[1][i] = df.BlockingVar.mean()
    

# print(std_arr, mean_arr)
# display(std_arr, mean_arr)

nrows = 2
ncols= 1
# r= nrows - 1
fig, axes= plt.subplots(nrows, ncols, figsize= (10, 10), sharex= True)
NumberHidden_list = NumberHidden_list[1:]
std_arr = std_arr[:, 1:]
mean_arr = mean_arr[:, 1:]

# plt.figure()
# plt.xticks(np.arange(17))
# plt.title(f"A quantum system with P: {NumberParticles} and D: {Dimension}")
axes[0].set_title(f"A quantum system with P: {NumberParticles} and D: {Dimension}")
axes[0].errorbar(NumberHidden_list, mean_arr[0], std_arr[0], fmt= "o-", label= "Mean Quantum Variance")
axes[0].legend(loc= 'best')
axes[0].set(xlabel= r"Number of Hidden Nodes", ylabel= r"$\overline{\sigma}^2$")
axes[1].errorbar(NumberHidden_list, mean_arr[1], std_arr[1], fmt= "o-", label= "Mean Blocking Variance")
axes[1].legend(loc= 'best')
axes[1].set(xlabel= r"Number of Hidden Nodes", ylabel= r"$\overline{\sigma}^2_{B}$")
for ax in axes.flat:
    ax.label_outer()
# plt.errorbar(NumberHidden_list, mean_arr[0], std_arr[0], fmt= "o-", label= "Quantum Variance")
# plt.errorbar(NumberHidden_list, mean_arr[1], std_arr[1], fmt= "o-", label= "Blocking")
# plt.plot(NumberHidden_list, std_arr[0], "o-", label= "std")
# plt.plot(NumberHidden_list, mean_arr[0], "o-", label= "mean")
# plt.plot(NumberHidden_list, std_arr[1], "o-", label= "std")
# plt.plot(NumberHidden_list, mean_arr[1], "o-", label= "mean")
# plt.legend()


save_sub_fig(fig, f"h_{wfmethod}WF_{gdmethod}GD_{NumberParticles}P_{Dimension}D")