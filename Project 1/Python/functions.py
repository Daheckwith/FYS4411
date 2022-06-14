import DataFrame as data

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def save_fig(name):
    Dir = "../Figures/"
    name = Dir + name
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')
    
def save_sub_fig(fig, name):
    Dir = "../Figures/"
    name = Dir + name
    fig.savefig(name, dpi= 180, bbox_inches= 'tight')
    
def get_param(filename):
    with open(filename, 'r') as file:
        first_line = file.readlines()[0]
        first_line = first_line.split()
        numberOfBins = eval(first_line[1])
        min_ = eval(first_line[3])
        max_ = eval(first_line[5])
    return numberOfBins, min_, max_

"""
def onebody_density(Dir, Thr= 1, Dim= 3, Par= 100, MC_Cycles= 2097152):
    import pandas as pd
    
    
    filename = f"onebody_{Thr}THREADS_{Dim}d_{Par}p_{MC_Cycles}"
    Dir += filename
    print(Dir)
    
    scalingfactor = Par*MC_Cycles
    numberOfBins, min_, max_ = get_param(Dir)
    df = pd.read_fwf(Dir, skiprows= 1, names= ["x", "y", "z"])
    df = df/scalingfactor
    half = int(numberOfBins/2)
    density = np.zeros(half)
    for i in range(half):
        bb = (df.x.iloc[i] + df.x.iloc[-1-i])**2
        bb += (df.y.iloc[i] + df.y.iloc[-1-i])**2
        bb += (df.z.iloc[i] + df.z.iloc[-1-i])**2
        b = np.sqrt(bb)
        density[-i-1] = b
    
    return density, min_, max_, numberOfBins
"""

def onebody_density(Dir, Thr= 1, Dim= 3, Par= 100, MC_Cycles= 2097152, correlated= False, interacting= False):
    import pandas as pd
    if not correlated:
        label = "Ideal"
    else:
        if not interacting:
            label = "non-interacting"
        else:
            label = "interacting"
    
    filename = f"onebody_{Thr}THREADS_{Dim}d_{Par}p_{MC_Cycles}"
    Dir += filename
    print(Dir)
    
    scalingfactor = Par*MC_Cycles
    numberOfBins, min_, max_ = get_param(Dir)
    df = pd.read_fwf(Dir, skiprows= 1, names= ["x", "y", "z"])
    df = df/scalingfactor
    
    density = np.zeros(numberOfBins)
    # plt.plot(np.linspace(min_, max_, numberOfBins), df.z, label= label)
    plt.plot(np.linspace(min_, max_, numberOfBins), df.x, label= label)
        
    
    return density, min_, max_, numberOfBins

def plot_density(density, min_, max_, numberOfBins, correlated= False, interacting= False):
    if not correlated:
        label = "Ideal"
    else:
        if not interacting:
            label = "non-interacting"
        else:
            label = "interacting"
    # plt.xlim(0, max_)
    plt.plot(np.linspace(0, max_, int(numberOfBins/2)), density, label=  label)

def parameter_analysis(df, Steps_list):
    title = df.columns[-1]
    
    alpha = df[df.columns[1]][0:16]
    
    acc_ratio = df[df.columns[7]]
    energy = df[df.columns[2]]
    
    x = alpha
    
    plt.figure()
    for step in Steps_list:
         indices = np.where(df[df.columns[-1]] == step)[0]
         y = energy[indices]
         acc_avg = np.mean(acc_ratio[indices])
         
         plt.plot(x,y, 'o-', label= f"{step}, Avg. Acc: {acc_avg:.2f}")
    
    # plt.title(f"Energy as a function of {df.columns[-1]}")
    plt.xlabel(r"$\alpha$"); plt.ylabel(r'$\left<E\right>$')
    plt.legend()
    
    plt.show()
    
    if title == "StepLength":
        save_fig("Energy_BF.png")
    else:
        save_fig("Energy_IMP.png")

def parameter_analysis_(df, MC_list):
    title = df.columns[-1]
    
    alpha = df[df.columns[1]][0:16]
    
    acc_ratio = df[df.columns[7]]
    energy = df[df.columns[2]]
    
    List = ["$2^{16}$", "$2^{18}$", "$2^{19}$", "$2^{20}$"]
    
    x = alpha
    
    plt.figure()
    for step, st in zip(MC_list, List):
         indices = np.where(df[df.columns[-2]] == step)[0]
         y = energy[indices]
         acc_avg = np.mean(acc_ratio[indices])
         
         plt.plot(x,y, 'o-', label= f"{st}, Avg. Acc: {acc_avg:.2f}")
    
    # if title == "StepLength":
    #     plt.title(f"Energy as a function of MC-cycles Brute-Force {title}: {df[title][0]}")
    # else:
    #     plt.title(f"Energy as a function of MC-cycles Importance {title}: {df[title][0]}")
    
    plt.xlabel(r"$\alpha$"); plt.ylabel(r'$\left<E\right>$')
    plt.legend()
    
    plt.show()
    
    if title == "StepLength":
        save_fig("Energy_BF_MC.png")
    else:
        save_fig("Energy_IMP_MC.png")
      
def parameter_analysis_BB(df, MC_list, NumDeriv= False):
    
    # cmap = plt.get_cmap('Paired')
    cmap = plt.get_cmap('tab10')
    
    title = df.columns[-1]
    
    alpha = df[df.columns[1]][0:16]
    
    variance = df[df.columns[4]]
    bootvar = df[df.columns[5]]
    blockingvar = df[df.columns[6]]
    # acc_ratio = df[df.columns[7]]
    
    
    x = alpha.copy()
    alpha.drop(6, inplace= True) # dropping 0.5 issues with blocking  
    x_drop = alpha.copy()
    
    List = ["$2^{16}$", "$2^{18}$", "$2^{19}$", "$2^{20}$"]
    fig1, ax1= plt.subplots(1, 1, sharex= True)
    fig2, ax2= plt.subplots(1, 1, sharex= True)
    fig3, ax3= plt.subplots(1, 1, sharex= True)
    for step, st in zip(MC_list, List):
        
        # s = -1
        indices = np.where(df[df.columns[-2]] == step)[0]
        var = variance[indices]
        boot = bootvar[indices]
        indices = np.delete(indices, 6) # dropping 0.5 issues with blocking  
        blocking = blockingvar[indices]
        
        # s += 1
        ax1.plot(x, var, "o-", label= f'Variance MC: {st}')
        # s += 1
        ax2.plot(x, boot, "o-", label= f'Bootstrapping MC: {st}')
        # s += 1
        ax3.plot(x_drop, blocking, 'o-', label= f'Blocking MC: {st}')            
        
        # plt.xlabel(r"$\alpha$"); plt.ylabel('Variance/Boot/Blocking')
        ax1.set(xlabel= r"$\alpha$", ylabel= 'Variance')
        ax2.set(xlabel= r"$\alpha$", ylabel= 'Boot Variance')
        ax3.set(xlabel= r"$\alpha$", ylabel= 'Blocking Variance')
        ax1.legend(loc= 'best')
        ax2.legend(loc= 'best')
        ax3.legend(loc= 'best')
        # plt.legend()
    
    
    plt.show()
    
    if not NumDeriv:
        if title == "StepLength":
            save_sub_fig(fig1, "Variance_BF.png")
            save_sub_fig(fig2, "Boot_BF.png")
            save_sub_fig(fig3, "Blocking_BF.png")
        else:
            save_sub_fig(fig1, "Variance_IMP.png")
            save_sub_fig(fig2, "Boot_IMP.png")
            save_sub_fig(fig3, "Blocking_IMP.png")
    else:
        if title == "StepLength":
            save_sub_fig(fig1, "NumVariance_BF.png")
            save_sub_fig(fig2, "NumBoot_BF.png")
            save_sub_fig(fig3, "NumBlocking_BF.png")
        else:
            save_sub_fig(fig1, "NumVariance_IMP.png")
            save_sub_fig(fig2, "NumBoot_IMP.png")
            save_sub_fig(fig3, "NumBlocking_IMP.png")
        
        
def parallelization(Dir, Par = 10):
    Thr_list    = [1, 2, 4, 8, 12]
    time_list   = []
    
    for Thr in Thr_list:
        file = data.get_filename(Dir, Thr= Thr, Par= Par)
        df, time = data.initiate_df(file)
        time_list.append(float(time))
    
    
    #plt.close('all')
    plt.figure()
    # plt.title('Time elapsed per number of threads for 10 Particles and $2^{21}$ MC-cycles')
    plt.plot(Thr_list, time_list, 'o')
    plt.grid()
    plt.xlabel(r'Number of Threads'); plt.ylabel(r'$t\ [s]$')
    plt.legend(['Time elapsed'])
    save_fig("time_per_thread.png")
    plt.show()
    

def analytical_vs_numerical(Dir, Importance= False, BB= False, NumDeriv= False, Thr= 8, Dim= 3, Par= 10, MC_Cycles= 2097152):
    Dir += f"{Dim}D/"
    if not NumDeriv:
        Dir += "Analytical/"
    else:
        Dir += "Numerical/"
        
    df = data.get_filename(Dir, Importance= Importance, BB= BB, NumDeriv= NumDeriv, Dim= Dim, Par = Par, MC_Cycles= MC_Cycles)
    # print(f"filname: {df}")
    df, time = data.initiate_df(df)
    return df, time


def plot_analytical_numerical(Dir, Importance= False, BB= False, NumDeriv= False, Thr= 8, Dim= 3, Par= 10, MC_Cycles= 2097152):
    df_Analytical, time_Analytical = analytical_vs_numerical(Dir, Importance= Importance, BB= BB, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles)
    df_Numerical, time_Numerical = analytical_vs_numerical(Dir, Importance= Importance, BB=  BB, NumDeriv= True, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles)
    columns = df_Analytical.columns

    df_Analytical[columns[4]] = np.sqrt(df_Analytical[columns[4]])  #Calculates the deviation by
    df_Numerical[columns[4]] = np.sqrt(df_Numerical[columns[4]])    #taking the sqrt of the variance
    df_Numerical[columns[1]] = df_Numerical[columns[1]] + 0.01      #Offset the x-axis by 0.01 for the numerical solution
    
    df_Analytical[columns[2]] = df_Analytical[columns[2]]/Par
    df_Numerical[columns[2]] = df_Numerical[columns[2]]/Par

    ax = df_Analytical.plot( x= columns[1], y= columns[2], kind= 'scatter',\
        c= "r", yerr = columns[4])
    df_Numerical.plot( x= columns[1], y= columns[2], kind=  'scatter',\
                  yerr = columns[4], xlabel = r'$ \alpha $', ylabel = r'$\left<E\right>$',\
                  ax= ax, title = f'Average energies and their corresponding deviation for {Dim}D Particles: {Par}', grid = True)
    plt.legend(["Analytical", "Numerical"])

def subplots_analytical_numerical(Dir, Par_list, Importance= False, BB= False, NumDeriv= False, Thr= 8, Dim= 3, MC_Cycles= 2097152):
# def subplots_analytical_numerical(Dir, Dim, Par_list):    
    nrows = Par_list.__len__()
    ncols= 1 
    r= nrows - 1
    time_table = np.zeros((len(Par_list), 2))
    # plt.close('all')
    fig, axes= plt.subplots(nrows, ncols, sharex= True)

    for Par in Par_list:
        df_Analytical, time_Analytical = analytical_vs_numerical(Dir, Importance= Importance, BB= BB, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles)
        df_Numerical, time_Numerical = analytical_vs_numerical(Dir, Importance= Importance, BB= BB, NumDeriv= True, Dim= Dim, Par= Par, MC_Cycles= MC_Cycles)
        # df_Analytical, time_Analytical = analytical_vs_numerical(Dir, Dim= Dim, Par= Par)
        # df_Numerical, time_Numerical = analytical_vs_numerical(Dir, NumDeriv= True, Dim= Dim, Par= Par)
        columns = df_Analytical.columns
        
        idx1 = df_Analytical.index[df_Analytical['Alpha'] == 0.5][0]
        idx2 = df_Numerical.index[df_Numerical['Alpha'] == 0.5][0]
        print(f"Alpha: {df_Analytical['Alpha'].loc[idx1]}, Energy: {df_Analytical['Energy'].loc[idx1]}, Var: {df_Analytical['Variance'].loc[idx1]}")
        print(f"Alpha: {df_Numerical['Alpha'].loc[idx2]}, Energy: {df_Numerical['Energy'].loc[idx2]}, Var: {df_Numerical['Variance'].loc[idx2]}")
        df_Numerical[columns[1]] = df_Numerical[columns[1]] + 0.01 #Offset the x-axis by 0.01 for the numerical solution   
        
        time_table[Par_list.index(Par), 0] = time_Analytical
        time_table[Par_list.index(Par), 1] = time_Numerical
        
        axes[r].errorbar(
            df_Analytical[columns[1]],
            df_Analytical[columns[2]]/Par,
            df_Analytical[columns[4]]/Par,
            label= "Analytical",
            fmt= ".",
            capsize= 3
            )
        axes[r].errorbar(
            df_Numerical[columns[1]],
            df_Numerical[columns[2]]/Par,
            df_Numerical[columns[4]]/Par,
            label= "Numerical",
            fmt= ".",
            capsize= 3
            )
        
        axes[r].grid()
        axes[r].set_title(f"{Dim}D Particles: {Par}")
        r -= 1
        print(f"Time An: {time_Analytical}, Time Num: {time_Numerical}")
        
    axes[0].legend(loc= 'best')
    # print('-----------------------')
    
    for ax in axes.flat:
        ax.set(xlabel= r"$\alpha$", ylabel= r"$\langle E\rangle/N$")
    for ax in axes.flat:
        ax.label_outer()
    
    fig.set_size_inches((6, 7), forward=False)
    if "VMC_BF" in Dir:
        save_fig(f"subplot_{Dim}D_BF.png")
    else:
        save_fig(f"subplot_{Dim}D_IMP.png")
    plt.close("all")
    print(time_table)