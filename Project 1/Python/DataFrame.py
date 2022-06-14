import os, numpy as np, pandas as pd

def get_filename(Dir, GD= False, Importance= False, BB= False, NumDeriv= False, Thr= 8, Dim= 3, Par= 10, MC_Cycles= 2097152):
    
    if not GD:
        filename = Dir + 'vmc_attributes_'
    else:
        filename = Dir + 'gd_vmc_attributes_'
    
    if not Importance:
        pass
    else:
        filename += "Importance_"
        
    if not BB:
        pass
    else:
        filename += "BB_"
        
    if not NumDeriv:
        filename += "analyticalDeriv_"
    else:
        filename += "numericalDeriv_"
        
    filename += f"{Thr}THREADS_{Dim}d_{Par}p_{MC_Cycles}"
    print(filename)
    return filename

def get_time(filename):
    with open(filename, 'r') as file:
        last_line = file.readlines()[-1]
        # time = last_line[-8:-1]
        time = float(last_line[30:])
    
    return time

def initiate_df(filename, learningRate = False):
    df = pd.read_fwf(filename, header= 0, skipfooter= 1)    
    if not learningRate:
        df.drop(labels = df.columns[0], axis = 1, inplace = True) #Drops an unneeded column
        df.drop(labels = df.columns[1], axis = 1, inplace = True) #Drops an unneeded column
    else:
        df.drop(labels = df.columns[0], axis = 1, inplace = True) #Drops an unneeded column
    
    time = get_time(filename)
    
    return df, time


def df_alpha_sort(df):
    df.sort_values(df.columns[1], axis = 0, inplace = True, ignore_index = True)

"""
def df_step_sort(df):
    df.sort_values(df.columns[-1], axis = 0, inplace = True, ignore_index = True) 
    
def df_turn_numeric(df, s = 0):
    # Depricated mby
    names_list = df.columns
    for i in names_list:
        df[i] = pd.to_numeric(df[i])
    if (s == 1):
        df[names_list[1]] = df[names_list[1]] + 0.01
    s += 1
    
    return names_list, s
"""  

def df_sequential_alpha_sort(df, Parameter_list):
    for i in range(Parameter_list.__len__()):
        df.iloc[i*16:16*(i+1)] = df.iloc[i*16:16*(i+1)].sort_values(df.columns[1], ignore_index = True)
        # df_IMP.iloc[i*16:16*(i+1)] = df_IMP.iloc[i*16:16*(i+1)].sort_values(df_IMP.columns[1], ignore_index = True)
