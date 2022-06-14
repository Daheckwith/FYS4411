import matplotlib.pyplot as plt

def save_fig(name):
    Dir = "../Figures/"
    name = Dir + name
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')
    
def save_sub_fig(fig, name):
    Dir = "../Figures/"
    name = Dir + name
    fig.savefig(name, dpi= 180, bbox_inches= 'tight')

def get_filename(Dir, wf= "general", gd= "momentum", interaction= False, P=2, D=2, H=2):
    filename = Dir + "Data_"
    
    # For method chose between "general" and "squared"
    if wf == "general":
        filename += "generalWF_"
    else:
        filename += "squaredWF_"
    
    # For method chose between "simple", "momentum and "adam"
    if gd == "momentum":
        filename += "momentumGD_"
    elif gd == "adam":
        filename += "adamGD_"
    else:
        filename += "simpleGD_"
    
    # P= NumberParticles D= Dimension H= NumberHidden
    if not interaction:
        filename += f"{P}P_{D}D_{H}H.txt"
        # filename += f"{P}P_{D}D_{H}H_timestep05.txt"
    else:
        filename += f"{P}P_{D}D_{H}H_Interaction.txt"
    
    print(filename)
    return filename