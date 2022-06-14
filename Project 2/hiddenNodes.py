import nqs, gd, hamiltonian as h
from blocking import block
# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np, numpy.random as npr
import pandas as pd

# seed_nr = 30
seed_nr = None
    
# Computing the derivative of the energy and the energy 
def EnergyMinimization(a, b, w, TimeStep):
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5
    sqrt_TimeStep = np.sqrt(TimeStep)
    AcceptanceRatio = 0
    # print(f"Running EnergyMinimization with {NumberMCcycles} MC-cycles")
    # positions
    PositionOld = np.zeros(NumParDim, np.double)
    PositionNew = np.zeros(NumParDim, np.double)
    
    # Quantum force
    # QuantumForceOld = np.zeros((NumberParticles,Dimension), np.double)
    # QuantumForceNew = np.zeros((NumberParticles,Dimension), np.double)

    # seed for rng generator 
    npr.seed(seed_nr)
    energy_arr = np.zeros(NumberMCcycles)
    energy = 0.0; energy_sqrd = 0.0
    DeltaE = 0.0

    # # EnergyDer = np.empty((3,),dtype=object)
    # # DeltaPsi = np.empty((3,),dtype=object)
    # # DerivativePsiE = np.empty((3,),dtype=object)
    # EnergyDer = [np.copy(a),np.copy(b),np.copy(w)]
    # DeltaPsi = [np.copy(a),np.copy(b),np.copy(w)]
    # DerivativePsiE = [np.copy(a),np.copy(b),np.copy(w)]
    # for i in range(3): EnergyDer[i].fill(0.0)
    # for i in range(3): DeltaPsi[i].fill(0.0)
    # for i in range(3): DerivativePsiE[i].fill(0.0)
    
    EnergyDer = [np.zeros(a.shape),np.zeros(b.shape),np.zeros(w.shape)]
    DeltaPsi = [np.zeros(a.shape),np.zeros(b.shape),np.zeros(w.shape)]
    DerivativePsiE = [np.zeros(a.shape),np.zeros(b.shape),np.zeros(w.shape)]
    
    #Initial position
    PositionOld = npr.normal(size=NumParDim)*sqrt_TimeStep
    # for i in range(NumberParticles):
    #     PositionOld[i,:] = npr.normal(size=Dimension)*sqrt_TimeStep
        # for j in range(Dimension):
        #     PositionOld[i,j] = npr.normal()*sqrt_TimeStep
    
    wfold = wf.WaveFunction(PositionOld,a,b,w)
    QuantumForceOld = wf.QuantumForce(PositionOld,a,b,w)

    #Loop over MC MCcycles
    for MCcycle in range(NumberMCcycles):
        #Trial position moving one particle at the time
        for i in range(NumberParticles):
            PositionNew[i,:] = PositionOld[i,:]+npr.normal(size=Dimension)*sqrt_TimeStep+\
                                   QuantumForceOld[i,:]*TimeStep*D
            # for j in range(Dimension):
            #     PositionNew[i,j] = PositionOld[i,j]+npr.normal()*sqrt_TimeStep+\
            #                            QuantumForceOld[i,j]*TimeStep*D
            
            wfnew = wf.WaveFunction(PositionNew,a,b,w)
            QuantumForceNew = wf.QuantumForce(PositionNew,a,b,w)
            
            GreensFunction = 0.0
            temp_arr       = 0.5*(QuantumForceOld[i,:]+QuantumForceNew[i,:])*\
                                  (D*TimeStep*0.5*(QuantumForceOld[i,:]-QuantumForceNew[i,:])-\
                                  PositionNew[i,:]+PositionOld[i,:])
            GreensFunction += temp_arr.sum()
            # for j in range(Dimension):
            #     GreensFunction += 0.5*(QuantumForceOld[i,j]+QuantumForceNew[i,j])*\
            #                           (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-\
            #                           PositionNew[i,j]+PositionOld[i,j])
      
            GreensFunction = np.exp(GreensFunction)
            ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
            
            #Metropolis-Hastings test to see whether we accept the move
            if npr.rand() <= ProbabilityRatio:
                PositionOld[i,:] = PositionNew[i,:]
                QuantumForceOld[i,:] = QuantumForceNew[i,:]
                wfold = wfnew
                AcceptanceRatio += 1
        
        DeltaE = hamil.LocalEnergy(PositionOld,a,b,w)
        DerPsi = wf.DerivativeWFansatz(PositionOld,a,b,w)
        
        DeltaPsi[0] += DerPsi[0]
        DeltaPsi[1] += DerPsi[1]
        DeltaPsi[2] += DerPsi[2]
        
        energy += DeltaE
        energy_sqrd += DeltaE**2
        energy_arr[MCcycle] = DeltaE
        
        DerivativePsiE[0] += DerPsi[0]*DeltaE
        DerivativePsiE[1] += DerPsi[1]*DeltaE
        DerivativePsiE[2] += DerPsi[2]*DeltaE
            
    # We calculate mean values
    energy /= NumberMCcycles
    energy_sqrd /= NumberMCcycles
    
    AcceptanceRatio /= NumberMCcycles*NumberParticles
    
    DerivativePsiE[0] /= NumberMCcycles
    DerivativePsiE[1] /= NumberMCcycles
    DerivativePsiE[2] /= NumberMCcycles
    
    DeltaPsi[0] /= NumberMCcycles
    DeltaPsi[1] /= NumberMCcycles
    DeltaPsi[2] /= NumberMCcycles
    
    EnergyDer[0]  = 2*(DerivativePsiE[0]-DeltaPsi[0]*energy)
    EnergyDer[1]  = 2*(DerivativePsiE[1]-DeltaPsi[1]*energy)
    EnergyDer[2]  = 2*(DerivativePsiE[2]-DeltaPsi[2]*energy)
    return energy, energy_sqrd, energy_arr, EnergyDer, AcceptanceRatio


if __name__ == "__main__":
    # NumberMCcycles_list= [8, 16, 32]
    NumberMCcycles_list= [2**14, 2**15, 2**16]
    LearningRate_list= [0.001, 0.005, 0.01, 0.05, 0.1]
    NumberHidden_list = [1, 2, 4, 8, 16]
    len_MC = NumberMCcycles_list.__len__(); len_eta = LearningRate_list.__len__()
    len_h  = NumberHidden_list.__len__()
    
    eps  = 1e-8
    sigma=1; sig2 = sigma**2
    TimeStep = 0.1
    MaxIterations = 50; NumberMCcycles = 2**16 
    NumberParticles = 1; Dimension = 2;
    NumParDim = (NumberParticles, Dimension)
    for i in range(len_h):
        
        NumberHidden = NumberHidden_list[i]
        print("Configuration")
        print(f"MC: {NumberMCcycles}, P: {NumberParticles}, D: {Dimension}, H: {NumberHidden}")
        Params=(NumberParticles, Dimension, NumberHidden)
        
        columns= ["Energy", "Energy_sqrd", "QuantVar", "BlockingVar", "MCcycles", "TimeStep", "Eta", "AcceptanceRation"]
        len_col = columns.__len__()
        np_df = np.zeros((MaxIterations, len_col))
        
        # For method choose between "general" and "squared"
        wfmethod = "general"; interaction = True # used in LocalEnergy
        wf = nqs.NQS(Params, sigma= sigma, method= wfmethod)
        hamil = h.Hamiltonian(Params, sigma= sigma, method= wfmethod, interaction= interaction)
        
        # guess for parameters
        npr.seed(seed_nr)
        a=npr.normal(loc=0.0, scale=0.001, size=NumParDim)
        b=npr.normal(loc=0.0, scale=0.001, size=NumberHidden)
        w=npr.normal(loc=0.0, scale=0.001, size=Params)
        # a=npr.uniform(0.0, 1, size=NumParDim)
        # b=npr.uniform(0.0, 1, size=NumberHidden)
        # w=npr.uniform(0.0, 1, size=Params)
        
        # For method choose between "simple", "momentum and "adam"
        eta = 0.05; iter = 0; gdmethod= "adam"
        agd_method = gd.GradientDescent(eta= eta, method= gdmethod)
        bgd_method = gd.GradientDescent(eta= eta, method= gdmethod, initiator= 1) # initiator= 1 so it won't print
        wgd_method = gd.GradientDescent(eta= eta, method= gdmethod, initiator= 2) # initiator= 2 so it won't print
        # The reason I chose to initiate 3 gradient descent objects
        # is because each of the variational parameters have different shapes
        # and the parameters within ADAM are dependent on the shape of the parameter.
        
        
        np.seterr(invalid='raise')
        
        while iter < MaxIterations:
            Energy, Energy_sqrd, Energy_arr, EDerivative, AcceptanceRatio = EnergyMinimization(a, b, w, TimeStep)
            quant_var = Energy_sqrd - Energy**2 # Quantum variance
            block_var = block(Energy_arr) 
            
            agradient = EDerivative[0]
            bgradient = EDerivative[1]
            wgradient = EDerivative[2]
            a -= agd_method.learner(a, agradient)
            b -= bgd_method.learner(b, bgradient)
            w -= wgd_method.learner(w, wgradient)
            
            stuff = [Energy, Energy_sqrd, quant_var, block_var, NumberMCcycles, TimeStep, eta, AcceptanceRatio]
            for l in range(len_col): np_df[iter][l] = stuff[l]
            
            if iter > 1:
                diff = np.abs(np_df[iter][0] - np_df[iter - 1][0])
                if diff <= eps:
                    break
            iter += 1
        
        #nice printout with Pandas
        df= pd.DataFrame(np_df, columns= columns)
        if not interaction:
            df.to_string(f"Data/Data_{wfmethod}WF_{gdmethod}GD_{NumberParticles}P_{Dimension}D_{NumberHidden}H.txt")
        else:
            df.to_string(f"Data/Data_{wfmethod}WF_{gdmethod}GD_{NumberParticles}P_{Dimension}D_{NumberHidden}H_Interaction.txt")
        print(df)