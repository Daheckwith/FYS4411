import numpy as np
from common import Qfac

class Hamiltonian:
    def __init__(self, Params, sigma= 1.0, method= "general", interaction= False):
        # Params=[NumberParticles, Dimension, NumberHidden]
        self.NumberParticles = Params[0]
        self.Dimension = Params[1]
        self.NumberHidden = Params[2]
        self.sigma = sigma; self.sig2 = self.sigma**2
        self.method = method; self.chooseHamiltonian()
        self.interaction = interaction
        if interaction == True:
            print("Interaction between particles enabled")
        
    def chooseHamiltonian(self):
        if self.method == "general":
            self.LocalEnergy = self.gen_LocalEnergy
        elif self.method == "squared":
            self.LocalEnergy = self.sqrd_LocalEnergy
    
    # Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
    def gen_LocalEnergy(self,r,a,b,w):
        locenergy = 0.0
        
        Q = Qfac(r,b,w, self.NumberHidden)
        exp_Q = np.exp(Q)
        exp_Q_ = np.exp(-Q)

        # for iq in range(self.NumberParticles):
        #     for ix in range(self.Dimension):
        #         sum1 = 0.0
        #         sum2 = 0.0
        #         for ih in range(self.NumberHidden):
        #             sum1 += w[iq,ix,ih]/(1+exp_Q_[ih])
        #             sum2 += w[iq,ix,ih]**2 * exp_Q[ih] / (1.0 + exp_Q[ih])**2
        
        #         dlnpsi1 = (-(r[iq,ix] - a[iq,ix]) + sum1)/self.sig2
        #         dlnpsi2 = -1/self.sig2 + sum2/self.sig2**2
        #         locenergy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq,ix]**2)
        for iq in range(self.NumberParticles):
            for ix in range(self.Dimension):
                sum1 = (w[iq,ix,:]/(1+exp_Q_)).sum()
                sum2 = (w[iq,ix,:]**2 * exp_Q / (1.0 + exp_Q)**2).sum()
                    
        
                dlnpsi1 = (-(r[iq,ix] - a[iq,ix]) + sum1)/self.sig2
                dlnpsi2 = -1/self.sig2 + sum2/self.sig2**2
                locenergy += 0.5*(-dlnpsi1**2 - dlnpsi2 + r[iq,ix]**2)
                
        if(self.interaction==True):
            for iq1 in range(self.NumberParticles):
                for iq2 in range(iq1):
                    distance = 0.0
                    for ix in range(self.Dimension):
                        distance += (r[iq1,ix] - r[iq2,ix])**2
                        
                    locenergy += 1/np.sqrt(distance)
                    
        return locenergy    
    
    # Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
    def sqrd_LocalEnergy(self,r,a,b,w):
        locenergy = 0.0
        
        Q = Qfac(r,b,w, self.NumberHidden)
        exp_Q = np.exp(Q)
        exp_Q_ = np.exp(-Q)
        
        for iq in range(self.NumberParticles):
            for ix in range(self.Dimension):
                sum1 = (w[iq,ix,:]/(1+exp_Q_)).sum()
                sum2 = (w[iq,ix,:]**2 * exp_Q / (1.0 + exp_Q)**2).sum()
                    
        
                dlnpsi1 = 0.5*(-(r[iq,ix] - a[iq,ix]) + sum1)/self.sig2
                dlnpsi2 = -0.5/self.sig2 + 0.5*sum2/self.sig2**2
                locenergy += 0.5*(-dlnpsi1**2 - dlnpsi2 + r[iq,ix]**2)
                
        if(self.interaction==True):
            for iq1 in range(self.NumberParticles):
                for iq2 in range(iq1):
                    distance = 0.0
                    for ix in range(self.Dimension):
                        distance += (r[iq1,ix] - r[iq2,ix])**2
                        
                    locenergy += 1/np.sqrt(distance)
                    
        return locenergy