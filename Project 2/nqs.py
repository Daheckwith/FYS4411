import numpy as np
from common import Qfac

class NQS:
    def __init__(self, Params, sigma= 1.0, method= "general"):
        # Params=[NumberParticles, Dimension, NumberHidden]
        self.NumberParticles = Params[0]
        self.Dimension = Params[1]
        self.NumberHidden = Params[2]
        self.sigma = sigma; self.sig2 = self.sigma**2
        self.method = method; self.chooseWF()
        
    def chooseWF(self):
        if self.method == "general":
            print("working with the preset for the general solution of the rbm")
            self.WaveFunction = self.gen_WaveFunction
            self.DerivativeWFansatz = self.gen_DerivativeWFansatz
            self.QuantumForce = self.gen_QuantumForce
        elif self.method == "squared":
            print("working with the preset for the squared solution of the rbm")
            self.WaveFunction = self.sqrd_WaveFunction
            self.DerivativeWFansatz = self.sqrd_DerivativeWFansatz
            self.QuantumForce = self.sqrd_QuantumForce
    
    def gen_WaveFunction(self, r,a,b,w):        
        Psi1 = 0.0
        Psi2 = 1.0
        self.Q = Qfac(r,b,w, self.NumberHidden)
        self.exp_Q = np.exp(self.Q)
        self.exp_Q_ = np.exp(-self.Q)
        
        temp_arr = (r-a)**2
        Psi1 = temp_arr.sum()
        # for iq in range(self.NumberParticles):
        #     for ix in range(self.Dimension):
        #         Psi1 += (r[iq,ix]-a[iq,ix])**2
                
        for ih in range(self.NumberHidden):
            Psi2 *= (1.0 + self.exp_Q[ih])
            
        Psi1 = np.exp(-Psi1/(2*self.sig2))
    
        return Psi1*Psi2
    
    def sqrd_WaveFunction(self, r,a,b,w):
        Psi1 = 0.0
        Psi2 = 1.0
        self.Q = Qfac(r,b,w, self.NumberHidden)
        self.exp_Q = np.exp(self.Q)
        self.exp_Q_ = np.exp(-self.Q)
        
        temp_arr = (r-a)**2
        Psi1 = temp_arr.sum()
        # for iq in range(self.NumberParticles):
        #     for ix in range(self.Dimension):
        #         Psi1 += (r[iq,ix]-a[iq,ix])**2
                
        for ih in range(self.NumberHidden):
            Psi2 *= (1.0 + self.exp_Q[ih])
            
        Psi1 = np.exp(-Psi1/(4*self.sig2))
        Psi2 = np.sqrt(Psi2)
    
        return Psi1*Psi2
        
    # Derivate of wave function ansatz as function of variational parameters
    def gen_DerivativeWFansatz(self, r,a,b,w):
        # Has it's own calculation of Qfac because of the if-test
        # for the Metropolis-Hastings test. Since, if the move is not
        # accepted PositionOld is not update and the latest Qfac
        # is always that of PositionNew
        Q = Qfac(r,b,w, self.NumberHidden)
        exp_Q_ = np.exp(-Q)
        
        WfDer = np.empty((3,),dtype=object)
        WfDer = [np.copy(a),np.copy(b),np.copy(w)]
        
        WfDer[0] = (r-a)/self.sig2
        WfDer[1] = 1 / (1 + exp_Q_)
        
        # What Morten had but this can't be right
        # for ih in range(self.NumberHidden):
        #     WfDer[2][:,:,ih] = w[:,:,ih] / (self.sig2*(1+np.exp(-Q[ih])))
        
        # for i in range(self.NumberParticles):
        #     for j in range(self.Dimension):
        #         for ih in range(self.NumberHidden):
        #             WfDer[2][i,j,ih] = r[i][j]/(self.sig2*(1+np.exp(-Q[ih])))
        for ih in range(self.NumberHidden):
            WfDer[2][:,:,ih] = r/(self.sig2*(1+exp_Q_[ih]))
        return  WfDer
    
    # Derivate of wave function ansatz as function of variational parameters
    def sqrd_DerivativeWFansatz(self, r,a,b,w):
        # Has it's own calculation of Qfac because of the if-test
        # for the Metropolis-Hastings test. Since, if the move is not
        # accepted PositionOld is not update and the latest Qfac
        # is always that of PositionNew
        Q = Qfac(r,b,w, self.NumberHidden)
        exp_Q_ = np.exp(-Q)
        
        # WfDer = np.empty((3,),dtype=object)
        # WfDer = [np.copy(a),np.copy(b),np.copy(w)]
        WfDer = [np.zeros(a.shape),np.zeros(b.shape),np.zeros(w.shape)]
        
        
        WfDer[0] = 0.5*(r-a)/self.sig2
        WfDer[1] = 0.5/(1 + exp_Q_)
        
        # What Morten had but this can't be right
        # for ih in range(self.NumberHidden):
        #     WfDer[2][:,:,ih] = w[:,:,ih] / (self.sig2*(1+np.exp(-Q[ih])))
        
        # for i in range(self.NumberParticles):
        #     for j in range(self.Dimension):
        #         for ih in range(self.NumberHidden):
        #             WfDer[2][i,j,ih] = r[i][j]/(self.sig2*(1+np.exp(-Q[ih])))
        for ih in range(self.NumberHidden):
            WfDer[2][:,:,ih] = 0.5*r/(self.sig2*(1+exp_Q_[ih]))
        return  WfDer
    
    # Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
    def gen_QuantumForce(self, r,a,b,w):
        
        qforce = np.zeros((self.NumberParticles,self.Dimension), np.double)
        sum1 = qforce.copy()
        
        # No need to calculate Qfac again since QuantumFoce
        # always comes right after WaveFunction
        # self.Q = Qfac(r,b,w, self.NumberHidden)
        
        for ih in range(self.NumberHidden):
            sum1 += w[:,:,ih]/(1+self.exp_Q_[ih])
                    
            
        qforce = 2*(-(r-a) + sum1)/self.sig2
        
        return qforce
    
    # Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
    def sqrd_QuantumForce(self, r,a,b,w):
        
        qforce = np.zeros((self.NumberParticles,self.Dimension), np.double)
        sum1 = qforce.copy()
        
        # No need to calculate Qfac again since QuantumFoce
        # always comes right after WaveFunction
        # self.Q = Qfac(r,b,w, self.NumberHidden)
        
        for ih in range(self.NumberHidden):
            sum1 += w[:,:,ih]/(1+self.exp_Q_[ih])
        
        qforce = (-(r-a) + sum1)/self.sig2
        
        return qforce