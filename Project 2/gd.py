import numpy as np
class GradientDescent:
    def __init__(self, eta= 0.001, gamma= 0.8, beta1= 0.9, beta2= 0.999, method= "simple", initiator= 0):
        self.eta = eta # Learning rate
        self.gamma = gamma # momentum parameter
        self.beta1 = beta1; self.beta1_t = beta1 # adam
        self.beta2 = beta2; self.beta2_t = beta2 # adam
        self.m_t = 0; self.s_t = 0; self.eps = 1e-8 # adam preset
        
        if method in ["simple", "momentum", "adam"]:
            self.method = method
        else:
            print(f"From initiator {initiator}")
            print("The input given doens't match any of the available methods.")
            print("Choose between \"simple\", \"momentum\" or \"adam\"")
            self.method= str(input("Type in your input: "))
            
        self.learner = self.chooselearner()
        
        if initiator == 0:
            self.printout()
            
    
    def chooselearner(self):
        if self.method == "simple":
            return self.simpleGD
        elif self.method == "momentum":
            return self.momentumGD
        elif self.method == "adam":
            return self.adamGD
            
    def printout(self):        
        if self.method == "simple":
            print("simpleGD has been chosen as learner")
        elif self.method == "momentum":
            print("momentumGD has been chosen as learner")
        elif self.method == "adam":
            print("adamGD has been chosen as learner")
        
        print("The parameters chosen for gradient descent:")
        print(f"Learning rate: {self.eta}, Momentum parameter: {self.gamma}\n Explicitly for ADAM")
        print(f"First hyper parameter: {self.beta1}, Second hyper parameter: {self.beta2}")
    
    def simpleGD(self, var, gradient):
        return self.eta*gradient
    
    def momentumGD(self, var, gradient):
        return self.gamma*var + self.eta*gradient
    
    def adamGD(self, var, gradient):
        self.m_t = self.beta1*self.m_t + (1-self.beta1)*gradient
        self.s_t = self.beta2*self.s_t + (1-self.beta2)*gradient**2
        m_hat = self.m_t/(1 - self.beta1_t)
        s_hat = self.s_t/(1 - self.beta2_t)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        
        return self.eta*m_hat/(np.sqrt(s_hat) + self.eps)
    
    def adamGD_(self, var, gradient):
        self.m_t = self.beta1*self.m_t + (1-self.beta1)*gradient
        self.s_t = self.beta2*self.s_t + (1-self.beta2)*gradient**2
        eta_t = self.eta*np.sqrt(1-self.beta2_t)/(1-self.beta1_t) 
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        return eta_t*self.m_t/(np.sqrt(self.s_t) + self.eps)