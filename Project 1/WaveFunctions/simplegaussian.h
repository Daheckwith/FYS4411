#pragma once
#include "wavefunction.h"

class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(class System* system, double alpha);
    double evaluate(std::vector<class Particle*> particles);
    double computeDoubleDerivative(std::vector<class Particle*> particles, bool DDeriv_input, int k);
    std::vector<double> computeQuantumForce(Particle* particle, int k);
    double evaluateWFDeriv(std::vector<class Particle*> particles);
    double energyGradient();
};
