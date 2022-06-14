#pragma once
#include <vector>
#include <iostream>


class WaveFunction {
public:
    WaveFunction(class System* system);
    int getNumberOfParameters() { return m_numberOfParameters; }
    std::vector<double> getParameters() { return m_parameters; }
    virtual double evaluate(std::vector<class Particle*> particles) = 0;
    virtual double computeDoubleDerivative(std::vector<class Particle*> particles, bool DDeriv_input, int k) = 0;
    virtual std::vector<double> computeQuantumForce(Particle* particle, int k) = 0;
    virtual double evaluateWFDeriv(std::vector<class Particle*> particles) = 0;
    virtual double energyGradient() = 0;

protected:
    int     m_numberOfParameters = 0;
    int     m_numberOfDimensions = 0;
    std::vector<double> m_parameters = std::vector<double>();
    class System* m_system = nullptr;
    double m_alpha = 0;
    double m_beta = 0;
    double m_h = 1e-6;
    double m_bosonDiameter = 0;
};
