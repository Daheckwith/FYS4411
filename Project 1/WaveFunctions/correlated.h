#pragma once
#include "wavefunction.h"


class Correlated : public WaveFunction {
public:
  Correlated(class System* system, double alpha, double beta, double bosonDiameter);
  double evaluate(std::vector<class Particle*> particles);
  double computeDoubleDerivative(std::vector<class Particle*> particles, bool DDeriv_input, int k);
  std::vector<double> computeQuantumForce(Particle* particle, int k);
  double evaluateWFDeriv(std::vector<class Particle*> particles);
  double energyGradient();
private:
  double oneBodyPart(std::vector<class Particle*> particles);
  // double computeDistance(Particle* p1, Particle* p2);
  double computeDistance(std::vector<double> pos1, std::vector<double> pos2);
  double correlatedPart(Particle* p1, Particle* p2);
  double dotProduct(std::vector<double> v1, std::vector<double> v2);
};
