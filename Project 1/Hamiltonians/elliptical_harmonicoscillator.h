#pragma once
#include "hamiltonian.h"
#include <vector>

class EllipticalHarmonicOscillator : public Hamiltonian {
public:
  EllipticalHarmonicOscillator(System * system, double gamma, double bosonDiameter);
  double computeLocalEnergy(std::vector<class Particle*> particles, bool DDeriv_input);
private:
  double computeDistance(std::vector<double> v1, std::vector<double> v2);
  double m_gamma = 0;
  double m_bosonDiameter = 0.00433;
  double m_numberOfDimensions = 0;
};
