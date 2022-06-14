#include "elliptical_harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

EllipticalHarmonicOscillator::EllipticalHarmonicOscillator(System* system, double gamma,
   double bosonDiameter) : Hamiltonian(system) {
  m_numberOfDimensions = system->getNumberOfDimensions();
  assert(gamma > 0);
  m_gamma = gamma;
  m_bosonDiameter = bosonDiameter;
}

double EllipticalHarmonicOscillator::computeLocalEnergy(std::vector<class Particle*> particles, bool DDeriv_input) {
  // std::cout << "Class EllipticalHarmonicOscillator Member computeLocalEnergy" << '\n';

  int dim = m_system->getNumberOfDimensions();
  int N   = m_system->getNumberOfParticles();

  double kineticEnergy, potentialEnergy, repulsiveEnergy, localEnergy;
  kineticEnergy = potentialEnergy = repulsiveEnergy = localEnergy = 0;
  std::vector<double> r_i, r_j; double r_ij;

  // Comment out '#pragma' when using the correlated case in either
  // bruteforce_importance or bruteforce_importance_param
  // since these functions are already parallelized.
  // #pragma omp parallel for private(r_i, r_j, r_ij) reduction(- : kineticEnergy) reduction(+ : potentialEnergy, repulsiveEnergy) num_threads(4)
  for (int i = 0; i < N; i++) {
    kineticEnergy -= 0.5*m_system->getWaveFunction()->computeDoubleDerivative(particles, DDeriv_input, i);
    // std::cout << "kineticEnergy: " << kineticEnergy << '\n';
    r_i = particles.at(i)->getPosition();
    potentialEnergy += 0.5*(r_i[0]*r_i[0] + r_i[1]*r_i[1] + m_gamma*m_gamma*r_i[2]*r_i[2]);
    // std::cout << "potentialEnergy: " << potentialEnergy << '\n';
    for (int j = i+1; j < N; j++) {
      r_j  = particles.at(j)->getPosition();
      r_ij = computeDistance(r_i, r_j);
      if (r_ij <= m_bosonDiameter) {
        repulsiveEnergy += 10000; // infinity
        std::cout << "Particle i: " << i << " and j: " << j << " have crashed!" << '\n';
      }
    } // end loop over N (repulsive energy)
  } // end loop over N (number of particles)
  localEnergy = kineticEnergy + potentialEnergy + repulsiveEnergy;
  // std::cout << "localEnergy: " << localEnergy << '\n';
  return localEnergy;
} // end computeLocalEnergy

double EllipticalHarmonicOscillator::computeDistance(std::vector<double> v1, std::vector<double> v2) {
  //std::cout << "Class EllipticalHarmonicOscillator Member computeDistance" << '\n';
  double dist_sqrd, diff;
  dist_sqrd = diff = 0;
  for (int i = 0; i < m_numberOfDimensions; i++) {
    diff = v1[i] - v2[i];
    dist_sqrd += diff*diff;
  }
  return sqrt(dist_sqrd);
} // end computeDistance
