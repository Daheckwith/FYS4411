#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include "wavefunction.h"
#include "simplegaussian.h"
#include "../GD_sampler.h"
#include "../system.h"
#include "../particle.h"

SimpleGaussian::SimpleGaussian(System* system, double alpha) :
        WaveFunction(system) {
    assert(alpha >= 0);
    m_alpha = alpha;
}

double SimpleGaussian::evaluate(std::vector<class Particle*> particles) {
     //std::cout << "Class SimpleGaussian Member evaluate" << '\n';

     double waveFunction = exp( -m_alpha*m_system->getSumOverR_iSquared() );
     return waveFunction;
}

double SimpleGaussian::computeDoubleDerivative(std::vector<class Particle*> particles, bool DDeriv_input, int k) {
    int dim = m_system->getNumberOfDimensions();
    int N   = m_system->getNumberOfParticles();

    if (DDeriv_input == 1) {
        // Analytical double derivative
        double sum_r_squared = m_system->getSumOverR_iSquared();
        return 4*m_alpha*m_alpha*sum_r_squared - 2*N*m_alpha*dim;
    } else {
        // The cumulative WaveFunction for all particles
        double waveFunction = evaluate(particles);

        // The inital part of the double derivative
        double initialPart = -2*dim*N*waveFunction;
        double doubleDerivative = initialPart;

        // Loop over particles
        for (auto particle : particles) {
            for (int i = 0; i < dim; i++) {
                particle->adjustPosition(m_h, i); // +h
                doubleDerivative += evaluate(particles);
                particle->adjustPosition(-2*m_h, i); // -h
                doubleDerivative += evaluate(particles);
                particle->adjustPosition(m_h, i); // Reset
            }
        }

        /*
        // Loop over dimensions
        for (int i = 0; i < dim; i++) {
            for (auto particle : particles) {
                particle->adjustPosition(m_h, i); // +h
                doubleDerivative += evaluate(particles);
                particle->adjustPosition(-2*m_h, i); // -h
                doubleDerivative += evaluate(particles);
                particle->adjustPosition(m_h, i); // Reset
            }
        } // end loop over dimensions
        */
        return doubleDerivative / (m_h*m_h*waveFunction);
    }
} // end computeDoubleDerivative

std::vector<double> SimpleGaussian::computeQuantumForce(Particle* particle, int k) {
  //std::cout << "Class SimpleGaussian Member computeQuantumForce" << '\n';
  std::vector<double> position = particle->getPosition();
  std::vector<double> qForce;
  for ( auto comp : position ) {
    qForce.push_back(-4*m_alpha*comp);
  }
  return qForce;
}

double SimpleGaussian::evaluateWFDeriv(std::vector<class Particle*> particles) {
  // returns \frac{\bar{\Psi}}{\Psi} for the evaluation of Gradient Descent
  return -m_system->getSumOverR_iSquared();
}

double SimpleGaussian::energyGradient() {
  double term_1 = m_system->getSampler()->getexpectWFDerivXEnergy();
  double term_2 = m_system->getSampler()->getexpectWFDerivXexpectEnergy();
  return 2*(term_1 - term_2);
}
