#include "harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;

HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
        Hamiltonian(system) {
    assert(omega > 0);
    m_omega  = omega;
}

double HarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles, bool DDeriv_input) {
     //std::cout << "Class HarmonicOscillator Memeber computeLocalEnergy" << '\n';

     double sum_r_squared    = m_system->getSumOverR_iSquared();
     double potentialEnergy  = 0.5*m_omega*m_omega*sum_r_squared;
     double doubleDerivative = m_system->getWaveFunction()->computeDoubleDerivative(particles, DDeriv_input, 0);
     double kineticEnergy    = -0.5*doubleDerivative;
     return kineticEnergy + potentialEnergy;
}
