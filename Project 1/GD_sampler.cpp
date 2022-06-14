#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include "omp.h"
#include "sampler.h"
#include "GD_sampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;
using std::ios;
using std::to_string;


GD_sampler::GD_sampler(System* system) : Sampler(system) {
  m_system = system;
}


void GD_sampler::sample(bool acceptedStep, bool DDeriv_input) {
  //std::cout << "Class GD_sampler Member sample" << '\n';
    // Make sure the sampling variable(s) are initialized at the first step.
    /*
    if (m_stepNumber == 0) {
        m_cumulativeEnergy    = 0;
        m_cumulativeEnergy_sq = 0;
        m_acceptedSteps       = 0;
        m_acceptanceRatio     = 0;
    }
    // No need for this since calling an instance new Sampler resets the values
    */

    if(acceptedStep) {
      m_acceptedSteps++;
    }


    m_localEnergy = m_system->getHamiltonian()->
                         computeLocalEnergy(m_system->getParticles(), DDeriv_input);

    double WFDeriv = m_system->getWaveFunction()->
                      evaluateWFDeriv(m_system->getParticles());

    m_localEnergyVec.push_back(m_localEnergy);
    m_cumulativeEnergy    += m_localEnergy;
    m_cumulativeEnergy_sq += m_localEnergy*m_localEnergy;

    m_cumulativeWFDeriv   += WFDeriv;
    m_cumulativeWFDerivXEnergy += WFDeriv*m_localEnergy;
} // end sample


void GD_sampler::computeAverages() {
    // std::cout << "Class GD_sampler Member computeAverages" << '\n';
    m_energy          = m_cumulativeEnergy / m_numberOfMetropolisSteps;
    m_energy_sqrd     = m_cumulativeEnergy_sq / m_numberOfMetropolisSteps;
    m_variance        = m_energy_sqrd - m_energy*m_energy;
    m_acceptanceRatio = double (m_acceptedSteps) / double (m_numberOfMetropolisSteps);

    m_WFDeriv     = m_cumulativeWFDeriv / m_numberOfMetropolisSteps;
    m_expectWFDerivXEnergy = m_cumulativeWFDerivXEnergy / m_numberOfMetropolisSteps;
    m_expectWFDerivXexpectEnergy = m_WFDeriv*m_energy;
}
