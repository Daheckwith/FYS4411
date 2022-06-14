#include <iostream>
#include <cassert>
#include <cmath>
#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"

// For testing
#include "WaveFunctions/correlated.h"

using std::string;


System::System() {
    m_random = new Random();
}

System::System(int seed) {
    m_random = new Random(seed);
}

bool System::metropolisStep() {
	 //std::cout << "Class System Member metropolisStep" << '\n';

   //Picking a random particle
   int rndIndx = getRandomEngine()->nextInt( m_numberOfParticles - 1 );
   Particle * rndParticle = m_particles.at(rndIndx);

   double change;
   std::vector<double> change_series;
   change_series.reserve(m_numberOfDimensions);
   //Changing the particle's coordinates by a random amount between -0.5 - 0.5
   for ( int dim = 0; dim < m_numberOfDimensions; dim++ ) {
       change = m_stepLength*(getRandomEngine()->nextDouble() - 0.5);
       //std::cout << "m_stepLength: " << m_stepLength << " change: " << change  << '\n';
       change_series.push_back(change);
       rndParticle->adjustPosition(change, dim);
   }

   newWaveFunction = m_waveFunction->evaluate(m_particles);
   double probabilityRatio =
          (newWaveFunction*newWaveFunction)/(oldWaveFunction*oldWaveFunction);

   //The Metroplis-step
   if ( getRandomEngine()->nextDouble() <= probabilityRatio ) {
       oldWaveFunction = newWaveFunction;
       return true;
   } else {
       //If rejected revert back
       for ( int dim = 0; dim < m_numberOfDimensions; dim++ ) {
           rndParticle->adjustPosition(-change_series.at(dim), dim);
       }
       return false;
   }
} // end metropolisStep

bool System::importanceStep(){
  //std::cout << "class System Member importanceStep" << '\n';

  //Picking a random particle
  int rndIndx = getRandomEngine()->nextInt( m_numberOfParticles - 1 );
  Particle * rndParticle = m_particles.at(rndIndx);

  std::vector<double> oldPos = rndParticle->getPosition();
  std::vector<double> oldQforce = m_waveFunction->computeQuantumForce(rndParticle, rndIndx);

  double change;
  //double timeStep = 0.01;
  std::vector<double> change_series;
  change_series.reserve(m_numberOfDimensions);

  //Changing the particle's coordinates based on the Langevin equation
  for ( int dim = 0; dim < m_numberOfDimensions; dim++ ) {
    change = oldQforce.at(dim)*0.5*m_timeStep + getRandomEngine()->nextGaussian(0, 1)*sqrt(m_timeStep);
    change_series.push_back(change);
    rndParticle->adjustPosition(change, dim);
  }

  std::vector<double> newPos = rndParticle->getPosition();
  std::vector<double> newQforce = m_waveFunction->computeQuantumForce(rndParticle, rndIndx);
  newWaveFunction = m_waveFunction->evaluate(m_particles);

  // Computing the probability ratio for the importance sampling algorithm
  // based on the Green's function and the probability density given by
  // the square of the wave functions
  double GreensFunctionRatio, term_1, term_2, probabilityRatio;
  for ( int dim = 0; dim < m_numberOfDimensions; dim++) {
    term_1 = newPos.at(dim) - oldPos.at(dim) - 0.5*m_timeStep*oldQforce.at(dim);
    term_2 = oldPos.at(dim) - newPos.at(dim) - 0.5*m_timeStep*newQforce.at(dim);
    GreensFunctionRatio += (term_1*term_1) - (term_2*term_2);
  }
  GreensFunctionRatio = GreensFunctionRatio/(4*0.5*m_timeStep);
  GreensFunctionRatio = exp(GreensFunctionRatio);
  probabilityRatio    = GreensFunctionRatio*\
  (newWaveFunction*newWaveFunction)/(oldWaveFunction*oldWaveFunction);

  //Metroplis-Hastings if-test
  if ( getRandomEngine()->nextDouble() <= probabilityRatio) {
    oldWaveFunction = newWaveFunction;
    return true;
  } else {
    //If rejected revert back
    for ( int dim = 0; dim < m_numberOfDimensions; dim++ ) {
      rndParticle->adjustPosition(-change_series.at(dim), dim);
    }
    return false;
  }
} // emd importanceStep

void System::runMetropolisSteps(int numberOfMetropolisSteps, bool DDeriv_input, bool Boot_Blocking_input) {
	//std::cout << "Class System Member runMetropolisSteps" << '\n';
    m_particles                 = m_initialState->getParticles();
    m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
    m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);

    oldWaveFunction = m_waveFunction->evaluate(m_particles);
    int equibPart = m_equilibrationFraction*numberOfMetropolisSteps;
    bool acceptedStep;

    //m_localEnergyVec.reserve(m_numberOfMetropolisSteps);

    if (m_importanceSampling_input == 1) {
      //The equilibration part
      for (int i = 0; i < equibPart; i++) {
        importanceStep();
      }

      for (int i = 0; i < numberOfMetropolisSteps; i++) {
        acceptedStep = importanceStep();
        m_sampler->sample(acceptedStep, DDeriv_input);
      }
    } else {
      //The equilibration part
      for (int i = 0; i < equibPart; i++) {
        metropolisStep();
      }
      for (int i = 0; i < numberOfMetropolisSteps; i++) {
        acceptedStep = metropolisStep();
        m_sampler->sample(acceptedStep, DDeriv_input);
      }
    } // if-else
    m_sampler->computeAverages();
    m_sampler->sampleEnergy(m_filename, Boot_Blocking_input);
    //m_sampler->printOutputToTerminal();
} // end runMetropolisSteps

void System::setNumberOfParticles(int numberOfParticles) {
    m_numberOfParticles = numberOfParticles;
}

void System::setNumberOfDimensions(int numberOfDimensions) {
    m_numberOfDimensions = numberOfDimensions;
}

void System::setStepLength(double stepLength) {
    assert(stepLength >= 0);
    m_stepLength = stepLength;
}

void System::setEquilibrationFraction(double equilibrationFraction) {
    assert(equilibrationFraction >= 0);
    m_equilibrationFraction = equilibrationFraction;
}

void System::setVariable(double variable) {
  assert(variable >= 0);
  m_variable = variable;
}

void System::setFilename(string filename) {
  m_filename = filename;
}

void System::setThreadNumber(int THREAD_NUM) {
  assert(THREAD_NUM >= 0);
  m_thread_num = THREAD_NUM;
}

void System::setImportanceSampling(bool ImportanceSampling_input, double timeStep) {
  if(ImportanceSampling_input == true){
    m_importanceSampling_input = ImportanceSampling_input;
    m_timeStep = timeStep;
  }
}

void System::setLearningRate(double learningRate) {
  assert(learningRate >= 0);
  m_learningRate = learningRate;
}

void System::setHamiltonian(Hamiltonian *hamiltonian) {
    m_hamiltonian = hamiltonian;
}

void System::setWaveFunction(WaveFunction *waveFunction) {
    m_waveFunction = waveFunction;
}

void System::setInitialState(InitialState *initialState) {
    m_initialState = initialState;
}

void System::setSampler(Sampler *sampler) {
  m_sampler = sampler;
}

double System::getSumOverR_iSquared() {
  double sum_r_squared = 0;
  for ( auto particle : m_particles) {
    for ( auto comp : particle->getPosition() ) {
      sum_r_squared += comp*comp;
    } // end loop over dimensions
  } // end loop over particles
  return sum_r_squared;
} // end getSumOverR_iSquared


void System::print( std::vector<double> v, int N){
    std::cout << '\n' << "[ ";
    for (int i = 0; i < N; i++) {
        std::cout<< v.at(i) << " ";
    }
    std::cout << " ]" << '\n';
}
