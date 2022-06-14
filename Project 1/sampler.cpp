#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include "sampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

#include "blocker.h"

using std::cout;
using std::endl;
using std::ios;
using std::to_string;
using namespace std::chrono;


Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::sample(bool acceptedStep, bool DDeriv_input) {
  //std::cout << "Class Sampler Member sample" << '\n';
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

    m_localEnergyVec.push_back(m_localEnergy);
    m_cumulativeEnergy    += m_localEnergy;
    m_cumulativeEnergy_sq += m_localEnergy*m_localEnergy;

    if (m_numberOfBins > 0) {
        int dim = m_system->getNumberOfDimensions();
        int idx;
        std::vector<double> position;
        for (auto particle : m_system->getParticles()) {
            position = particle->getPosition();
            for (int i = 0; i < dim; i++) {
                if (m_min < position[i] && position[i] < m_max) {
                    idx = (int)floor((position[i] - m_min)/m_binWidth);
                } else if (position[i] < m_min) {
                    idx = 0;
                } else if (position[i] > m_max) {
                    idx = m_numberOfBins - 1;
                }
                m_bins[i][idx] += 1.0;
            }
        } // auto particle
    } // if-test
} // end sample


void Sampler::computeAverages() {
    m_energy          = m_cumulativeEnergy / m_numberOfMetropolisSteps;
    m_energy_sqrd     = m_cumulativeEnergy_sq / m_numberOfMetropolisSteps;
    m_variance        = m_energy_sqrd - m_energy*m_energy;
    m_acceptanceRatio = double (m_acceptedSteps) / double (m_numberOfMetropolisSteps);
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}

void Sampler::printmat(double **A, int n, int m)
{
    for (size_t i = 0; i < n; ++i){
        printf("| ");
        for (size_t j = 0; j < m; ++j){
            printf("%4.0lf ", A[i][j]);
        }
        printf("|\n");
    }
}

void Sampler::setOneBodyDensity(double min, double max, int numberOfBins) {
    int dim = m_system->getNumberOfDimensions();

    m_numberOfBins = numberOfBins;
    m_min = min; m_max = max;
    m_binWidth = (m_max - m_min)/m_numberOfBins;
    // printf("%s\n", m_bins); // double**
    // printf("%s\n", *m_bins); // double*
    // printf("%s\n", **m_bins); // double
    // m_bins = (double **) calloc(dim, sizeof(double*));
    m_bins = (double **) calloc(dim, sizeof *m_bins);
    for (int i = 0; i < dim; i++) {
        // m_bins[i] = (double*) calloc(m_numberOfBins, sizeof(double));
        m_bins[i] = (double*) calloc(m_numberOfBins, sizeof **m_bins);
        // m_bins[i] = (double*) calloc(m_numberOfBins, sizeof *(m_bins[i]));
    }
    // The comments are some testing on my side to find different ways to allocate memory
}

void Sampler::writeOneBodyDensity(string &filename) {
    int dim = m_system->getNumberOfDimensions();
    int N = m_system->getNumberOfParticles();
    int numberOfSteps = m_system->getNumberOfMetropolisSteps();
    int scalingfactor = N*numberOfSteps;

    // printmat(m_bins, dim, m_numberOfBins);
    // std::cout << '\n';

    filename.append( "1THREADS_" );
    filename.append( to_string(dim) + "d_" );
    filename.append( to_string(N) + "p_");
    filename.append( to_string(numberOfSteps) );

    std::ofstream ofile;
    ofile.open(filename, ios::out | ios::trunc);

    ofile << "numberOfBins: "
    << m_numberOfBins << " min: " << m_min << " max: " << m_max << "\n";

    for (int i = 0; i < m_numberOfBins; i++) {
        ofile << std::setw(18) << m_bins[0][i];
        ofile << std::setw(18) << m_bins[1][i];
        ofile << std::setw(18) << m_bins[2][i] << "\n";
    }
    // for (int i = 0; i < dim; i++) {
    //     for (int j = 0; j < m_numberOfBins; j++) {
    //         ofile << std::setw(18) << m_bins[i][j];
    //         // ofile << std::setw(10) << m_bins[i][j];
    //         // ofile << std::setw(10) << std::setprecision(8) << m_bins[i][j]/scalingfactor;
    //     }
    //     ofile << "\n";
    // }
    ofile.close();
    free(m_bins[0]);
    free(m_bins);
}

std::vector<double> Sampler::randomEnergyVec(int N) {
  //std::cout << "Class Sampler Memeber randomEnergyVec" << '\n';
  int randInt; // int N = m_numberOfMetropolisSteps;
  double randEnergy; std::vector<double> randVec(N, 0);
  for (int i = 0; i < N; i++) {
    randInt    = m_system->getRandomEngine()->nextInt(N - 1);
    randEnergy = m_localEnergyVec.at(randInt);
    // randVec.push_back(randEnergy);
    randVec.at(i) = randEnergy;
  }
  return randVec;
}

std::vector<double> Sampler::slicing(std::vector<double> v, int X, int Y) {
  //std::cout << "class Sampler Memeber Slicing" << '\n';
  auto start = v.begin() + X;
  auto end = v.begin() + Y + 1;

  if (Y == v.size()) {
    std::vector<double> result(Y - X);
    std::copy(start, end, result.begin());
    return result;
  } else {
    std::vector<double> result(Y - X + 1);
    std::copy(start, end, result.begin());
    return result;
  } // if-else statement
}

double Sampler::computeAverage(std::vector<double> v) {
  //std::cout << "Class Sampler Memeber computeAverage" << '\n';
  double sum = 0; int N = v.size();
  for (int i = 0; i < N; i++) {
    sum += v.at(i);
  }
  return sum/N;
}

double Sampler::computeVariance(std::vector<double> v, double mean) {
  //std::cout << "Class Sampler Memeber computeVariance" << '\n';
  double var = 0; int N = v.size();
  for (int i = 0; i < N; i++) {
    var += (v.at(i) - mean) * (v.at(i) - mean);
  }
  var /= N;
  return var;
}

void Sampler::bootstrap(int nBoots) {
    double avg, var;

    std::vector<double> bootVarVec(nBoots, 0);
    double bootAvg, bootVar, bootAvgVar;

    int N = (int) m_numberOfMetropolisSteps/1000;
    std::vector<double> tempVec(N, 0);

    for (int i = 0; i < nBoots; i++) {
      tempVec = randomEnergyVec(N);
      bootAvg = computeAverage(tempVec);
      bootVar = computeVariance(tempVec, bootAvg);
      bootVarVec.at(i) = bootVar;
    }

    bootAvgVar = computeAverage(bootVarVec);
    m_bootVar  = bootAvgVar;
} // end bootstrap



void Sampler::blocking() {
    // Constructor accepts a vector
    // This vector must have a size which is a power of 2.
    Blocker block(m_localEnergyVec);

    // the public variables mean, mse_mean, stdErr and mse_stdErr are output
    // printf("Expected value = %g (with mean sq. err. = %g) \n", block.mean, block.mse_mean);
    // printf("Standard error = %g (with mean sq. err. = %g) \n", block.stdErr, block.mse_stdErr);
    m_blockingVar = block.stdErr*block.stdErr;
}

void Sampler::sampleEnergy(string filename, bool Boot_Blocking_input) {
    // std::cout << "Class Sampler Member sampleEnergy" << '\n';

    if (Boot_Blocking_input) {
        // Uncomment bootstrap if you want to use it
        // bootstrap(10000);
        // bootstrap(m_numberOfMetropolisSteps);
        blocking();
    }

    double alpha = m_system->getVariable();

    std::ofstream ofile;
    ofile.open(filename, ios::app);
    //ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << std::setw(18) << std::setprecision(8) << m_system->getThreadNumber();
    ofile << std::setw(18) << std::setprecision(8) << alpha;
    ofile << std::setw(18) << std::setprecision(8) << m_energy;
    ofile << std::setw(18) << std::setprecision(8) << m_energy_sqrd;
    ofile << std::setw(18) << std::setprecision(8) << m_variance;
    ofile << std::setw(18) << std::setprecision(8) << m_bootVar;
    ofile << std::setw(18) << std::setprecision(8) << m_blockingVar;
    ofile << std::setw(18) << std::setprecision(8) << m_acceptanceRatio;
    if (m_system->getImportanceSampling()) {
    ofile << std::setw(18) << std::setprecision(8) << m_numberOfMetropolisSteps;
    ofile << std::setw(18) << std::setprecision(8) << m_system->getTimeStep();
    ofile << std::setw(18) << std::setprecision(8) << m_system->getLearningRate() << "\n";
    } else {
    ofile << std::setw(18) << std::setprecision(8) << m_numberOfMetropolisSteps;
    ofile << std::setw(18) << std::setprecision(8) << m_system->getStepLength();
    ofile << std::setw(18) << std::setprecision(8) << m_system->getLearningRate() << "\n";
    }
    ofile.close();
} // end sampleEnergy
