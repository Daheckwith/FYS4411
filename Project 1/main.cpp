#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <omp.h>
#include "sampler.h"
#include "GD_sampler.h"
#include "system.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/correlated.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/elliptical_harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "Math/random.h"

using namespace std;

/* Functions */
void onebody_density(double alpha);
void bruteforce_importance_param(double alpha_min, double alpha_max, double alpha_spacing);
void bruteforce_importance(double alpha_min, double alpha_max, double alpha_spacing);
void gradient_descent(double alpha);
void file_initiater_1(string &filename, bool &DDeriv_input,
  bool &ImportanceSampling_input, bool &Boot_Blocking_input, int numberOfDimensions,
  int numberOfParticles, int numberOfSteps, int &NUM_THREADS);
/* Functions */

int main() {
    /*Uncomment to run*/
    // onebody_density(0.5);
    // bruteforce_importance_param(0.2, 0.95, 0.05); // Simple Gaussian
    // bruteforce_importance(0.2, 0.95, 0.05); // Simple Gaussian
    // bruteforce_importance(0.2, 0.8, 0.1); // Correlated
    gradient_descent(0.4);
    // gradient_descent(0.6);
    return 0;
}

void onebody_density(double alpha) {
    /*
    Calculates the one-body density for
    Interacting case
    WaveFunction: Correlated
    Hamiltonian: EllipticalHarmonicOscillator
        gamma = 2.82843
        beta = gamme
        a = 0.00433
    Non-interacting case
    WaveFunction: Correlated
    Hamiltonian: EllipticalHarmonicOscillator
        gamma = 1
        beta = gamme
        a = 0
    Ideal (SimpleGaussian)
    WaveFunction: SimpleGaussian
    Hamiltonian: HarmonicOscillator
    */

    std::cout << "Running onebody_density" << '\n';
    // Seed for the random number generator
    int seed = 2020;

    int numberOfDimensions  = 3;
    int numberOfParticles   = 10;
    int base = 2; int exponent = 10; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
    // int base = 2; int exponent = 21; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
    // int base = 10; int exponent = 1; int numberOfSteps = pow(base, exponent); // MC-cycles

    double stepLength           = 1;            // Metropolis step length. (BF)
    double timeStep             = 0.1;         // Metropolis-Hastings time step. (IMP)

    double omega                = 1.0;          // Oscillator frequency.
    double equilibration        = 0.1;          // Amount of the total steps used for equilibration.
    double characteristicLength = 1;

    double gamma                = 2.82843;      // omega_z/omega_{ho}
    // double gamma                = 1;      // omega_z/omega_{ho}/
    double beta                 = gamma;
    double bosonDiameter        = 0.00433;      // The hard-core diameter for the bosons
    // double bosonDiameter        = 0;      // The hard-core diameter for the bosons

    double start, stop, period;

    double min, max; int numberOfBins;
    string filename_ = "./Data/onebody_";
    min = -4; max = 4; numberOfBins = 100;

    string filename = "./Data/vmc_attributes_";
    bool DDeriv_input, ImportanceSampling_input, Boot_Blocking_input;
    int NUM_THREADS;
    ofstream ofile;

    std::cout << "Alpha: " << alpha << '\n';
    file_initiater_1(filename, DDeriv_input,
      ImportanceSampling_input, Boot_Blocking_input, numberOfDimensions,
      numberOfParticles, numberOfSteps, NUM_THREADS);


    start = omp_get_wtime();
    System* system = new System();
    // System* system = new System(seed);
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setFilename                 (filename);
    system->setVariable                 (alpha);
    system->setThreadNumber             (omp_get_thread_num());
    system->setImportanceSampling       (ImportanceSampling_input, timeStep);

    system->setSampler                  (new Sampler(system));
    system->setInitialState             (new RandomUniform(system,
                                                numberOfDimensions,
                                                numberOfParticles,
                                                characteristicLength));

    // system->setWaveFunction             (new SimpleGaussian(system, alpha));
    // system->setHamiltonian              (new HarmonicOscillator(system, omega));

    system->setWaveFunction             (new Correlated(system, alpha, beta, bosonDiameter));
    system->setHamiltonian              (new EllipticalHarmonicOscillator(system, gamma, bosonDiameter));

    system->getSampler()->setOneBodyDensity(min, max, numberOfBins);
    system->runMetropolisSteps          (numberOfSteps, DDeriv_input, Boot_Blocking_input);
    system->getSampler()->writeOneBodyDensity(filename_);

    stop = omp_get_wtime();
    period = stop - start;
    ofile.open(filename, ios::app);
    ofile << "\n" << "\n";
    ofile << "Time elpased for " << NUM_THREADS << " thread(s): "  << period << '\n';
    ofile.close();
} // end onebody_density


void bruteforce_importance_param(double alpha_min, double alpha_max, double alpha_spacing) {
    /*
    Does brute-force and importance variational Monte-Carlo. The analysis is made
    for an array of user defined values of alpha. Testing how paramters such as
    stepLength (BF) and timeStep (IMP) and number of MC-cycles affect the results
    such as energy, variance and acceptance ratio. Prest here is highlighted by
    comments. The correlated case can be added see onebody_density.
    But we didn't need that in our analysis.
    */

    std::cout << "Running bruteforce_importance_param" << '\n';
    // Seed for the random number generator
    int seed = 2020;

    int numberOfDimensions  = 3;
    int numberOfParticles   = 100;
    //int base = 2; int exponent = 21; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
    // int base = 2; int exponent = 20; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
    int base = 10; int exponent = 3; int numberOfSteps = pow(base, exponent); // MC-cycles

    double omega                = 1.0;          // Oscillator frequency.
    double alpha                = alpha_min;    // Variational parameter.
    double equilibration        = 0.1;          // Amount of the total steps used for equilibration.
    double characteristicLength = 1;

    double start, stop, period;

    string filename = "./Data/vmc_attributes_";
    bool DDeriv_input, ImportanceSampling_input, Boot_Blocking_input;
    int NUM_THREADS;
    ofstream ofile;

    file_initiater_1(filename, DDeriv_input,
    ImportanceSampling_input, Boot_Blocking_input, numberOfDimensions,
    numberOfParticles, numberOfSteps, NUM_THREADS);


    int n_steps = (alpha_max-alpha_min)/alpha_spacing;
    n_steps++;

    std::vector<double> alphaVec;
    for (int i = 0; i < n_steps; i++) {
    alpha = alpha_min + i*alpha_spacing;
    alphaVec.push_back(alpha);
    }


    double steps[] = {0.001, 0.01, 0.1, 0.5, 1};
    double numberOfSteps_MC[] =  {pow(2, 16), pow(2, 18), pow(2, 19), pow(2, 20)};

    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (auto step : numberOfSteps_MC){
    // for (auto step : steps){
      for (int i = 0; i < alphaVec.size(); i++){
          System* system = new System();
          //System* system = new System(seed);
          system->setEquilibrationFraction    (equilibration);
          // system->setStepLength               (step); // steps
          system->setStepLength               (1); // numberOfSteps_MC
          system->setFilename                 (filename);
          system->setVariable                 (alphaVec.at(i));
          system->setThreadNumber             (omp_get_thread_num());
          // system->setImportanceSampling       (ImportanceSampling_input, step); // steps
          system->setImportanceSampling       (ImportanceSampling_input, 0.1); // numberOfSteps_MC

          system->setSampler                  (new Sampler(system));
          system->setInitialState             (new RandomUniform(system,
                                                      numberOfDimensions,
                                                      numberOfParticles,
                                                      characteristicLength));
          system->setWaveFunction             (new SimpleGaussian(system, alphaVec.at(i)));
          system->setHamiltonian              (new HarmonicOscillator(system, omega));

          // system->runMetropolisSteps          (numberOfSteps, DDeriv_input, Boot_Blocking_input); // steps
          system->runMetropolisSteps          (step, DDeriv_input, Boot_Blocking_input); // numberOfSteps_MC
      }
    } // numberOfSteps_MC or steps for-loop

    stop = omp_get_wtime();
    period = stop - start;
    ofile.open(filename, ios::app);
    ofile << "\n" << "\n";
    ofile << "Time elpased for " << NUM_THREADS << " thread(s): "  << period << '\n';
    ofile.close();
} // end bruteforce_importance_param

void bruteforce_importance(double alpha_min, double alpha_max, double alpha_spacing) {
  /*
  Does brute-force and importance variational Monte-Carlo. See onebody_density
  for presets. The analysis is made for an array of user defined values of alpha.
  */
  std::cout << "Running bruteforce_importance" << '\n';
  // Seed for the random number generator
  int seed = 2020;

  int numberOfDimensions  = 3;
  int numberOfParticles   = 10;
  // int base = 2; int exponent = 21; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
  int base = 2; int exponent = 20; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
  // int base = 10; int exponent = 2; int numberOfSteps = pow(base, exponent); // MC-cycles

  double stepLength           = 1;            // Metropolis step length. (BF)
  double timeStep             = 0.1;         // Metropolis-Hastings time step. (IMP)

  double omega                = 1.0;          // Oscillator frequency.
  double alpha                = alpha_min;    // Variational parameter.
  double equilibration        = 0.1;          // Amount of the total steps used for equilibration.
  double characteristicLength = 1;

  double gamma                = 2.82843;      // omega_z/omega_{ho}
  double beta                 = gamma;
  double bosonDiameter        = 0.00433;      // The hard-core diameter for the bosons

  double start, stop, period;

  string filename = "./Data/vmc_attributes_";
  bool DDeriv_input, ImportanceSampling_input, Boot_Blocking_input;
  int NUM_THREADS;
  ofstream ofile;

  file_initiater_1(filename, DDeriv_input,
    ImportanceSampling_input, Boot_Blocking_input, numberOfDimensions,
    numberOfParticles, numberOfSteps, NUM_THREADS);


  int steps = (alpha_max-alpha_min)/alpha_spacing;
  steps++;

  std::vector<double> alphaVec;
  for (int i = 0; i < steps; i++) {
    alpha = alpha_min + i*alpha_spacing;
    alphaVec.push_back(alpha);
  }


  start = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < alphaVec.size(); i++) {
    System* system = new System();
    // System* system = new System(seed);
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setFilename                 (filename);
    system->setVariable                 (alphaVec.at(i));
    system->setThreadNumber             (omp_get_thread_num());
    system->setImportanceSampling       (ImportanceSampling_input, timeStep);

    system->setSampler                  (new Sampler(system));
    system->setInitialState             (new RandomUniform(system,
                                                numberOfDimensions,
                                                numberOfParticles,
                                                characteristicLength));
    // system->setWaveFunction             (new SimpleGaussian(system, alphaVec.at(i)));
    // system->setHamiltonian              (new HarmonicOscillator(system, omega));

    system->setWaveFunction             (new Correlated(system, alphaVec.at(i), beta, bosonDiameter));
    system->setHamiltonian              (new EllipticalHarmonicOscillator(system, gamma, bosonDiameter));

    system->runMetropolisSteps          (numberOfSteps, DDeriv_input, Boot_Blocking_input);
  }
  stop = omp_get_wtime();
  period = stop - start;
  ofile.open(filename, ios::app);
  ofile << "\n" << "\n";
  ofile << "Time elpased for " << NUM_THREADS << " thread(s): "  << period << '\n';
  ofile.close();
} // end bruteforce_importance

void gradient_descent(double alpha) {
    /*
    Gardient Descent method for finding optimal alpha. Takes in a star alpha
    and applies gradient descent from that point onward. It's possible to change
    the content in the two arrays numberOfSteps_MC[] and learningRates[]. Depending
    on the task and the precision needed. Note that the length of the array equals
    the number of "VMC-rounds".
    */
    std::cout << "Running gradient_descent" << '\n';
    std::cout << "Start alpha: " << alpha << '\n';
    // Seed for the random number generator
    int seed = 2020;

    int numberOfDimensions  = 3;
    int numberOfParticles   = 2;
    // int base = 2; int exponent = 21; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
    int base = 2; int exponent = 20; int numberOfSteps = pow(base, exponent); // MC-cycles for blocking
    // int base = 10; int exponent = 3; int numberOfSteps = pow(base, exponent); // MC-cycles

    double stepLength           = 1;       // Metropolis step length. (BF)
    double timeStep             = 0.1;    // Metropolis-Hastings time step. (IMP)

    double omega            = 1.0;         // Oscillator frequency
    double equilibration    = 0.1;         // Amount of the total steps used for equilibration.
    double characteristicLength = 1;

    // double gamma                = 2.82843;      // omega_z/omega_{ho}
    double gamma                = 1;      // omega_z/omega_{ho}
    double beta                 = gamma;
    double bosonDiameter        = 0.00433;      // The hard-core diameter for the bosons

    // double learningRate     = 0.001;       // Gamma_k in Gradient Descent
    double tol              = 1e-5;       // Tolarance to stop Gradient Descent
    double energyGradient, old_alpha;
    old_alpha = 0;

    double start, stop, period;

    string filename = "./Data/gd_vmc_attributes_";
    bool DDeriv_input, ImportanceSampling_input, Boot_Blocking_input;
    int NUM_THREADS;
    ofstream ofile;

    file_initiater_1(filename, DDeriv_input,
    ImportanceSampling_input, Boot_Blocking_input, numberOfDimensions,
    numberOfParticles, numberOfSteps, NUM_THREADS);

    double numberOfSteps_MC[] =  {pow(2, 17), pow(2, 18), pow(2,18)};
    double learningRates[]    = {0.001, 0.001, 0.0001};
    double iterations[]       = {50, 50, 50};


    // double numberOfSteps_MC[] =  {pow(2, 19)}; // After finding a convergence value
    // double learningRates[]    = {0.0001};
    // double iterations[]       = {10};

    int arr_size = sizeof(numberOfSteps_MC)/sizeof(numberOfSteps_MC[0]);
    double step, learningRate, max_iteration;
    start = omp_get_wtime();
    for (int i = 0; i < arr_size; i++) {
        step = numberOfSteps_MC[i];
        learningRate = learningRates[i];
        max_iteration = iterations[i];
        std::cout << "---------------------------------------------------" << '\n';
        std::cout << "Number of MC-cycles: " << step << '\n';
        std::cout << "learningRate: " << learningRate << '\n';
        std::cout << "max_iteration: " << max_iteration << '\n';
        int s = 0;
        while (abs(alpha - old_alpha) > tol  && s < max_iteration) {
          System* system = new System();
          // System* system = new System(seed);
          system->setEquilibrationFraction    (equilibration);
          system->setStepLength               (stepLength);
          system->setFilename                 (filename);
          system->setVariable                 (alpha);
          system->setThreadNumber             (omp_get_thread_num());
          system->setImportanceSampling       (ImportanceSampling_input, timeStep);
          system->setLearningRate             (learningRate);

          //system->setSampler                  (new Sampler(system));
          system->setSampler                  (new GD_sampler(system));
          system->setInitialState             (new RandomUniform(system,
                                                      numberOfDimensions,
                                                      numberOfParticles,
                                                      characteristicLength));
          // system->setWaveFunction             (new SimpleGaussian(system, alpha));
          // system->setHamiltonian              (new HarmonicOscillator(system, omega));

          system->setWaveFunction             (new Correlated(system, alpha, beta, bosonDiameter));
          system->setHamiltonian              (new EllipticalHarmonicOscillator(system, gamma, bosonDiameter));

          system->runMetropolisSteps          (step, DDeriv_input, Boot_Blocking_input);
          energyGradient = system->getWaveFunction()->energyGradient();
          old_alpha = alpha;
          // std::cout << "energyGradient: " << energyGradient << " Supposed: " << 0.5*old_alpha - (0.5/(old_alpha*old_alpha*old_alpha)) << '\n';
          alpha -= learningRate*energyGradient;
          s += 1;
          std::cout << "Iteration: " << s << " old alpha: " << old_alpha << " alpha: " << alpha << '\n' << '\n';
        }
    }
    stop = omp_get_wtime();
    period = stop - start;
    ofile.open(filename, ios::app);
    ofile << "\n" << "\n";
    ofile << "Time elpased for " << NUM_THREADS << " thread(s): "  << period << '\n';
    ofile.close();
} // end gradient_descent

void file_initiater_1(string &filename, bool &DDeriv_input,
  bool &ImportanceSampling_input, bool &Boot_Blocking_input, int numberOfDimensions,
  int numberOfParticles, int numberOfSteps, int &NUM_THREADS){
  /*
  Given inputs from the command line it generates a corresponding
  file with a suiting filename containing some of the information
  needed to distinguish the files.
  */

  std::cout << "Number of Dimensions: " << numberOfDimensions << '\n';
  std::cout << "Number of Particles: " << numberOfParticles << '\n';
  std::cout << "Number of MC-cycles: " << numberOfSteps << '\n';

  std::cout << "Select the number of OpenMP-threads: ";
  std::cin >> NUM_THREADS;
  std::cout << "You've selected " << NUM_THREADS << " thread(s)." << '\n';

  omp_set_num_threads(NUM_THREADS);

  std::cout << "Metropolis Computation: Type \"1\" for Importance Sampling or \"0\" for Brute Force Metropolis ";
  std::cin >> ImportanceSampling_input;
  if (ImportanceSampling_input == true) {
    std::cout << "Importance Sampling (True/1): " << ImportanceSampling_input << '\n';
    filename.append("Importance_");
  } else {
    std::cout << "Brute Force Metropolis (False/0): " << ImportanceSampling_input << '\n';
  }

  std::cout << "Bootstraping and blocking analysis: Type \"1\" to enable or type \"0\" to disable " << '\n';
  std::cin >> Boot_Blocking_input;
  if (Boot_Blocking_input == true) {
    std::cout << "Boot & Blocking Enabled (True/1): " << Boot_Blocking_input << '\n';
    filename.append("BB_");
  } else {
    std::cout << "Boot & Blocking Disabled (False/0): " << Boot_Blocking_input << '\n';
  }

  std::cout << "Local Energy Computation: Type \"1\" for Analytical or \"0\" for Numerical\n\
  Note the analyical solution is faster! ";
  std::cin >> DDeriv_input;
  if (DDeriv_input == true) {
    filename.append("analyticalDeriv_");
    std::cout << "True/1: You typed in " << DDeriv_input << '\n';
    std::cout << "Running analyical VMC over " << NUM_THREADS << " thread(s) with the parameters "\
    << numberOfDimensions << " dimension(s) and " << numberOfParticles << " particle(s)." << '\n';
  } else {
    filename.append("numericalDeriv_");
    std::cout << "False/0: You typed in " << DDeriv_input << '\n';
    std::cout << "Running numerical VMC over " << NUM_THREADS << " thread(s) with the parameters "\
    << numberOfDimensions << " dimension(s) and " << numberOfParticles << " particle(s)." << '\n';
  }
  filename.append( to_string(NUM_THREADS) + "THREADS_" );
  filename.append( to_string(numberOfDimensions) + "d_" );
  filename.append( to_string(numberOfParticles) + "p_");
  filename.append( to_string(numberOfSteps) );


  ofstream ofile;
  if (ImportanceSampling_input == true) {
    ofile.open(filename, ios::out | ios::trunc);
    ofile << std::setw(18) << std::setprecision(8) << "ThreadNr.";
    ofile << std::setw(18) << std::setprecision(8) << "Alpha";
    ofile << std::setw(18) << std::setprecision(8) << "Energy";
    ofile << std::setw(18) << std::setprecision(8) << "EnergySquared";
    ofile << std::setw(18) << std::setprecision(8) << "Variance";
    ofile << std::setw(18) << std::setprecision(8) << "BootVar";
    ofile << std::setw(18) << std::setprecision(8) << "BlockingVar";
    ofile << std::setw(18) << std::setprecision(8) << "AcceptanceRatio";
    ofile << std::setw(18) << std::setprecision(8) << "TotalSteps";
    ofile << std::setw(18) << std::setprecision(8) << "TimeStep";
    ofile << std::setw(18) << std::setprecision(8) << "LearningRate" << "\n";
    ofile.close();
  } else {
    ofile.open(filename, ios::out | ios::trunc);
    ofile << std::setw(18) << std::setprecision(8) << "ThreadNr.";
    ofile << std::setw(18) << std::setprecision(8) << "Alpha";
    ofile << std::setw(18) << std::setprecision(8) << "Energy";
    ofile << std::setw(18) << std::setprecision(8) << "EnergySquared";
    ofile << std::setw(18) << std::setprecision(8) << "Variance";
    ofile << std::setw(18) << std::setprecision(8) << "BootVar";
    ofile << std::setw(18) << std::setprecision(8) << "BlockingVar";
    ofile << std::setw(18) << std::setprecision(8) << "AcceptanceRatio";
    ofile << std::setw(18) << std::setprecision(8) << "TotalSteps";
    ofile << std::setw(18) << std::setprecision(8) << "StepLength";
    ofile << std::setw(18) << std::setprecision(8) << "LearningRate" << "\n";
    ofile.close();
  }
} // end file_initiater_1
