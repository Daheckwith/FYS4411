#pragma once
#include <vector>
#include <Math/random.h>

using std::string;

class System {
public:
    System();
    System(int seed);
    bool metropolisStep             ();
    bool importanceStep             ();
    void runMetropolisSteps         (int numberOfMetropolisSteps, bool DDeriv_input, bool Boot_Blocking_input);

    void setNumberOfParticles       (int numberOfParticles);
    void setNumberOfDimensions      (int numberOfDimensions);
    void setStepLength              (double stepLength);
    void setEquilibrationFraction   (double equilibrationFraction);
    void setVariable                (double variable);
    void setFilename                (string filename);
    void setThreadNumber            (int THREAD_NUM);
    void setImportanceSampling      (bool ImportanceSampling_input, double timeStep);
    void setLearningRate            (double learningRate);

    void setHamiltonian             (class Hamiltonian* hamiltonian);
    void setWaveFunction            (class WaveFunction* waveFunction);
    void setInitialState            (class InitialState* initialState);
    void setSampler                 (class Sampler* sampler);

    class WaveFunction*             getWaveFunction()   { return m_waveFunction; }
    class Hamiltonian*              getHamiltonian()    { return m_hamiltonian; }
    class Sampler*                  getSampler()        { return m_sampler; }
    std::vector<class Particle*>    getParticles()      { return m_particles; }
    class Random*                   getRandomEngine()   { return m_random; }

    int getNumberOfParticles()          { return m_numberOfParticles; }
    int getNumberOfDimensions()         { return m_numberOfDimensions; }
    int getNumberOfMetropolisSteps()    { return m_numberOfMetropolisSteps; }
    int getThreadNumber()               { return m_thread_num; }

    double getEquilibrationFraction()   { return m_equilibrationFraction; }
    double getVariable()                { return m_variable; }
    double getTimeStep()                { return m_timeStep; }
    double getStepLength()              { return m_stepLength; }
    double getLearningRate()            { return m_learningRate; }

    string getFilename()                { return m_filename; }
    bool   getImportanceSampling()      { return m_importanceSampling_input; }

    double getSumOverR_iSquared();
    void print(std::vector<double> v, int N);

private:
    int                             m_numberOfParticles = 0;
    int                             m_numberOfDimensions = 0;
    int                             m_numberOfMetropolisSteps = 0;
    int                             m_thread_num = 0;

    double                          m_stepLength = 0.1;
    double                          m_timeStep = 0;
    double                          m_equilibrationFraction = 0.0;
    double                          oldWaveFunction = 0;
    double                          newWaveFunction = 0;
    double                          m_variable = 0;
    double                          m_learningRate = 0;

    string                          m_filename;
    bool                            m_importanceSampling_input = false;

    class WaveFunction*             m_waveFunction = nullptr;
    class Hamiltonian*              m_hamiltonian = nullptr;
    class InitialState*             m_initialState = nullptr;
    class Sampler*                  m_sampler = nullptr;
    std::vector<class Particle*>    m_particles = std::vector<class Particle*>();
    class Random*                   m_random = nullptr;
};
