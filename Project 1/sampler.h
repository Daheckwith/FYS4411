#pragma once
#include <vector>

using std::string;

class Sampler {
public:
    Sampler(class System* system);

    virtual void sample(bool acceptedStep, bool DDeriv_input);
    virtual void computeAverages();

    void setNumberOfMetropolisSteps(int steps);
    void setOneBodyDensity(double min, double max, int numberOfBins);
    void writeOneBodyDensity(string &filename);
    void printmat(double **A, int n, int m);

    std::vector<double> randomEnergyVec(int N);
    std::vector<double> slicing(std::vector<double> v, int X, int Y);
    double computeAverage(std::vector<double> v);
    double computeVariance(std::vector<double> v, double mean);
    void bootstrap(int nBoots);
    void blocking();

    void sampleEnergy(string filename, bool Boot_Blocking_input);

    int    getStepNumber()                  { return m_stepNumber; }
    double getEnergy()                      { return m_energy; }
    double getLocalEnergy()                 { return m_localEnergy; }
    std::vector<double> getLocalEnergyVec() { return m_localEnergyVec; }
    double getexpectWFDerivXEnergy()        { return m_expectWFDerivXEnergy; }
    double getexpectWFDerivXexpectEnergy () { return m_expectWFDerivXexpectEnergy; }

protected:
    int     m_numberOfMetropolisSteps = 0;
    int     m_stepNumber = 0;
    int     m_acceptedSteps = 0;
    int     m_numberOfBins = 0;

    double  m_energy = 0;
    double  m_energy_sqrd = 0;
    double  m_localEnergy = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy_sq = 0;
    double  m_variance = 0;
    double  m_acceptanceRatio = 0;
    double  m_min = 0;
    double  m_max = 0;
    double  m_binWidth = 0;
    double  **m_bins = nullptr;

    std::vector<double>  m_localEnergyVec;


    double  m_WFDeriv = 0;
    double  m_cumulativeWFDeriv = 0;
    double  m_cumulativeWFDerivXEnergy = 0;
    double  m_expectWFDerivXEnergy = 0;
    double  m_expectWFDerivXexpectEnergy = 0;

    // Blocking & Bootstraping
    double  m_bootAvg = 0;
    double  m_bootVar = 0;

    double  m_blockingAvg = 0;
    double  m_blockingVar = 0;

    class System* m_system = nullptr;
};
