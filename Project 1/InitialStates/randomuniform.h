#pragma once
#include "initialstate.h"

class RandomUniform : public InitialState {
public:
    RandomUniform(System* system, int numberOfDimensions, int numberOfParticles, double characteristicLength);
    void setupInitialState();

private:
    std::vector<double> m_posParameters;
};
