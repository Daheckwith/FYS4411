#pragma once
#include <vector>

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computeLocalEnergy(std::vector<class Particle*> particles, bool DDeriv_input) = 0;

protected:
    class System* m_system = nullptr;
};
