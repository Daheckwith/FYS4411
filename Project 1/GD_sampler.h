#pragma once
#include "sampler.h"

class GD_sampler : public Sampler {
  /*
  The Gradient Descent sampler.
  Derived from the class Sampler
  */
public:
  GD_sampler(class System* system);
  void sample(bool acceptedStep, bool DDeriv_input);
  void computeAverages();
};
