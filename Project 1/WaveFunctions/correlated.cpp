#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <omp.h>
#include "wavefunction.h"
#include "correlated.h"
#include "../GD_sampler.h"
#include "../system.h"
#include "../particle.h"


Correlated::Correlated(System* system, double alpha, double beta, double bosonDiameter) :
      WaveFunction(system) {
  assert(alpha >= 0);
  assert(beta  >= 0);
  m_numberOfDimensions = system->getNumberOfDimensions();
  assert(m_numberOfDimensions == 3);
  m_alpha         = alpha;
  m_beta          = beta;
  m_bosonDiameter = bosonDiameter;
  std::cout << "m_beta: " << m_beta << " m_bosonDiameter: " << m_bosonDiameter << '\n';
}

double Correlated::evaluate(std::vector<class Particle*> particles) {
  //std::cout << "Class Correlated Memeber evaluate" << '\n';
  double waveFunction = oneBodyPart(particles);
  int N   = m_system->getNumberOfParticles();
  // #pragma omp parallel for reduction(*: waveFunction) num_threads(2)
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      waveFunction *= correlatedPart(particles.at(i), particles.at(j));
    }
  }
  return waveFunction;
}


double Correlated::oneBodyPart(std::vector<class Particle*> particles) {
  //std::cout << "Class Correlated Memeber oneBodyPart" << '\n';
  double positionSum = 0;
  std::vector<double> position;
  // #pragma omp parallel for reduction(+ : positionSum) num_threads(4)
  for (auto particle : particles) {
    position = particle->getPosition();
    positionSum += position[0]*position[0] + position[1]*position[1];
    positionSum += m_beta*position[2]*position[2];
  }
  return exp(-m_alpha*positionSum);
}

// double Correlated::computeDistance(Particle* p1, Particle* p2) {
//   //std::cout << "Class Correlated Member computeDistance" << '\n';
//   std::vector<double> pos1 = p1->getPosition();
//   std::vector<double> pos2 = p2->getPosition();
//   double dist_sqrd, diff;
//   dist_sqrd = diff = 0;
//   for (int i = 0; i < m_numberOfDimensions; i++) {
//     diff = pos1[i] - pos2[i];
//     dist_sqrd += diff*diff;
//   }
//   return sqrt(dist_sqrd);
// }

double Correlated::computeDistance(std::vector<double> pos1, std::vector<double> pos2) {
  //std::cout << "Class Correlated Member computeDistance" << '\n';
  double dist_sqrd, diff;
  dist_sqrd = diff = 0;
  for (int i = 0; i < m_numberOfDimensions; i++) {
    diff = pos1[i] - pos2[i];
    dist_sqrd += diff*diff;
  }
  return sqrt(dist_sqrd);
}

double Correlated::correlatedPart(Particle* p1, Particle* p2) {
  //std::cout << "Class Correlated Member interactionPart" << '\n';
  std::vector<double> pos1 = p1->getPosition();
  std::vector<double> pos2 = p2->getPosition();
  double dist = computeDistance(pos1, pos2);
  if (dist <= m_bosonDiameter) {
    std::cout << "correlated crash" << '\n';
    return 0;
  } else {
    return (1 - m_bosonDiameter/dist);
  }
} // end correlatedPart

double Correlated::dotProduct(std::vector<double> v1, std::vector<double> v2) {
  //std::cout << "Class Correlated Memeber dotProduct" << '\n';
  double sum = 0;
  for (int i = 0; i < v1.size(); i++) {
    sum += v1.at(i)*v2.at(i);
  }
  return sum;
}


double Correlated::computeDoubleDerivative(std::vector<class Particle*> particles, bool DDeriv_input, int k) {
  // std::cout << "Class Correlated Member computeDoubleDerivative" << '\n';
  //
  // Here we compute the double derivative Laplician / WaveFunction.
  // We devied the calculation in four terms.
  // Term1 is nabla_k^2 phi/phi
  // Term2 is 2nabla_k phi/phi * sum(nabla_k u(r_kl))
  // Term3 is sum(nabla_k u(r_kl)) * sum(nabla_k u(r_kl))
  // Term4 is sum(nabla_k^2 u(r_kl))


  int dim = m_numberOfDimensions;
  int N   = m_system->getNumberOfParticles();
  double a = m_bosonDiameter;

  std::vector<double> r_k, r_l;
  r_k = particles.at(k)->getPosition();
  double x_k, x_k2, y_k, y_k2, z_k, z_k2;
  x_k = r_k[0]; y_k = r_k[1]; z_k = r_k[2];
  x_k2 = x_k*x_k; y_k2 = y_k*y_k; z_k2 = z_k*z_k;

  double first_term, second_term, third_term, fourth_term;
  first_term = 0; second_term = 0; third_term = 0; fourth_term = 0;

  first_term =  4*m_alpha*m_alpha*(x_k2 + y_k2 + m_beta*m_beta*z_k2);
  first_term -= 2*m_alpha*(2 + m_beta);

  double Laplician, uPrimeOverR, uPrimePrime, r_kl;
  std::vector<double> nablaPhi(3 , -4*m_alpha), uGrad(3, 0);
  nablaPhi[0] *= x_k; nablaPhi[1] *= y_k; nablaPhi[2] *= m_beta*z_k;

  for (int l = 0; l < N; l++) {
    if(l == k) { continue; }
    r_l  = particles.at(l)->getPosition();
  	r_kl = computeDistance(r_k, r_l);
    // r_kl = computeDistance(particles[k], particles[l]);
  	uPrimeOverR = a/(r_kl*r_kl*(r_kl - a));
  	uGrad[0] += uPrimeOverR*( r_k[0] - r_l[0] );
  	uGrad[1] += uPrimeOverR*( r_k[1] - r_l[1] );
  	uGrad[2] += uPrimeOverR*( r_k[2] - r_l[2] );
  	uPrimePrime = (a*a - 2*a*r_kl)/(r_kl*r_kl*(r_kl-a)*(r_kl-a));
  	fourth_term += uPrimePrime + 2*uPrimeOverR;
  }

  second_term = dotProduct(nablaPhi, uGrad);
  third_term  = dotProduct(uGrad, uGrad);
  Laplician   = first_term + second_term + third_term + fourth_term;
  return Laplician;
} // end computeDoubleDerivative

std::vector<double> Correlated::computeQuantumForce(Particle* particle, int k) {
  // std::cout << "Class Correlated Member computeQuantumForce" << '\n';
  int N   = m_system->getNumberOfParticles();
  double a = m_bosonDiameter;

  std::vector<double> position = particle->getPosition();
  std::vector<double> qForce, r_k, r_l;
  qForce.push_back(-4*m_alpha*position[0]);
  qForce.push_back(-4*m_alpha*position[1]);
  qForce.push_back(-4*m_alpha*m_beta*position[2]);

  std::vector<class Particle*> particles;
  particles = m_system->getParticles();

  double uPrimeOverR, r_kl;
  r_k = position;

  for (int l = 0; l < N; l++) {
    if(l == k) { continue; }
    r_l  = particles.at(l)->getPosition();
    r_kl = computeDistance(r_k, r_l); // abs(r_k - r_l) or sqrt{ (r_k - r_l)^2 }
  	uPrimeOverR = a/(r_kl*r_kl*(r_kl - a));
  	qForce[0] += 2*uPrimeOverR*( r_k[0] - r_l[0] );
  	qForce[1] += 2*uPrimeOverR*( r_k[1] - r_l[1] );
  	qForce[2] += 2*uPrimeOverR*( r_k[2] - r_l[2] );
  }

  return qForce;
}

double Correlated::evaluateWFDeriv(std::vector<class Particle*> particles) {
  // std::cout << "Class Correlated Member evaluateWFDeriv" << '\n';
  double positionSum = 0;
  std::vector<double> position;
  for (auto particle : particles) {
    position = particle->getPosition();
    positionSum += position[0]*position[0] + position[1]*position[1];
    positionSum += m_beta*position[2]*position[2];
  }
  return -positionSum;
}

double Correlated::energyGradient() {
  //std::cout << "Class Correlated Member energyGradient" << '\n';
  double term_1 = m_system->getSampler()->getexpectWFDerivXEnergy();
  double term_2 = m_system->getSampler()->getexpectWFDerivXexpectEnergy();
  return 2*(term_1 - term_2);
}
