/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>
#include <cassert>
#include <array>
#include "particle_filter.h"
#include <iomanip>

using namespace std;

bool DEBUG = false;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Initialize random generator
  std::random_device rd;
  std::mt19937 gen(rd());

  // Normal distributions for x, y and theta
  std::normal_distribution<double> xrand(x    ,std[0]);
  std::normal_distribution<double> yrand(y    ,std[1]);
  std::normal_distribution<double> trand(theta,std[2]);

  num_particles = 50;
  for (int i = 0; i< num_particles; ++i) {
    Particle P;
    P.id     = i;
    P.x      = xrand(gen);
    P.y      = yrand(gen);
    P.theta  = trand(gen);
    P.weight = 1.0;
    
    particles.push_back(P);
    weights.push_back(1.0);

    if (DEBUG) {
      std::cout << "P# " << std::setw(4) << P.id;
      std::cout << " X " << std::setw(12) << P.x;
      std::cout << " Y " << std::setw(12) << P.y;
      std::cout << " T " << std::setw(12) << P.theta;
      std::cout << " W " << std::setw(12) << P.weight << std::endl;
    }
    
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Initialize random generator
  std::random_device rd;
  std::mt19937 gen(rd());

  // Normal distributions for x, y and theta
  std::normal_distribution<> xrand(0, std_pos[0]);
  std::normal_distribution<> yrand(0, std_pos[1]);
  std::normal_distribution<> trand(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    // Last position
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double t0 = particles[i].theta;

    // Update position
    if (std::fabs(yaw_rate) > 0.0001) {
      particles[i].x     = x0 + velocity / yaw_rate * (+ sin(t0 + yaw_rate*delta_t) - sin(t0)) + xrand(gen);
      particles[i].y     = y0 + velocity / yaw_rate * (- cos(t0 + yaw_rate*delta_t) + cos(t0)) + yrand(gen);
      particles[i].theta = t0 + yaw_rate*delta_t + trand(gen);
    }
    else {      
      particles[i].x     = x0 + velocity * delta_t * cos(t0) + xrand(gen);
      particles[i].y     = y0 + velocity * delta_t * sin(t0) + yrand(gen);
      particles[i].theta = t0 + trand(gen);
    }

    if (DEBUG) {
      std::cout << "Prediction P# " << setw(4) << i;
      std::cout << " x = " << setw(12) << particles[i].x;
      std::cout << " y = " << setw(12) << particles[i].y;
      std::cout << " t = " << setw(12) << particles[i].theta << std::endl;
    }
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.


  for (int i = 0; i < observations.size(); ++i) {
    double x1 = observations[i].x;
    double y1 = observations[i].y;
    int   id1 = observations[i].id;

    // Initialize value with a large number
    double min_dist = DBL_MAX;
    int min_ij = -1;
  
    for (int j = 0; j < predicted.size(); ++j) {
      double x2 = predicted[j].x;
      double y2 = predicted[j].y;
      int   id2 = predicted[j].id;
     
      double cur_dist = dist(x1, y1, x2, y2);

      // Update
      if (cur_dist < min_dist) {
	min_dist = cur_dist;
	min_ij = id2;
      }
    }

    // Update association vector for observations
    observations[i].id = min_ij;

    if (DEBUG) {
      std::cout << "DATA-ASSOC: " << setw(4) << i;
      std::cout << " MATCH = " << setw(4) << min_ij;
      std::cout << " DIST  = " << setw(12) << min_dist << std::endl;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
				   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  // Update weight for each particle
  for (int i = 0; i < num_particles; ++i) {

    // Assume the particle's location and orientation
    double xi = particles[i].x;
    double yi = particles[i].y;
    double ti = particles[i].theta;

    // Initialize vector to hold predictions
    std::vector<LandmarkObs> predictions;
    int num_predictions = 0;

    // Obtain all the Map landmarks within the sensor range
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      int   idj = map_landmarks.landmark_list[j].id_i;
      double xj = map_landmarks.landmark_list[j].x_f;
      double yj = map_landmarks.landmark_list[j].y_f;

      // Subselect only those landmarks within sensor range from particle i and add to predictions
      if (std::fabs(xj-xi) <= 2*sensor_range && std::fabs(yi-yj) <= 2*sensor_range) {
	predictions.push_back(LandmarkObs{idj, xj, yj});
	num_predictions++;
      }
    }
    if (num_predictions <= 0) {
      std::cout << " Num predictions = 0!!" << std::endl;
      return;
    }
    
    // Now transform observations to MAP coordinate system and...
    double ct = std::cos(ti);
    double st = std::sin(ti);

    std::vector<LandmarkObs> observations_t;
    
    for (int k = 0; k < observations.size(); ++k) {
      double xk = observations[k].x * ct - observations[k].y * st + xi;
      double yk = observations[k].x * st + observations[k].y * ct + yi;
      observations_t.push_back(LandmarkObs{observations[k].id, xk, yk});
    }

    // ...associate with landmark predictions
    dataAssociation(predictions, observations_t);

    // **********************************************************************
    // Now let's update the weights from multivariate normal distribution
    // **********************************************************************
    // Re-Initialize weight
    double w = 1.0;

    // Co-variance of measurements
    double sx = std_landmark[0]; // Is this actually in terms of range???
    double sy = std_landmark[1]; // Is this actually in terms of bearing???
    double factor = 1.0/(2*M_PI*sx*sy);  // Common multiplicative factor for weight
    
    // Loop over observations - measurement x_i
    for (int k = 0; k < observations_t.size(); ++k) {
      double x_meas = observations_t[k].x;
      double y_meas = observations_t[k].y;
      int   id_meas = observations_t[k].id;
      
      // Locate appropriate predicted measurement - mu_i
      bool found = false;
      int j = 0;
      double x_obs = predictions[j].x;
      double y_obs = predictions[j].y;

      while (!found && j<num_predictions) {
	if (predictions[j].id == id_meas) {
	  found = true;
	  x_obs = predictions[j].x;
	  y_obs = predictions[j].y;
	}
	j++;
      }

      if (!found) std::cout << "NOT FOUND!!" << std::endl;

      //assert(found==true);

      /*
      R = [ cos(theta)  -sin(theta) ]
          [ sin(theta)   cos(theta) ]
      
      S = [ sx  0 ]
          [ 0  sy ]
      
      Rt= [ cos(theta)   sin(theta) ]
          [-sin(theta)   cos(theta) ]

      S*Rt= [ sx*ct  sx*st ]
            [-sy*st  sy*ct ]

      R*S*Rt =  [ sx*ct  sx*st ]  \/  [ ct  -st ]
                [-sy*st  sy*ct ]  /\  [ st   ct ]

           ==> [ sx*(ct^2 + st^2)      -sx*ct*st + sx*ct*st]
               [-sy*ct*st + sy*ct*st    sy*(ct^2 + st^2)   ]

           ==> [ sx  0 ]
               [ 0  sy ]

      Polar to cartesian transformation does not affect covariance matrix!
      */   
      
      // Update weight (using bivariate gaussian normal distribution with cross-correlation rho = 0)
      w *= factor*exp(-((x_meas-x_obs)*(x_meas-x_obs)/(2*sx*sx) + (y_meas-y_obs)*(y_meas-y_obs)/(2*sy*sy)));
      
      if (DEBUG) {
	std::cout << " w update " << std::setw(4) << k;
	std::cout << " to " << std::setw(12) << w;
	std::cout << " fac " << std::setw(12) << factor;
	std::cout << " sx  " << std::setw(12) << sx;
	std::cout << " sx  " << std::setw(12) << sy;	
	std::cout << " dx " << std::setw(12) << x_meas-x_obs;
	std::cout << " dy " << std::setw(12) << y_meas-y_obs;
	std::cout << std::endl;
      }
    }
    particles[i].weight = w;
    weights[i] = w;

    if (DEBUG) {
      std::cout << " FINAL W = " << std::setw(12) << w << std::endl;
    }
  }
}  

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::random_device rd;
  std::mt19937 gen(rd());
    
  std::discrete_distribution<int> ddist(weights.begin(), weights.end());

  std::vector<Particle> old_particles = particles;  
  for (int i = 0; i < num_particles; ++i) {
    int id_sel = ddist(gen);
    particles[i].x = old_particles[id_sel].x;
    particles[i].y = old_particles[id_sel].y;
    particles[i].theta = old_particles[id_sel].theta;
    particles[i].weight = old_particles[id_sel].weight;
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
