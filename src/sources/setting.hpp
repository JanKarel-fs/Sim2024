#ifndef SETTING_HPP
#define SETTING_HPP

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include "../geometry/vector.hpp"
#include "loadDataFile.hpp"
#include "findSection.hpp"

using namespace std;

class Setting {
public:
  int grid_type;
  int mCells;
  int nCells;
  int ghostCells;
  double rhoInit;
  double pInit;
  Vector2d uInit;
  int numOfBoundaries;
  map<string, string> usedBC;
  int flux;
  double p0;
  double rho0;
  double alpha;
  double Ma2is;
  double CFL;
  double kappa;
  int stop;
  int temporalOrder;
  int spatialOrder;
  int limiter;
  vector<double> alphaK;

  Setting(const string& fileName);
  ~Setting() {};
};


#endif
