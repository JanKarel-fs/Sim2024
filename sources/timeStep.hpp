#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP

#include <cmath>
#include "../fvm/cellfield.hpp"
#include "../fvm/grid.hpp"
#include "../compressible.hpp"
#include "setting.hpp"

using namespace std;

double timeStep(const CellField<Compressible>& w, const Grid& g, const Setting& setting);

#endif
