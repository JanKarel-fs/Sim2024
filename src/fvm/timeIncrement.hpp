#ifndef TIMEINCREMENT_HPP
#define TIMEINCREMENT_HPP

#include "sources/linearSolver.hpp"

template <typename var>
void (*timeIncrement)(LinearSolver<var>& linSolver, const CellField<var>& w,
		      const CellField<var>& wOld, const double& dt);

#endif
