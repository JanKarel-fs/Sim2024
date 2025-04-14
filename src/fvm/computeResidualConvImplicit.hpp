#ifndef COMPUTERESIDUALCONVIMPLICIT_HPP
#define COMPUTERESIDUALCONVIMPLICIT_HPP

#include "grid.hpp"
#include "cellfield.hpp"
#include "grad.hpp"
#include "limiter.hpp"
#include "../geometry/vector.hpp"
#include "../geometry/matrix.hpp"
#include "sources/typedefs.hpp"
#include "sources/linearSolver.hpp"
#include <omp.h>

template <typename var>
void computeResidualConvImplicit(const CellField<var>& w, const Grid& g,
				 const map<string, bcWithJacobian>& BC,
				 LinearSolver<var>& linSolver, const Setting& setting) {

  CellField<Vector2<var> > gradW(g);
  CellField<var> psi(g);
  grad<var>(w, gradW, g);
  limiter<var>(w, gradW, psi, g);
  
  // vypocet toku stenami ve smeru i
#pragma omp parallel for
  for (int i=0; i<w.M(); i++) {
    for (int j=0; j<w.N()+1; j++) {
      const Face& f = g.faceI(i, j);

      Vector2d rL(g.center(i, j), f.center);
      Vector2d rR(g.center(i, j-1), f.center);
      
      var wl = w[i][j] + psi[i][j] * (gradW[i][j].x * rL.x + gradW[i][j].y * rL.y);
      var wr = w[i][j-1] + psi[i][j-1] * (gradW[i][j-1].x * rR.x + gradW[i][j-1].y * rR.y);

      pair<pair<Matrixd, Matrixd>, var> increment = var::fluxImplicit(wl, wr, f.s);

      int leftOffset = var::nVars * (j*w.M() + i);
      int rightOffset = var::nVars * ((j-1)*w.M() + i);

      double leftVolume = g.volume(i, j);
      double rightVolume = g.volume(i, j-1);

      const var& flx = increment.second;
      pair<Matrixd, Matrixd>& Jacobians = increment.first;

      if (j==0 || j==w.N()) {
	var wb;
	int offset;
	double coeff, volume;
	if (j==0) {
	  wb = wl;
	  offset = leftOffset;
	  coeff = -1.;
	  volume = leftVolume;
	}
	else {
	  wb = wr;
	  offset = rightOffset;
	  coeff = 1.;
	  volume = rightVolume;
	}
	auto it = BC.find(f.name);
	Matrixd BJacobian = it->second.second(wb, f.s, setting);
	Jacobians.second = Jacobians.second * BJacobian;

	int row[var::nVars], col[var::nVars];
	double values[var::nVars*var::nVars];
	
	for (int k=0; k<var::nVars; k++) {
	  linSolver.rhs[ offset + k] += coeff*flx[k] / volume;
	  row[k] = offset + k;
	  col[k] = offset + k;
	  for (int m=0; m<var::nVars; m++) {
	    values[k*var::nVars + m] = -coeff*Jacobians.first[k][m] / volume;
	    values[k*var::nVars + m] += -coeff*Jacobians.second[k][m] / volume;
	  }
	}
	MatSetValues(linSolver.A, var::nVars, row, var::nVars, col, values, ADD_VALUES);
      }
      else {
	int rowL[var::nVars], colL[var::nVars], rowR[var::nVars], colR[var::nVars];
	double valuesLL[var::nVars*var::nVars], valuesLR[var::nVars*var::nVars];
	double valuesRL[var::nVars*var::nVars], valuesRR[var::nVars*var::nVars];
	
	for (int k=0; k<var::nVars; k++) {
	  linSolver.rhs[leftOffset + k] += -1.*flx[k] / leftVolume;
	  linSolver.rhs[rightOffset + k] += flx[k] / rightVolume;
	  rowL[k] = leftOffset + k;
	  colL[k] = leftOffset + k;
	  rowR[k] = rightOffset + k;
	  colR[k] = rightOffset + k;
	  for (int m=0; m<var::nVars; m++) {
	    valuesLL[k*var::nVars + m] = Jacobians.first[k][m] / leftVolume;
	    valuesLR[k*var::nVars + m] = Jacobians.second[k][m] / leftVolume;

	    valuesRL[k*var::nVars + m] = -1.*Jacobians.first[k][m] / rightVolume;
	    valuesRR[k*var::nVars + m] = -1.*Jacobians.second[k][m] / rightVolume;
	  }
	}

	MatSetValues(linSolver.A, var::nVars, rowL, var::nVars, colL, valuesLL, ADD_VALUES);
	MatSetValues(linSolver.A, var::nVars, rowL, var::nVars, colR, valuesLR, ADD_VALUES);
	MatSetValues(linSolver.A, var::nVars, rowR, var::nVars, colL, valuesRL, ADD_VALUES);
	MatSetValues(linSolver.A, var::nVars, rowR, var::nVars, colR, valuesRR, ADD_VALUES);
      }
    }
  }

  // vypocet toku stenami ve smeru j
#pragma omp parallel for
  for (int j=0; j<w.N(); j++) {
    for (int i=0; i<w.M()+1; i++) {
      const Face& f = g.faceJ(i, j);

      Vector2d rL(g.center(i-1, j), f.center);
      Vector2d rR(g.center(i, j), f.center);
      
      var wl = w[i-1][j] + psi[i-1][j] * (gradW[i-1][j].x * rL.x + gradW[i-1][j].y * rL.y);
      var wr = w[i][j] + psi[i][j] * (gradW[i][j].x * rR.x + gradW[i][j].y * rR.y);
      
      pair<pair<Matrixd, Matrixd>, var> increment = var::fluxImplicit(wl, wr, f.s);

      int leftOffset = var::nVars * (j*w.M() + i-1);
      int rightOffset = var::nVars * (j*w.M() + i);

      double leftVolume = g.volume(i-1, j);
      double rightVolume = g.volume(i, j);

      const var& flx = increment.second;
      pair<Matrixd, Matrixd>& Jacobians = increment.first;

      if (i==0 || i==w.M()) {
	var wb;
	int offset;
	double coeff, volume;
	if (i==0) {
	  wb = wr;
	  offset = rightOffset;
	  coeff = 1.;
	  volume = rightVolume;
	}
	else {
	  wb = wl;
	  offset = leftOffset;
	  coeff = -1.;
	  volume = leftVolume;
	}
	auto it = BC.find(f.name);
	Matrixd BJacobian = it->second.second(wb, f.s, setting);
	Jacobians.second = Jacobians.second * BJacobian;

	int row[var::nVars], col[var::nVars];
	double values[var::nVars*var::nVars];
	
	for (int k=0; k<var::nVars; k++) {
	  linSolver.rhs[offset + k] += coeff*flx[k] / volume;
	  row[k] = offset + k;
	  col[k] = offset + k;
	  for (int m=0; m<var::nVars; m++) {
	    values[k*var::nVars + m] = -coeff*Jacobians.first[k][m] / volume;
	    values[k*var::nVars + m] += -coeff*Jacobians.second[k][m] / volume;
	  }
	}
	MatSetValues(linSolver.A, var::nVars, row, var::nVars, col, values, ADD_VALUES);
      }
      else {
	int rowL[var::nVars], colL[var::nVars], rowR[var::nVars], colR[var::nVars];
	double valuesLL[var::nVars*var::nVars], valuesLR[var::nVars*var::nVars];
	double valuesRL[var::nVars*var::nVars], valuesRR[var::nVars*var::nVars];
	
	for (int k=0; k<var::nVars; k++) {
	  linSolver.rhs[leftOffset + k] += -1.*flx[k] / leftVolume;
	  linSolver.rhs[rightOffset + k] += flx[k] / rightVolume;
	  rowL[k] = leftOffset + k;
	  colL[k] = leftOffset + k;
	  rowR[k] = rightOffset + k;
	  colR[k] = rightOffset + k;
	  for (int m=0; m<var::nVars; m++) {
	    valuesLL[k*var::nVars + m] = Jacobians.first[k][m] / leftVolume;
	    valuesLR[k*var::nVars + m] = Jacobians.second[k][m] / leftVolume;
	    
	    valuesRL[k*var::nVars + m] = -1.*Jacobians.first[k][m] / rightVolume;
	    valuesRR[k*var::nVars + m] = -1.*Jacobians.second[k][m] / rightVolume;
	  }
	}

	MatSetValues(linSolver.A, var::nVars, rowL, var::nVars, colL, valuesLL, ADD_VALUES);
	MatSetValues(linSolver.A, var::nVars, rowL, var::nVars, colR, valuesLR, ADD_VALUES);
	MatSetValues(linSolver.A, var::nVars, rowR, var::nVars, colL, valuesRL, ADD_VALUES);
	MatSetValues(linSolver.A, var::nVars, rowR, var::nVars, colR, valuesRR, ADD_VALUES);
      }
    }
  }
}

#endif
