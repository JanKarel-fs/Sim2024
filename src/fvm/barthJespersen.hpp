#ifndef BARTHJESPERSEN_HPP
#define BARTHJESPERSEN_HPP

#include <vector>
#include <omp.h>
#include "cellfield.hpp"
#include "grid.hpp"

using namespace std;

template <typename var>
void barthJespersen(const CellField<var>& w, const CellField<Vector2<var> >& gradW,
		    CellField<var>& psi, const Grid& g) {

  #pragma omp parallel for
  for (int i=w.Imin(); i<w.Imax(); i++) {
    for (int j=w.Jmin(); j<w.Jmax(); j++) {
      psi[i][j].zero();
    }
  }

  #pragma omp parallel for
  for (int i=0; i<w.M(); i++) {
    for (int j=0; j<w.N(); j++) {

      const var& wij = w[i][j];
      const Point2d& center = g.center(i, j);

      vector<var> wk(4);
      vector<Point2d> centers_k(4);

      wk[0] = w[i-1][j];  wk[1] = w[i+1][j];
      wk[2] = w[i][j-1];  wk[3] = w[i][j+1];

      centers_k[0] = g.faceJ(i,j).center;  centers_k[1] = g.faceJ(i+1, j).center;
      centers_k[2] = g.faceI(i,j).center;  centers_k[3] = g.faceI(i, j+1).center;

      var wMax = wij;
      var wMin = wij;

      for (int k=0; k<4; k++) {
	wMax = var::max(wMax, wk[k]);
	wMin = var::min(wMin, wk[k]);
      }

      for (int m=0; m<var::nVars; m++) {
	psi[i][j][m] = 1.;

	for (int k=0; k<4; k++) {
	  Vector2d r(g.center(i,j), centers_k[k]);
	  double delta2 = gradW[i][j].x[m] * r.x + gradW[i][j].y[m] * r.y;

	  double psiM = 1.;

	  if (delta2 > 0.) {
	    psiM = min(1., (wMax[m] - wij[m])/delta2);
	  }
	  else if (delta2 < 0.) {
	    psiM = min(1., (wMin[m] - wij[m])/delta2);
	  }

	  psi[i][j][m] = min(psi[i][j][m], psiM);
	}
      }
    }
  }
}

#endif
