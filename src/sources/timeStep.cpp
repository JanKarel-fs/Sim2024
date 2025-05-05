#include "timeStep.hpp"

double timeStep(const CellField<Compressible>& w, const Grid& g, Setting& setting) {
  double dt = 1e5;

  int M = w.M();
  int N = w.N();

# pragma omp parallel for reduction(min:dt)
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      const Compressible& wij = w[i][j];

      Point2d A = g.faceJ(i, j).center;
      Point2d B = g.faceJ(i+1, j).center;
      Point2d C = g.faceI(i, j).center;
      Point2d D = g.faceI(i, j+1).center;

      Vector2d s1(A, B);
      Vector2d s2(C, D);

      double dX = s1.length();
      double dY = s2.length();
      
      double lambda = 0.;

      switch (setting.convection) {
      case 0: break;
      case 1: {
	Vector2d u = wij.rhoU / wij.rho;
	double a = wij.a();
	double uTilde = dot(u, s1) / dX;
	double vTilde = dot(u, s2) / dY;
	double lambdaConv = (fabs(uTilde) + a) / dX + (fabs(vTilde) + a) / dY;
	lambda += lambdaConv;
      }
	break;
      default:
	cout << "No such possibility for a computation of a convective part of a time step!" << endl;
	exit(10);
      }

      switch (setting.diffusion) {
      case 0: break;
      case 1: {
	double mu = wij.mu();
	double rho = wij.rho;
	double lambdaDiff = 2. * mu/rho * (1. / (dX*dX) + 1. / (dY*dY));
	lambda += lambdaDiff;
      }
	break;
      default:
	cout << "No such possibility for a computation of a dissipative part of a time step!" << endl;
	exit(10);
      }

      double dtij = setting.CFL / lambda;

      dt = min(dt, dtij);
    }
  }

  setting.updateCFL();

  return dt;
}
