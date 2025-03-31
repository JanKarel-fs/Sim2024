#include "initialisation.hpp"

void initialisation(CellField<Compressible>& w, const Setting& setting) {
  system("mkdir -p results");
  system("rm -f results/*");

  Compressible::kappa = setting.kappa;

  switch (setting.flux) {
  case 1:
    Compressible::flux = Compressible::Upwind;
    break;
  default:
    cout << "Not a such numerical flux!" << endl;
    cout << "Use 1 - Upwind" << endl;
    exit(53);
  }

  switch(setting.spatialOrder) {
  case 1:
    grad<Compressible> = zeroGrad<Compressible>;
    limiter<Compressible> = zeroLimiter<Compressible>;
    break;
  case 2:
    grad<Compressible> = gradLSM<Compressible>;
    switch(setting.limiter) {
    case 1:
      limiter<Compressible> = barthJespersen<Compressible>;
      break;
    case 2:
      limiter<Compressible> = venkatakrishnan<Compressible>;
      break;
    default:
      cout << "No a such possibility for a limiter!" << endl;
      cout << "Possibilities are: 1 - Barth-Jespersen, 2 - Venkatakrishnan" << endl;
      exit(62);
    }
    break;
  default:
    cout << "No a such possibility for a spatial order!" << endl;
    cout << "Possibilities are: 1 - 1st order, 2 - 2nd order" << endl;
    exit(11);
  }

  const double& rhoInit = setting.rhoInit;
  const Vector2d& uInit = setting.uInit;
  const double& pInit = setting.pInit;

  double eInit = pInit / (Compressible::kappa - 1.)
               + 0.5 * rhoInit * (uInit.x*uInit.x + uInit.y*uInit.y);

  for (int i=0; i<w.M(); i++) {
    for (int j=0; j<w.N(); j++) {
      w[i][j] = Compressible(rhoInit, rhoInit*uInit, eInit);
    }
  }
}

