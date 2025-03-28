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

