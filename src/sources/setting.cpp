#include "setting.hpp"

Setting::Setting(const string& fileName) {
  set<string> sections;
  map<string, vector<string>> dataFile;

  sections.insert("GRID");  sections.insert("INITIAL_CONDITIONS");
  sections.insert("BOUNDARY_CONDITIONS");
  sections.insert("FLUX_SPLITTER");  sections.insert("TIME");  sections.insert("SYSTEM");
  sections.insert("PHYSICAL_VALUES");  sections.insert("SAVING");
  sections.insert("ACCURACY");

  loadDataFile(fileName, sections, dataFile);

  // nacitani informaci o siti
  string section = "GRID";
  findSection(dataFile, "grid_type", section, grid_type);
  switch (grid_type) {
  case 1:
    findSection(dataFile, "mCells", section, mCells);
    findSection(dataFile, "nCells", section, nCells);
    break;
  case 2:
    findSection(dataFile, "name1", section, name1);
    findSection(dataFile, "name2", section, name2);
    break;
  default:
    cout << "Nepodporovany typ site!" << endl;
    cout << "Possibilities are: 1 for grid type GAMM, 2 for a grid type XY!" << endl;
    exit(51);
  }
  findSection(dataFile, "ghostCells", section, ghostCells);

  // nacitani informaci o pocatecnich podminkach
  section = "INITIAL_CONDITIONS";
  findSection(dataFile, "rhoInit", section, rhoInit);
  findSection(dataFile, "pInit", section, pInit);
  double x, y;
  findSection(dataFile, "uInit", section, x);
  findSection(dataFile, "vInit", section, y);
  uInit = Vector2d(x, y);

  // nacitani informaci o okrajovych podminkach
  section = "BOUNDARY_CONDITIONS";
  findSection(dataFile, "numOfBoundaries", section, numOfBoundaries);

  for (int i=1; i<=numOfBoundaries; i++) {
    stringstream part;
    string stringPart;

    part << i;
    part >> stringPart;

    string boundary = "boundary" + stringPart;
    string bcType = "bcType" + stringPart;

    string boundaryValue, bcTypeValue;

    findSection(dataFile, boundary, section, boundaryValue);
    findSection(dataFile, bcType, section, bcTypeValue);

    usedBC[boundaryValue] = bcTypeValue;
  }

  findSection(dataFile, "alpha", section, alpha);
  findSection(dataFile, "M2is", section, Ma2is);

  // nacitani informaci o numerickem toku
  section = "FLUX_SPLITTER";
  findSection(dataFile, "flux", section, flux);

  // nacitani informaci o numerickem toku
  section = "ACCURACY";
  findSection(dataFile, "spatialOrder", section, spatialOrder);
  if (spatialOrder == 2)
    findSection(dataFile, "limiter", section, limiter);

  findSection(dataFile, "temporalOrder", section, temporalOrder);
  switch (temporalOrder) {
  case 1:
    alphaK.resize(1);
    alphaK[0] = 1.;
    break;
  case 2:
    alphaK.resize(3);
    alphaK[0] = 0.5;  alphaK[1] = 0.5;  alphaK[2] = 1.;
    break;
  default:
    cout << "No a such possibility for a temporal order!" << endl;
    cout << "Possibilities are: 1 - 1st order, 2 - 2nd order" << endl;
    exit(63);
  }

  // nacitani informaci o case
  section = "TIME";
  findSection(dataFile, "CFL", section, CFLmax);
  findSection(dataFile, "CFLbegin", section, CFL);
  findSection(dataFile, "incrementIts", section, incrementIts);

  if (incrementIts < 1) {
    cout << "CFL incerement interations must be a poitive number!" << endl;
    cout << "Variable \"incrementIts\" is set on 1!" << endl;

    incrementIts = 1.;
  }

  incCoeff = pow(CFLmax/CFL, 1./incrementIts);

  // nacitani informaci o systemu
  section = "SYSTEM";
  findSection(dataFile, "solver", section, solver);
  findSection(dataFile, "convection", section, convection);
  findSection(dataFile, "diffusion", section, diffusion);
  findSection(dataFile, "nodeWeightType", section, nodeWeightType);

  // nacitani informaci o fyzikalnich promennych
  section = "PHYSICAL_VALUES";
  findSection(dataFile, "kappa", section, kappa);
  findSection(dataFile, "rho0", section, rho0);
  findSection(dataFile, "p0", section, p0);
  findSection(dataFile, "R", section, R);
  findSection(dataFile, "Pr", section, Pr);

  // nacitani informaci o ukonceni programu
  section = "SAVING";
  findSection(dataFile, "stop", section, stop);
}

void Setting::updateCFL() {
  CFL *= incCoeff;
  CFL = min(CFL, CFLmax);
}
