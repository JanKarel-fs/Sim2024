#ifndef GRID_HPP
#define GRID_HPP

#include <string>
#include <vector>
#include <map>
#include <utility>
#include "../geometry/point.hpp"
#include "../geometry/vector.hpp"
#include "../geometry/field.hpp"

using namespace std;

class Grid;

typedef void (*coefficients)(Grid& g);

class Node {
public:
  Point2d vertex;
  vector<double> alpha;
};

class Face {
public:
  Point2d center;  // stred steny
  Vector2d s;      // normalovy vektor prenasobeny velikosti steny
  string name;

  void updateI(Grid& g, int i, int j);
  void updateJ(Grid& g, int i, int j);
};

class Grid {
protected:
  int Mnodes;    // pocet uzlu ve smeru i
  int Nnodes;    // pocet uzlu ve smeru j
  int Mvolumes;  // pocet bunek ve smeru i
  int Nvolumes;  // pocet bunek ve smeru j
  int ghost;     // pocet vrstev pomocnych bunek

  Field2<Node> nodes;
  Field2<Face> facesI;
  Field2<Face> facesJ;
  Field2<Point2d> centers;  // pole stredu bunek
  Field2<double> volumes;   // pole objemu bunek
  
public:
  Grid() {};
  virtual ~Grid() {};

  double x(const int& i, const int& j) const;
  double y(const int& i, const int& j) const;
  Point2d& vertex(const int& i, const int& j) const;
  vector<double>& alpha(const int& i, const int& j) const;
  Node& node(const int& i, const int& j) const;
  Face& faceI(const int& i, const int& j) const;
  Face& faceJ(const int& i, const int& j) const;
  Point2d& center(const int& i, const int& j) const;
  double volume(const int& i, const int& j) const;

  void update();

  int Mnd() const {return Mnodes;};
  int Nnd() const {return Nnodes;};
  int Mvol() const {return Mvolumes;};
  int Nvol() const {return Nvolumes;};
  int gh() const {return ghost;};

  void updateVolumes();
  void updateCenters();

  static void (*computeAlpha)(Grid& g);
  static void computeAlphaWeight(Grid& g);
  static void computeAlphaLSM(Grid& g);

  static map<string, coefficients> coeffList;
};  

#endif
