#ifndef COMPRESSIBLE_HPP
#define COMPRESSIBLE_HPP

#include <iostream>
#include <cmath>
#include <algorithm>
#include "geometry/vector.hpp"
#include "geometry/matrix.hpp"

#include "primitiveVars.hpp"

using namespace std;

class PrimitiveVars;

class Compressible {
public:
  double rho;
  Vector2d rhoU;
  double e;

  Compressible() {};
  Compressible(const double& rho_, const Vector2d& rhoU_, const double& e_):
    rho(rho_), rhoU(rhoU_), e(e_) {};
  ~Compressible() {};

  void zero() {
    rho = 0.; rhoU = Vector2d(0., 0.), e = 0.;
  }

  void ones() {
    rho = 1.; rhoU = Vector2d(1., 1.), e = 1.;
  }

  double p() const;
  double a() const;
  double Ma() const;
  double T() const;
  double mu() const;
  double k() const;

  static Compressible max(const Compressible& a, const Compressible& b);
  static Compressible min(const Compressible& a, const Compressible& b);
  static Compressible fabs(const Compressible& a);
  static Compressible sqrt(const Compressible& a);

  static double kappa;
  static double R;     // merna plynova konstanta (nekorektni znaceni)
  static double cp;
  static double cv;
  static double Pr;

  static Compressible (*flux)(const Compressible& wl, const Compressible& wr, const Vector2d& s);
  static Compressible Upwind(const Compressible& wl, const Compressible& wr, const Vector2d& s);
  static Compressible Rusanov(const Compressible& wl, const Compressible& wr, const Vector2d& s);
  static Compressible fluxDissipative(const Vector2<PrimitiveVars>& gradP, const Compressible& wFace, const Vector2d& s);

  static pair<pair<Matrixd, Matrixd>, Compressible> (*fluxImplicit)(const Compressible& wl,
						      const Compressible& wr, const Vector2d& s);
  static pair<pair<Matrixd, Matrixd>, Compressible> UpwindImplicit(const Compressible& wl,
						      const Compressible& wr, const Vector2d& s);
  static pair<pair<Matrixd, Matrixd>, Compressible> RusanovImplicit(const Compressible& wl,
						      const Compressible& wr, const Vector2d& s);

  static pair<pair<Matrixd, Matrixd>, Compressible> fluxDissipativeImplicit(const Compressible& wl,
					 const Compressible& wr, const Point2d& L, const Point2d& R,
					 const Vector2<PrimitiveVars>& gradP, const Vector2d& s);

  static const int nVars = 4;

  double& operator[](int k) {
    switch(k) {
    case 0: return rho;
      break;
    case 1: return rhoU.x;
      break;
    case 2: return rhoU.y;
      break;
    case 3: return e;
      break;
    default: cout << "Compressible operator []: out of a range!" << endl;
      exit(61);
    }
  }

  double operator[](int k) const {
    switch(k) {
    case 0: return rho;
      break;
    case 1: return rhoU.x;
      break;
    case 2: return rhoU.y;
      break;
    case 3: return e;
      break;
    default: cout << "Compressible operator []: out of a range!" << endl;
      exit(61);
    }
  }
};

inline Compressible operator+(const Compressible& a, const Compressible& b) {
  return Compressible(a.rho+b.rho, a.rhoU+b.rhoU, a.e+b.e);
}

inline Compressible operator-(const Compressible& a, const Compressible& b) {
  return Compressible(a.rho-b.rho, a.rhoU-b.rhoU, a.e-b.e);
}

inline Compressible operator*(const Compressible& a, const Compressible& b) {
  return Compressible(a.rho*b.rho, a.rhoU*b.rhoU, a.e*b.e);
}

template <typename S>
inline Compressible operator*(const Compressible& a, const S& b) {
  return Compressible(a.rho*b, a.rhoU*b, a.e*b);
}

template <typename S>
inline Compressible operator*(const S& b, const Compressible& a) {
  return Compressible(a.rho*b, a.rhoU*b, a.e*b);
}

template <typename S>
inline Compressible operator/(const Compressible& a, const S& b) {
  return Compressible(a.rho/b, a.rhoU/b, a.e/b);
}

inline Compressible operator+=(Compressible& a, const Compressible& b) {
  a.rho+=b.rho; a.rhoU+=b.rhoU; a.e+=b.e;
  return a;
}

inline Compressible operator-=(Compressible& a, const Compressible& b) {
  a.rho-=b.rho; a.rhoU-=b.rhoU; a.e-=b.e;
  return a;
}

template <typename S>
inline Compressible operator*=(Compressible& a, const S& b) {
  a.rho*=b; a.rhoU*=b; a.e*=b;
  return a;
}

template <typename S>
inline Compressible operator/=(Compressible& a, const S& b) {
  a.rho/=b; a.rhoU/=b; a.e/=b;
  return a;
}

#endif
