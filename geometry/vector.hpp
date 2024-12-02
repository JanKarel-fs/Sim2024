#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>
#include "point.hpp"

template <typename T>
class Vector2 {
public:
  T x;
  T y;

  Vector2() {};
  Vector2(const T& _x, const T& _y): x(_x), y(_y) {};
  Vector2(const Point2<T>& a, const Point<T>& b): x(b.x-a.x), y(b.y-a.y) {};
  ~Vector() {};

  inline T length() const {
    return sqrt(x*x + y*y);
  }
};

template <typename T>
inline Point2<T> operator+(const Point<T>& a, const Vector2<T>& b) {
  return Point2<T>(a.x+b.x, a.y+b.y);
}

template <typename T>
inline Point2<T> operator-(const Point<T>& a, const Vector2<T>& b) {
  return Point2<T>(a.x-b.x, a.y-b.y);
}

template <typename T>
inline Vector2<T> operator+(const Vector2<T>& a, const Vector2<T>& b) {
  return Vector2<T>(a.x+b.x, a.y+b.y);
}

template <typename T>
inline Vector2<T> operator-(const Vector2<T>& a, const Vector2<T>& b) {
  return Vector2<T>(a.x-b.x, a.y-b.y);
}

template <typename T>
inline Vector2<T> operator-(const Point2<T>& a, const Point2<T>& b) {
  return Vector2<T>(b, a);
}

template <typename T>
inline Vector2<T> operator*(const Vector2<T>& a, const Vector2<T>& b) {
  return Vector2<T>(a.x*b.x, a.y*b.y);
}

template <typename T, typename S>
inline Vector2<T> operator*(const Vector2<T>& a, const S& b) {
  return Vector2<T>(a.x*b, a.y*b);
}

template <typename T, typename S>
inline Vector2<T> operator*(const S& b, const Vector2<T>& a) {
  return Vector2<T>(a.x*b, a.y*b);
}

template <typename T, typename S>
inline Vector2<T> operator/(const Vector2<T>& a, const S& b) {
  return Vector2<T>(a.x/b, a.y/b);
}

template <typename T>
inline Vector2<T> operator+=(Vector2<T>& a, const Vector2<T>& b) {
  a.x+=b.x;  a.y+=b.y;
  return a;
}

template <typename T>
inline Vector2<T> operator-=(Vector2<T>& a, const Vector2<T>& b) {
  a.x-=b.x;  a.y-=b.y;
  return a;
}

template <typename T, typename S>
inline Vector2<T> operator*=(Vector2<T>& a, const S& b) {
  a.x*=b;  a.y*=b;
  return a;
}

template <typename T, typename S>
inline Vector2<T> operator/=(Vector2<T>& a, const S& b) {
  a.x/=b;  a.y/=b;
  return a;
}

template <typename T>
inline T dot(const Vector2<T>& a, const Vector2<T>& b) {
  return a.x*b.x + a.y*b.y;
}

template <typename T>
inline T cross(const Vector2<T>& a, const Vector2<T>& b) {
  return a.x*b.y - a.y*b.x;
}

typedef Vector2<double> Vector2d;

#endif
