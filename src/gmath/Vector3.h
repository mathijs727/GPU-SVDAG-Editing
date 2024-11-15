/**
 *  ============================================================================
 *  MIT License
 *
 *  Copyright (c) 2016 Eric Phillips
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *  ============================================================================
 *
 *
 *  This file implements a series of math functions for manipulating a
 *  3D vector.
 *
 *  Created by Eric Phillips on October 8, 2016.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <cuda_math.h>

struct Vector3
{
    union
    {
        struct
        {
            double X;
            double Y;
            double Z;
        };
        double data[3];
    };


    /**
     * Constructors.
     */
    inline Vector3();
    inline explicit Vector3(double data[]);
    inline explicit Vector3(double value);
    inline Vector3(double x, double y);
    inline Vector3(double x, double y, double z);
    inline explicit Vector3(float3 v);
    inline explicit Vector3(double3 v);


    /**
     * Constants for common vectors.
     */
    static inline Vector3 Zero();
    static inline Vector3 One();
    static inline Vector3 Right();
    static inline Vector3 Left();
    static inline Vector3 Up();
    static inline Vector3 Down();
    static inline Vector3 Forward();
    static inline Vector3 Backward();

    /**
     * Returns the angle between two vectors in radians.
     * @param a: The first vector.
     * @param b: The second vector.
     * @return: A scalar value.
     */
    static inline double Angle(Vector3 a, Vector3 b);

    /**
     * Returns a vector with its magnitude clamped to maxLength.
     * @param vector: The target vector.
     * @param maxLength: The maximum length of the return vector.
     * @return: A new vector.
     */
    static inline Vector3 ClampMagnitude(Vector3 vector, double maxLength);

    /**
     * Returns the component of a in the direction of b (scalar projection).
     * @param a: The target vector.
     * @param b: The vector being compared against.
     * @return: A scalar value.
     */
    static inline double Component(Vector3 a, Vector3 b);

    /**
     * Returns the cross product of two vectors.
     * @param lhs: The left side of the multiplication.
     * @param rhs: The right side of the multiplication.
     * @return: A new vector.
     */
    static inline Vector3 Cross(Vector3 lhs, Vector3 rhs);

    /**
     * Returns the distance between a and b.
     * @param a: The first point.
     * @param b: The second point.
     * @return: A scalar value.
     */
    static inline double Distance(Vector3 a, Vector3 b);

    /**
     * Returns the dot product of two vectors.
     * @param lhs: The left side of the multiplication.
     * @param rhs: The right side of the multiplication.
     * @return: A scalar value.
     */
    static inline double Dot(Vector3 lhs, Vector3 rhs);

    /**
     * Converts a spherical representation of a vector into cartesian
     * coordinates.
     * This uses the ISO convention (radius r, inclination theta, azimuth phi).
     * @param rad: The magnitude of the vector.
     * @param theta: The angle in the XY plane from the X axis.
     * @param phi: The angle from the positive Z axis to the vector.
     * @return: A new vector.
     */
    static inline Vector3 FromSpherical(double rad, double theta, double phi);

    /**
     * Returns a vector linearly interpolated between a and b, moving along
     * a straight line. The vector is clamped to never go beyond the end points.
     * @param a: The starting point.
     * @param b: The ending point.
     * @param t: The interpolation value [0-1].
     * @return: A new vector.
     */
    static inline Vector3 Lerp(Vector3 a, Vector3 b, double t);

    /**
     * Returns a vector linearly interpolated between a and b, moving along
     * a straight line.
     * @param a: The starting point.
     * @param b: The ending point.
     * @param t: The interpolation value [0-1] (no actual bounds).
     * @return: A new vector.
     */
    static inline Vector3 LerpUnclamped(Vector3 a, Vector3 b, double t);

    /**
     * Returns the magnitude of a vector.
     * @param v: The vector in question.
     * @return: A scalar value.
     */
    static inline double Magnitude(Vector3 v);

    /**
     * Returns a vector made from the largest components of two other vectors.
     * @param a: The first vector.
     * @param b: The second vector.
     * @return: A new vector.
     */
    static inline Vector3 Max(Vector3 a, Vector3 b);

    /**
     * Returns a vector made from the smallest components of two other vectors.
     * @param a: The first vector.
     * @param b: The second vector.
     * @return: A new vector.
     */
    static inline Vector3 Min(Vector3 a, Vector3 b);

    /**
     * Returns a vector "maxDistanceDelta" units closer to the target. This
     * interpolation is in a straight line, and will not overshoot.
     * @param current: The current position.
     * @param target: The destination position.
     * @param maxDistanceDelta: The maximum distance to move.
     * @return: A new vector.
     */
    static inline Vector3 MoveTowards(Vector3 current, Vector3 target,
                               double maxDistanceDelta);

    /**
     * Returns a new vector with magnitude of one.
     * @param v: The vector in question.
     * @return: A new vector.
     */
    static inline Vector3 Normalized(Vector3 v);

    /**
     * Returns an arbitrary vector orthogonal to the input.
     * This vector is not normalized.
     * @param v: The input vector.
     * @return: A new vector.
     */
    static inline Vector3 Orthogonal(Vector3 v);

    /**
     * Creates a new coordinate system out of the three vectors.
     * Normalizes "normal", normalizes "tangent" and makes it orthogonal to
     * "normal" and normalizes "binormal" and makes it orthogonal to both
     * "normal" and "tangent".
     * @param normal: A reference to the first axis vector.
     * @param tangent: A reference to the second axis vector.
     * @param binormal: A reference to the third axis vector.
     */
    static inline void OrthoNormalize(Vector3 &normal, Vector3 &tangent,
                               Vector3 &binormal);

    /**
     * Returns the vector projection of a onto b.
     * @param a: The target vector.
     * @param b: The vector being projected onto.
     * @return: A new vector.
     */
    static inline Vector3 Project(Vector3 a, Vector3 b);

    /**
     * Returns a vector projected onto a plane orthogonal to "planeNormal".
     * This can be visualized as the shadow of the vector onto the plane, if
     * the light source were in the direction of the plane normal.
     * @param vector: The vector to project.
     * @param planeNormal: The normal of the plane onto which to project.
     * @param: A new vector.
     */
    static inline Vector3 ProjectOnPlane(Vector3 vector, Vector3 planeNormal);

    /**
     * Returns a vector reflected off the plane orthogonal to the normal.
     * The input vector is pointed inward, at the plane, and the return vector
     * is pointed outward from the plane, like a beam of light hitting and then
     * reflecting off a mirror.
     * @param vector: The vector traveling inward at the plane.
     * @param planeNormal: The normal of the plane off of which to reflect.
     * @return: A new vector pointing outward from the plane.
     */
    static inline Vector3 Reflect(Vector3 vector, Vector3 planeNormal);

    /**
     * Returns the vector rejection of a on b.
     * @param a: The target vector.
     * @param b: The vector being projected onto.
     * @return: A new vector.
     */
    static inline Vector3 Reject(Vector3 a, Vector3 b);

    /**
     * Rotates vector "current" towards vector "target" by "maxRadiansDelta".
     * This treats the vectors as directions and will linearly interpolate
     * between their magnitudes by "maxMagnitudeDelta". This function does not
     * overshoot. If a negative delta is supplied, it will rotate away from
     * "target" until it is pointing the opposite direction, but will not
     * overshoot that either.
     * @param current: The starting direction.
     * @param target: The destination direction.
     * @param maxRadiansDelta: The maximum number of radians to rotate.
     * @param maxMagnitudeDelta: The maximum delta for magnitude interpolation.
     * @return: A new vector.
     */
    static inline Vector3 RotateTowards(Vector3 current, Vector3 target,
                                 double maxRadiansDelta,
                                 double maxMagnitudeDelta);

    /**
     * Multiplies two vectors element-wise.
     * @param a: The lhs of the multiplication.
     * @param b: The rhs of the multiplication.
     * @return: A new vector.
     */
    static inline Vector3 Scale(Vector3 a, Vector3 b);

    /**
     * Returns a vector rotated towards b from a by the percent t.
     * Since interpolation is done spherically, the vector moves at a constant
     * angular velocity. This rotation is clamped to 0 <= t <= 1.
     * @param a: The starting direction.
     * @param b: The ending direction.
     * @param t: The interpolation value [0-1].
     */
    static inline Vector3 Slerp(Vector3 a, Vector3 b, double t);

    /**
     * Returns a vector rotated towards b from a by the percent t.
     * Since interpolation is done spherically, the vector moves at a constant
     * angular velocity. This rotation is unclamped.
     * @param a: The starting direction.
     * @param b: The ending direction.
     * @param t: The interpolation value [0-1].
     */
    static inline Vector3 SlerpUnclamped(Vector3 a, Vector3 b, double t);

    /**
     * Returns the squared magnitude of a vector.
     * This is useful when comparing relative lengths, where the exact length
     * is not important, and much time can be saved by not calculating the
     * square root.
     * @param v: The vector in question.
     * @return: A scalar value.
     */
    static inline double SqrMagnitude(Vector3 v);

    /**
     * Calculates the spherical coordinate space representation of a vector.
     * This uses the ISO convention (radius r, inclination theta, azimuth phi).
     * @param vector: The vector to convert.
     * @param rad: The magnitude of the vector.
     * @param theta: The angle in the XY plane from the X axis.
     * @param phi: The angle from the positive Z axis to the vector.
     */
    static inline void ToSpherical(Vector3 vector, double &rad, double &theta,
                            double &phi);


	inline Vector3 abs() const
	{
		return Vector3(std::abs(X), std::abs(Y), std::abs(Z) );

	}
	inline double largest() const
	{
		return std::max( X, std::max( Y, Z ) );
	}


    /**
     * Operator overloading.
     */
    inline struct Vector3& operator+=(const double rhs);
    inline struct Vector3& operator-=(const double rhs);
    inline struct Vector3& operator*=(const double rhs);
    inline struct Vector3& operator/=(const double rhs);
    inline struct Vector3& operator+=(const Vector3 rhs);
    inline struct Vector3& operator-=(const Vector3 rhs);
};

inline Vector3 operator-(Vector3 rhs);
inline Vector3 operator+(Vector3 lhs, const double rhs);
inline Vector3 operator-(Vector3 lhs, const double rhs);
inline Vector3 operator*(Vector3 lhs, const double rhs);
inline Vector3 operator/(Vector3 lhs, const double rhs);
inline Vector3 operator+(const double lhs, Vector3 rhs);
inline Vector3 operator-(const double lhs, Vector3 rhs);
inline Vector3 operator*(const double lhs, Vector3 rhs);
inline Vector3 operator/(const double lhs, Vector3 rhs);
inline Vector3 operator*(Vector3 lhs, Vector3 rhs);
inline Vector3 operator+(Vector3 lhs, const Vector3 rhs);
inline Vector3 operator-(Vector3 lhs, const Vector3 rhs);
inline bool operator==(const Vector3 lhs, const Vector3 rhs);
inline bool operator!=(const Vector3 lhs, const Vector3 rhs);

HOST_DEVICE float3 make_float3(Vector3 v) { return make_vector3<float3>(float(v.X), float(v.Y), float(v.Z)); }


/*******************************************************************************
 * Implementation
 */

Vector3::Vector3() : X(0), Y(0), Z(0) {}
Vector3::Vector3(double data[]) : X(data[0]), Y(data[1]), Z(data[2]) {}
Vector3::Vector3(double value) : X(value), Y(value), Z(value) {}
Vector3::Vector3(double x, double y) : X(x), Y(y), Z(0) {}
Vector3::Vector3(double x, double y, double z) : X(x), Y(y), Z(z) {}
Vector3::Vector3(float3 v) : X(v.x), Y(v.y), Z(v.z) {}
Vector3::Vector3(double3 v) : X(v.x), Y(v.y), Z(v.z) {}


Vector3 Vector3::Zero() { return Vector3(0, 0, 0); }
Vector3 Vector3::One() { return Vector3(1, 1, 1); }
Vector3 Vector3::Right() { return Vector3(1, 0, 0); }
Vector3 Vector3::Left() { return Vector3(-1, 0, 0); }
Vector3 Vector3::Up() { return Vector3(0, 1, 0); }
Vector3 Vector3::Down() { return Vector3(0, -1, 0); }
Vector3 Vector3::Forward() { return Vector3(0, 0, 1); }
Vector3 Vector3::Backward() { return Vector3(0, 0, -1); }


double Vector3::Angle(Vector3 a, Vector3 b)
{
    double v = Dot(a, b) / (Magnitude(a) * Magnitude(b));
    v = fmax(v, -1.0);
    v = fmin(v, 1.0);
    return acos(v);
}

Vector3 Vector3::ClampMagnitude(Vector3 vector, double maxLength)
{
    double length = Magnitude(vector);
    if (length > maxLength)
        vector *= maxLength / length;
    return vector;
}

double Vector3::Component(Vector3 a, Vector3 b)
{
    return Dot(a, b) / Magnitude(b);
}

Vector3 Vector3::Cross(Vector3 lhs, Vector3 rhs)
{
    double x = lhs.Y * rhs.Z - lhs.Z * rhs.Y;
    double y = lhs.Z * rhs.X - lhs.X * rhs.Z;
    double z = lhs.X * rhs.Y - lhs.Y * rhs.X;
    return Vector3(x, y, z);
}

double Vector3::Distance(Vector3 a, Vector3 b)
{
    return Vector3::Magnitude(a - b);
}

double Vector3::Dot(Vector3 lhs, Vector3 rhs)
{
    return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z;
}

Vector3 Vector3::FromSpherical(double rad, double theta, double phi)
{
    Vector3 v;
    v.X = rad * sin(theta) * cos(phi);
    v.Y = rad * sin(theta) * sin(phi);
    v.Z = rad * cos(theta);
    return v;
}

Vector3 Vector3::Lerp(Vector3 a, Vector3 b, double t)
{
    if (t < 0) return a;
    else if (t > 1) return b;
    return LerpUnclamped(a, b, t);
}

Vector3 Vector3::LerpUnclamped(Vector3 a, Vector3 b, double t)
{
    return (b - a) * t + a;
}

double Vector3::Magnitude(Vector3 v)
{
    return sqrt(SqrMagnitude(v));
}

Vector3 Vector3::Max(Vector3 a, Vector3 b)
{
    double x = a.X > b.X ? a.X : b.X;
    double y = a.Y > b.Y ? a.Y : b.Y;
    double z = a.Z > b.Z ? a.Z : b.Z;
    return Vector3(x, y, z);
}

Vector3 Vector3::Min(Vector3 a, Vector3 b)
{
    double x = a.X > b.X ? b.X : a.X;
    double y = a.Y > b.Y ? b.Y : a.Y;
    double z = a.Z > b.Z ? b.Z : a.Z;
    return Vector3(x, y, z);
}

Vector3 Vector3::MoveTowards(Vector3 current, Vector3 target,
                             double maxDistanceDelta)
{
    Vector3 d = target - current;
    double m = Magnitude(d);
    if (m < maxDistanceDelta || m == 0)
        return target;
    return current + (d * maxDistanceDelta / m);
}

Vector3 Vector3::Normalized(Vector3 v)
{
    double mag = Magnitude(v);
    if (mag == 0)
        return Vector3::Zero();
    return v / mag;
}

Vector3 Vector3::Orthogonal(Vector3 v)
{
    return v.Z < v.X ? Vector3(v.Y, -v.X, 0) : Vector3(0, -v.Z, v.Y);
}

void Vector3::OrthoNormalize(Vector3 &normal, Vector3 &tangent,
                             Vector3 &binormal)
{
    normal = Normalized(normal);
    tangent = ProjectOnPlane(tangent, normal);
    tangent = Normalized(tangent);
    binormal = ProjectOnPlane(binormal, tangent);
    binormal = ProjectOnPlane(binormal, normal);
    binormal = Normalized(binormal);
}

Vector3 Vector3::Project(Vector3 a, Vector3 b)
{
    double m = Magnitude(b);
    return Dot(a, b) / (m * m) * b;
}

Vector3 Vector3::ProjectOnPlane(Vector3 vector, Vector3 planeNormal)
{
    return Reject(vector, planeNormal);
}

Vector3 Vector3::Reflect(Vector3 vector, Vector3 planeNormal)
{
    return vector - 2 * Project(vector, planeNormal);
}

Vector3 Vector3::Reject(Vector3 a, Vector3 b)
{
    return a - Project(a, b);
}

Vector3 Vector3::RotateTowards(Vector3 current, Vector3 target,
                               double maxRadiansDelta,
                               double maxMagnitudeDelta)
{
    double magCur = Magnitude(current);
    double magTar = Magnitude(target);
    double newMag = magCur + maxMagnitudeDelta *
        ((magTar > magCur) - (magCur > magTar));
    newMag = fmin(newMag, fmax(magCur, magTar));
    newMag = fmax(newMag, fmin(magCur, magTar));

    double totalAngle = Angle(current, target) - maxRadiansDelta;
    if (totalAngle <= 0)
        return Normalized(target) * newMag;
    else if (totalAngle >= M_PI)
        return Normalized(-target) * newMag;

    Vector3 axis = Cross(current, target);
    double magAxis = Magnitude(axis);
    if (magAxis == 0)
        axis = Normalized(Cross(current, current + Vector3(3.95, 5.32, -4.24)));
    else
        axis /= magAxis;
    current = Normalized(current);
    Vector3 newVector = current * cos(maxRadiansDelta) +
        Cross(axis, current) * sin(maxRadiansDelta);
    return newVector * newMag;
}

Vector3 Vector3::Scale(Vector3 a, Vector3 b)
{
    return Vector3(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
}

Vector3 Vector3::Slerp(Vector3 a, Vector3 b, double t)
{
    if (t < 0) return a;
    else if (t > 1) return b;
    return SlerpUnclamped(a, b, t);
}

Vector3 Vector3::SlerpUnclamped(Vector3 a, Vector3 b, double t)
{
    double magA = Magnitude(a);
    double magB = Magnitude(b);
    a /= magA;
    b /= magB;
    double dot = Dot(a, b);
    dot = fmax(dot, -1.0);
    dot = fmin(dot, 1.0);
    double theta = acos(dot) * t;
    Vector3 relativeVec = Normalized(b - a * dot);
    Vector3 newVec = a * cos(theta) + relativeVec * sin(theta);
    return newVec * (magA + (magB - magA) * t);
}

double Vector3::SqrMagnitude(Vector3 v)
{
    return v.X * v.X + v.Y * v.Y + v.Z * v.Z;
}

void Vector3::ToSpherical(Vector3 vector, double &rad, double &theta,
                          double &phi)
{
    rad = Magnitude(vector);
    double v = vector.Z / rad;
    v = fmax(v, -1.0);
    v = fmin(v, 1.0);
    theta = acos(v);
    phi = atan2(vector.Y, vector.X);
}


struct Vector3& Vector3::operator+=(const double rhs)
{
    X += rhs;
    Y += rhs;
    Z += rhs;
    return *this;
}

struct Vector3& Vector3::operator-=(const double rhs)
{
    X -= rhs;
    Y -= rhs;
    Z -= rhs;
    return *this;
}

struct Vector3& Vector3::operator*=(const double rhs)
{
    X *= rhs;
    Y *= rhs;
    Z *= rhs;
    return *this;
}

struct Vector3& Vector3::operator/=(const double rhs)
{
    X /= rhs;
    Y /= rhs;
    Z /= rhs;
    return *this;
}

struct Vector3& Vector3::operator+=(const Vector3 rhs)
{
    X += rhs.X;
    Y += rhs.Y;
    Z += rhs.Z;
    return *this;
}

struct Vector3& Vector3::operator-=(const Vector3 rhs)
{
    X -= rhs.X;
    Y -= rhs.Y;
    Z -= rhs.Z;
    return *this;
}

Vector3 operator-(Vector3 rhs) { return rhs * -1; }
Vector3 operator+(Vector3 lhs, const double rhs) { return lhs += rhs; }
Vector3 operator-(Vector3 lhs, const double rhs) { return lhs -= rhs; }
Vector3 operator*(Vector3 lhs, const double rhs) { return lhs *= rhs; }
Vector3 operator/(Vector3 lhs, const double rhs) { return lhs /= rhs; }
Vector3 operator+(const double lhs, Vector3 rhs) { return rhs += lhs; }
Vector3 operator-(const double lhs, Vector3 rhs) { return rhs -= lhs; }
Vector3 operator*(const double lhs, Vector3 rhs) { return rhs *= lhs; }
Vector3 operator/(const double lhs, Vector3 rhs) { return rhs /= lhs; }
Vector3 operator*(Vector3 lhs, Vector3 rhs) { return Vector3(lhs.X * rhs.X, lhs.Y * rhs.Y, lhs.Z * rhs.Z); }
Vector3 operator+(Vector3 lhs, const Vector3 rhs) { return lhs += rhs; }
Vector3 operator-(Vector3 lhs, const Vector3 rhs) { return lhs -= rhs; }

bool operator==(const Vector3 lhs, const Vector3 rhs)
{
    return lhs.X == rhs.X && lhs.Y == rhs.Y && lhs.Z == rhs.Z;
}

bool operator!=(const Vector3 lhs, const Vector3 rhs)
{
    return !(lhs == rhs);
}

inline double3 make_double3(const Vector3& v)
{
	return make_double3(v.X, v.Y, v.Z);
}
