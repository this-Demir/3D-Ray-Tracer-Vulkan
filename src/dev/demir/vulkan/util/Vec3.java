package dev.demir.vulkan.util;

import java.nio.ByteBuffer;

/**
 * Represents a 3D Vector or Point.
 * This is a clean re-implementation of the logic from 'raytracer-java/core/Vec3.java'
 * to be used by the CPU-side BVH builder.
 *
 * NOTE: This class is for CPU-side math only.
 * The equivalent logic in GLSL is the built-in 'vec3' type.
 */
public class Vec3 {

    public final double x, y, z;

    public Vec3(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    // --- Basic Operations ---

    public Vec3 add(Vec3 v) {
        return new Vec3(x + v.x, y + v.y, z + v.z);
    }

    public Vec3 sub(Vec3 v) {
        return new Vec3(x - v.x, y - v.y, z - v.z);
    }

    public Vec3 mul(double t) {
        return new Vec3(x * t, y * t, z * t);
    }

    public Vec3 div(double t) {
        return mul(1.0 / t);
    }

    public Vec3 negative() {
        return new Vec3(-x, -y, -z);
    }

    // --- Vector Operations ---

    public double lengthSquared() {
        return x * x + y * y + z * z;
    }

    public double length() {
        return Math.sqrt(lengthSquared());
    }

    public static Vec3 unitVector(Vec3 v) {
        return v.div(v.length());
    }

    public Vec3 unit() {
        return unitVector(this);
    }

    public static double dot(Vec3 u, Vec3 v) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    }

    public static Vec3 cross(Vec3 u, Vec3 v) {
        return new Vec3(
                u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x
        );
    }

    // --- Static Helpers (for min/max) ---

    public static Vec3 min(Vec3 a, Vec3 b) {
        return new Vec3(
                Math.min(a.x, b.x),
                Math.min(a.y, b.y),
                Math.min(a.z, b.z)
        );
    }

    public static Vec3 max(Vec3 a, Vec3 b) {
        return new Vec3(
                Math.max(a.x, b.x),
                Math.max(a.y, b.y),
                Math.max(a.z, b.z)
        );
    }

    /**
     * Helper to get a component by index (0=x, 1=y, 2=z).
     * Used by the BVH builder for sorting along axes.
     */
    public double get(int axis) {
        if (axis == 0) return x;
        if (axis == 1) return y;
        return z;
    }

    @Override
    public String toString() {
        return "Vec3(" + x + ", " + y + ", " + z + ")";
    }

    public Vec3 normalize() {
        double len2 = lengthSquared();
        if (len2 == 0.0) return this;
        double invLen = 1.0 / Math.sqrt(len2);
        return new Vec3(x * invLen, y * invLen, z * invLen);
    }


    /**
     * NEW: Helper method to store this vector's data in a ByteBuffer
     * as three FLOATS, for use with std140 UBOs.
     */
    public void store(int offset, ByteBuffer buffer) {
        buffer.putFloat(offset, (float)x);
        buffer.putFloat(offset + 4, (float)y);
        buffer.putFloat(offset + 8, (float)z);
    }
}