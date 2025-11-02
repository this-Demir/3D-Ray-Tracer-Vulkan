package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Represents a Ray in 3D space.
 * Logic is re-implemented based on 'raytracer-java/core/Ray.java'.
 * This is used by the CPU-side BVH builder.
 */
public class Ray {

    public final Vec3 origin;
    public final Vec3 direction;

    public Ray(Vec3 origin, Vec3 direction) {
        this.origin = origin;
        this.direction = direction;
    }

    /**
     * Calculates a point along the ray. P(t) = A + t*b
     * @param t The distance along the ray.
     * @return The point in 3D space.
     */
    public Vec3 at(double t) {
        return origin.add(direction.multiply(t));
    }
}