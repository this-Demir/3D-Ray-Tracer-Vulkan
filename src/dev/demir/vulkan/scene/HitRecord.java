package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Stores information about a ray-object intersection.
 * Logic is re-implemented based on 'raytracer-java/core/HitRecord.java'.
 * This is used by the CPU-side BVH builder.
 *
 * (Note: Material details are simplified for now,
 * as we are focused on BVH structure first.)
 */
public class HitRecord {

    public double t;      // Time/distance of the hit
    public Vec3 p;      // Point of intersection
    public Vec3 normal; // Normal at the intersection

    // (Material field will be added later when we port materials)
    // public Material material;

    public boolean frontFace; // True if the ray hit the front face

    /**
     * Sets the face normal based on the ray direction.
     * @param r The incoming ray.
     * @param outwardNormal The geometry's normal vector.
     */
    public void setFaceNormal(Ray r, Vec3 outwardNormal) {
        frontFace = Vec3.dot(r.direction, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : outwardNormal.negative();
    }
}