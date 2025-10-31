package dev.demir.vulkan.bvh;

import dev.demir.vulkan.util.Vec3;

/**
 * Represents an Axis-Aligned Bounding Box (AABB).
 * This is a clean re-implementation of the logic from 'raytracer-java/core/AABB.java'.
 *
 * It provides the core logic for the BVH build process, specifically
 * calculating the bounding box of child nodes.
 *
 */
public class AABB {

    // Axis-Aligned Bounding Box

    public final Vec3 min;
    public final Vec3 max;

    /**
     * Creates a new Bounding Box from two points.
     * @param min The minimum corner (lowest x, y, z).
     * @param max The maximum corner (highest x, y, z).
     */
    public AABB(Vec3 min, Vec3 max) {
        this.min = min;
        this.max = max;
    }

    /**
     * Creates a new AABB that encompasses two existing AABBs.
     * This is the core function for building the BVH tree.
     *
     * @param box0 The first bounding box.
     * @param box1 The second bounding box.
     * @return A new AABB that contains both box0 and box1.
     */
    public static AABB surroundingBox(AABB box0, AABB box1) {
        if (box0 == null) return box1;
        if (box1 == null) return box0;

        Vec3 small = Vec3.min(box0.min, box1.min);
        Vec3 big   = Vec3.max(box0.max, box1.max);

        return new AABB(small, big);
    }

    /**
     * Gets the longest axis of the bounding box.
     * Used by the BVH builder to decide where to split the objects.
     * @return 0 for X, 1 for Y, 2 for Z.
     */
    public int getLongestAxis() {
        double x_diag = max.x - min.x;
        double y_diag = max.y - min.y;
        double z_diag = max.z - min.z;

        if (x_diag > y_diag && x_diag > z_diag) {
            return 0; // X axis
        } else if (y_diag > z_diag) {
            return 1; // Y axis
        } else {
            return 2; // Z axis
        }
    }
}