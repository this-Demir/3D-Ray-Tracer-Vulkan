package dev.demir.vulkan.scene;

import dev.demir.vulkan.bvh.AABB;

/**
 * CORRECTED Interface for any object used in the CPU-side BVH build.
 * Re-implemented based on 'raytracer-java/core/Hittable.java'.
 *
 * For our GPU pipeline, the CPU ONLY needs to know the object's bounding box.
 * The actual 'hit' logic will be implemented in the GLSL shader.
 */
public interface Hittable {

    /**
     * Calculates the bounding box for this object.
     * This is essential for building the BVH.
     * @return The AABB enclosing the object.
     */
    AABB boundingBox();
}