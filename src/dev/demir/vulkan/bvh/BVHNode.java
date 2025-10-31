package dev.demir.vulkan.bvh;

import dev.demir.vulkan.scene.Hittable;
import dev.demir.vulkan.scene.Ray;
import dev.demir.vulkan.scene.HitRecord;

/**
 * Represents a simple data node in the BVH tree.
 * CORRECTED: This class is now just a data container.
 * The complex build logic is moved to 'BVHBuilder.java',
 * which implements the logic from 'raytracer-java/core/BVHNode.java'.
 */
public class BVHNode implements Hittable {

    public final Hittable left;
    public final Hittable right;
    public final AABB bbox;

    /**
     * Simple constructor that just stores the children and their combined box.
     * The logic to *decide* these children is in BVHBuilder.
     */
    public BVHNode(Hittable left, Hittable right, AABB bbox) {
        this.left = left;
        this.right = right;
        this.bbox = bbox;
    }

    @Override
    public AABB boundingBox() {
        return this.bbox;
    }


    public HitRecord hit(Ray r, double tMin, double tMax) {
        // This logic will only be implemented on the GPU.
        throw new UnsupportedOperationException(
                "BVHNode.hit() is not implemented on the CPU. " +
                        "Traversal logic is handled by the GPU compute shader."
        );
    }
}