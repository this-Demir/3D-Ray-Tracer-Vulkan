package dev.demir.vulkan.scene;

import dev.demir.vulkan.bvh.AABB;
import dev.demir.vulkan.util.Vec3;

/**
 * Represents a single Triangle in the scene.
 * This class implements the 'Hittable' interface, allowing it
 * to be stored and sorted by the BVHNode builder.
 *
 * CORRECTED: Now stores material color and type data.
 */
public class Triangle implements Hittable {

    public final Vec3 v0, v1, v2;
    private final AABB bbox;
    public final int materialIndex;
    public final int vertexIndex;

    public final float r, g, b;
    public final float materialType; // 0.0f = Lambertian

    /**
     * Creates a new Triangle object.
     * @param v0 Vertex 0
     * @param v1 Vertex 1
     * @param v2 Vertex 2
     * @param matIndex The index of this triangle's material.
     * @param vertIndex The starting index (e.g., i * 3) of this triangle in the main vertex buffer.
     * @param r Red component
     * @param g Green component
     * @param b Blue component
     * @param matType Material type (0.0f for Lambertian)
     */
    public Triangle(Vec3 v0, Vec3 v1, Vec3 v2, int matIndex, int vertIndex,
                    float r, float g, float b, float matType) {
        this.v0 = v0;
        this.v1 = v1;
        this.v2 = v2;
        this.materialIndex = matIndex;
        this.vertexIndex = vertIndex;

        this.r = r;
        this.g = g;
        this.b = b;


        this.materialType = matType;

        this.bbox = calculateBoundingBox();
    }

    @Override
    public AABB boundingBox() {
        return this.bbox;
    }

    /**
     * Calculates the AABB that tightly encloses this triangle's 3 vertices.
     */
    private AABB calculateBoundingBox() {
        Vec3 min = Vec3.min(v0, Vec3.min(v1, v2));
        Vec3 max = Vec3.max(v0, Vec3.max(v1, v2));

        double epsilon = 0.0001;
        if (max.x - min.x < epsilon) max = max.add(new Vec3(epsilon, 0, 0));
        if (max.y - min.y < epsilon) max = max.add(new Vec3(0, epsilon, 0));
        if (max.z - min.z < epsilon) max = max.add(new Vec3(0, 0, epsilon));

        return new AABB(min, max);
    }

    /**
     * Calculates the center point of the triangle's bounding box.
     * Used by the BVH builder for sorting.
     */
    public Vec3 getCenter() {
        return bbox.min.add(bbox.max).div(2.0);
    }
}