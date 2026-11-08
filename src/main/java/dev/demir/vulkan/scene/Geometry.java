package dev.demir.vulkan.scene;

/**
 * Shape data shared by instances: either a triangle mesh loaded from a file, or a
 * built-in analytic primitive. Each distinct geometry becomes one BLAS on the GPU.
 */
public sealed interface Geometry permits MeshData, SphereGeometry {

    /** Cache key; also what ModelInstance stores as its model path. */
    String key();

    /** Triangles in the geometry (0 for analytic primitives). */
    int triangleCount();
}
