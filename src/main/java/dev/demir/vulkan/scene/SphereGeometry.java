package dev.demir.vulkan.scene;

/**
 * The analytic unit sphere (radius 1, centered at the origin, in object space).
 * The instance transform places and sizes it; on the GPU it is a single-AABB BLAS whose
 * exact intersection is computed in the shader, so it stays perfectly round at any size.
 */
public enum SphereGeometry implements Geometry {
    INSTANCE;

    /** Model path used for sphere instances (no file behind it). */
    public static final String KEY = "builtin:sphere";

    @Override
    public String key() {
        return KEY;
    }

    @Override
    public int triangleCount() {
        return 0;
    }
}
