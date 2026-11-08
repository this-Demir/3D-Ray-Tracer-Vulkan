package dev.demir.vulkan.scene;

import java.util.List;

/**
 * Immutable snapshot of the scene, handed from the UI / scene rebuild thread to the
 * Vulkan render thread. Holds no Vulkan objects.
 */
public record SceneDescription(List<Instance> instances) {

    public SceneDescription {
        instances = List.copyOf(instances);
    }

    /**
     * One placed copy of a geometry.
     *
     * @param geometry  shared shape data (one BLAS per distinct geometry on the GPU)
     * @param transform object-to-world, row-major 3x4
     * @param material  surface material
     */
    public record Instance(Geometry geometry, float[] transform, Material material) {
    }

    public long triangleCount() {
        long total = 0;
        for (Instance instance : instances) {
            total += instance.geometry().triangleCount();
        }
        return total;
    }
}
