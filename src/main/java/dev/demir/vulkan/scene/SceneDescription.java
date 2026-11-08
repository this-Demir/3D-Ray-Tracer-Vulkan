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
     * One placed copy of a mesh.
     *
     * @param mesh         shared mesh data (one BLAS per distinct mesh on the GPU)
     * @param transform    object-to-world, row-major 3x4
     * @param albedo       r, g, b
     * @param materialType 0 = matte, 1 = metal, 2 = fuzzy metal, 3 = emissive
     */
    public record Instance(MeshData mesh, float[] transform, float[] albedo, float materialType) {
    }

    public long triangleCount() {
        long total = 0;
        for (Instance instance : instances) {
            total += instance.mesh().triangleCount();
        }
        return total;
    }
}
