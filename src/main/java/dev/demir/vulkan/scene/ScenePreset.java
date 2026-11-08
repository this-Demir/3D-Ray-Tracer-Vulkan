package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

import java.util.List;
import java.util.function.Supplier;

/**
 * A ready-made scene: objects, camera and render settings.
 *
 * @param name     shown in the scene selector
 * @param objects  creates fresh, editable objects each time the preset is loaded
 * @param cameraOrigin camera position
 * @param cameraLookAt point the camera looks at
 * @param vfov     vertical field of view in degrees
 * @param settings sky, exposure, bounce and sample limits
 */
public record ScenePreset(String name, Supplier<List<ModelInstance>> objects,
                          Vec3 cameraOrigin, Vec3 cameraLookAt, double vfov, RenderSettings settings) {

    @Override
    public String toString() {
        return name;
    }
}
