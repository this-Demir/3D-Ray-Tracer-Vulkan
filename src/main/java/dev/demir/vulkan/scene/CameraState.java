package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Immutable camera snapshot handed from the UI thread to the render thread.
 * A ray for image coordinates (u, v) in [0, 1] goes from origin towards
 * lowerLeft + u * horizontal + v * vertical.
 */
public record CameraState(Vec3 origin, Vec3 lowerLeft, Vec3 horizontal, Vec3 vertical) {
}
