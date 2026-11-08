package dev.demir.vulkan.engine;

/**
 * Render statistics published by the render thread about once per second.
 *
 * @param renderFps frames rendered per second (independent of how often the UI repaints)
 * @param gpuMs     average GPU time per frame from timestamp queries, in milliseconds
 * @param samples   samples per pixel accumulated since the last change
 */
public record EngineStats(double renderFps, double gpuMs, long samples) {

    public static final EngineStats NONE = new EngineStats(0, 0, 0);
}
