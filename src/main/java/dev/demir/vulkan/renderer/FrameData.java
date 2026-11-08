package dev.demir.vulkan.renderer;

/**
 * A finished frame, handed from the render thread to the UI thread.
 * Pixels are 0x00RRGGBB ints (the layout of BufferedImage.TYPE_INT_RGB), row by row, top row first.
 * The array is pooled: the UI gives the frame back with VulkanEngine.recycleFrame() after copying it.
 */
public record FrameData(int[] pixels, int width, int height) {
}
