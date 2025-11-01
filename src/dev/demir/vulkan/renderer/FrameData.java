package dev.demir.vulkan.renderer;

import java.nio.ByteBuffer;

/**
 * POJO: Render'ı tamamlanmış bir karenin (frame) piksel verisini
 * VRT (Vulkan Render Thread) -> FXAT (JavaFX Thread) arasında taşır.
 */
public class FrameData {

    public final ByteBuffer pixelData;
    // TODO : RENDER STATISTICS

    public FrameData(ByteBuffer pixelData) {
        this.pixelData = pixelData;
    }
}