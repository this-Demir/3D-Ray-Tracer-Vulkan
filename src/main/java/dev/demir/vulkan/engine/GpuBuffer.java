package dev.demir.vulkan.engine;

import java.nio.ByteBuffer;

import static org.lwjgl.system.MemoryUtil.memByteBuffer;

/**
 * A VkBuffer with its own memory allocation. Created and destroyed by {@link GpuMemory}.
 *
 * @param buffer        VkBuffer handle
 * @param memory        VkDeviceMemory handle
 * @param size          requested size in bytes
 * @param deviceAddress GPU address (0 unless created with SHADER_DEVICE_ADDRESS usage)
 * @param mappedAddress persistent CPU mapping (0 unless the memory is host-visible)
 */
public record GpuBuffer(long buffer, long memory, long size, long deviceAddress, long mappedAddress) {

    /** CPU view of a host-visible buffer (native byte order). */
    public ByteBuffer mapped() {
        if (mappedAddress == 0) {
            throw new IllegalStateException("Buffer is not host-visible");
        }
        return memByteBuffer(mappedAddress, (int) size);
    }
}
