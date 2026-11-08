package dev.demir.vulkan.engine;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.function.Consumer;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memCopy;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.vulkan.VK10.*;

/**
 * Buffer/memory helper for the render thread: picks memory types, creates buffers
 * (optionally with a device address or a persistent CPU mapping), uploads data to
 * VRAM through a staging buffer, and runs one-off command buffers.
 *
 * Not thread-safe; used only on the Vulkan render thread.
 */
public final class GpuMemory {

    private final VkDevice device;
    private final VkQueue queue;
    private final VkPhysicalDeviceMemoryProperties memoryProperties;
    private final long commandPool; // transient pool for one-off submissions

    public GpuMemory(VkDevice device, VkQueue queue, int queueFamilyIndex) {
        this.device = device;
        this.queue = queue;
        this.memoryProperties = VkPhysicalDeviceMemoryProperties.calloc();
        vkGetPhysicalDeviceMemoryProperties(device.getPhysicalDevice(), memoryProperties);
        try (MemoryStack stack = stackPush()) {
            VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                    .sType$Default()
                    .flags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT)
                    .queueFamilyIndex(queueFamilyIndex);
            LongBuffer pPool = stack.mallocLong(1);
            check(vkCreateCommandPool(device, poolInfo, null, pPool), "vkCreateCommandPool");
            commandPool = pPool.get(0);
        }
    }

    /**
     * Index of a memory type allowed by typeBits that has all 'required' flags,
     * preferring one that also has all 'preferred' flags.
     */
    public int findMemoryType(int typeBits, int required, int preferred) {
        int fallback = -1;
        for (int i = 0; i < memoryProperties.memoryTypeCount(); i++) {
            if ((typeBits & (1 << i)) == 0) continue;
            int flags = memoryProperties.memoryTypes(i).propertyFlags();
            if ((flags & required) != required) continue;
            if ((flags & preferred) == preferred) return i;
            if (fallback == -1) fallback = i;
        }
        if (fallback == -1) {
            throw new RuntimeException("No suitable memory type (required flags 0x" + Integer.toHexString(required) + ")");
        }
        return fallback;
    }

    /**
     * Creates a buffer with its own allocation. Host-visible memory is mapped persistently;
     * buffers with SHADER_DEVICE_ADDRESS usage get their GPU address filled in.
     */
    public GpuBuffer createBuffer(long size, int usage, int requiredMemory, int preferredMemory) {
        try (MemoryStack stack = stackPush()) {
            VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                    .sType$Default()
                    .size(size)
                    .usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer pBuffer = stack.mallocLong(1);
            check(vkCreateBuffer(device, bufferInfo, null, pBuffer), "vkCreateBuffer");
            long buffer = pBuffer.get(0);

            VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
            vkGetBufferMemoryRequirements(device, buffer, requirements);
            int memoryType = findMemoryType(requirements.memoryTypeBits(), requiredMemory, preferredMemory);

            boolean needsAddress = (usage & VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
            VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                    .sType$Default()
                    .allocationSize(requirements.size())
                    .memoryTypeIndex(memoryType);
            if (needsAddress) {
                VkMemoryAllocateFlagsInfo flags = VkMemoryAllocateFlagsInfo.calloc(stack)
                        .sType$Default()
                        .flags(VK12.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
                allocInfo.pNext(flags.address());
            }
            LongBuffer pMemory = stack.mallocLong(1);
            int result = vkAllocateMemory(device, allocInfo, null, pMemory);
            if (result != VK_SUCCESS) {
                vkDestroyBuffer(device, buffer, null);
                throw new RuntimeException("vkAllocateMemory failed (" + result + ") for " + size + " bytes");
            }
            long memory = pMemory.get(0);
            check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

            long deviceAddress = 0;
            if (needsAddress) {
                deviceAddress = VK12.vkGetBufferDeviceAddress(device,
                        VkBufferDeviceAddressInfo.calloc(stack).sType$Default().buffer(buffer));
            }
            long mappedAddress = 0;
            int flags = memoryProperties.memoryTypes(memoryType).propertyFlags();
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                PointerBuffer pData = stack.mallocPointer(1);
                check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, pData), "vkMapMemory");
                mappedAddress = pData.get(0);
            }
            return new GpuBuffer(buffer, memory, size, deviceAddress, mappedAddress);
        }
    }

    /**
     * Creates a DEVICE_LOCAL (VRAM) buffer and fills it with the given bytes via a staging buffer.
     */
    public GpuBuffer createDeviceLocalBuffer(ByteBuffer data, int usage) {
        long size = data.remaining();
        GpuBuffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        try {
            memCopy(memAddress(data), staging.mappedAddress(), size);
            GpuBuffer target = createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
            submitAndWait(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).size(size);
                    vkCmdCopyBuffer(cmd, staging.buffer(), target.buffer(), region);
                }
            });
            return target;
        } finally {
            destroy(staging);
        }
    }

    public void destroy(GpuBuffer buffer) {
        if (buffer == null) return;
        vkDestroyBuffer(device, buffer.buffer(), null);
        vkFreeMemory(device, buffer.memory(), null); // also unmaps
    }

    /**
     * Records commands into a one-off command buffer, submits it and waits until the GPU is done.
     */
    public void submitAndWait(Consumer<VkCommandBuffer> recorder) {
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType$Default()
                    .commandPool(commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);
            PointerBuffer pCmd = stack.mallocPointer(1);
            check(vkAllocateCommandBuffers(device, allocInfo, pCmd), "vkAllocateCommandBuffers");
            VkCommandBuffer cmd = new VkCommandBuffer(pCmd.get(0), device);
            try {
                vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack)
                        .sType$Default()
                        .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
                recorder.accept(cmd);
                check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(pCmd);
                check(vkQueueSubmit(queue, submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
                check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
            } finally {
                vkFreeCommandBuffers(device, commandPool, pCmd);
            }
        }
    }

    public void destroy() {
        vkDestroyCommandPool(device, commandPool, null);
        memoryProperties.free();
    }

    static void check(int result, String call) {
        if (result != VK_SUCCESS) {
            throw new RuntimeException(call + " failed: VkResult " + result);
        }
    }
}
