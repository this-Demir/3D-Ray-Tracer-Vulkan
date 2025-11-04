package dev.demir.vulkan.renderer;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.channels.FileChannel;
import java.util.function.LongConsumer;

import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwTerminate;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.VK10.*;

/**
 * "Phase 1: Refactor Engine"
 * This class is the refactored "dumb" renderer extracted from VulkanApp.java.
 * It contains ALL Vulkan logic but does not manage threads.
 *
 * It is NOT thread-safe and MUST be called only by the RenderLoop (VRT).
 */
public class VulkanRenderer {

    // --- Configuration (Could be provided by RenderLoop later; hard-coded for now) ---
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String SHADER_PATH = "shaders_spv/compute_with_dynamic_light_source.spv";
    private static final boolean ENABLE_VALIDATION_LAYERS = true;

    // --- Vulkan Core Objects ---
    private VkInstance instance;
    private long debugMessenger = VK_NULL_HANDLE;
    private VkPhysicalDevice physicalDevice;
    private VkDevice device;
    private VkQueue computeQueue;
    private int computeQueueFamilyIndex = -1;

    // --- Compute Resources ---
    private long computeImage = VK_NULL_HANDLE;
    private long computeImageMemory = VK_NULL_HANDLE;
    private long computeImageView = VK_NULL_HANDLE;
    private long stagingBuffer = VK_NULL_HANDLE;
    private long stagingBufferMemory = VK_NULL_HANDLE;

    // --- Pipeline Objects ---
    private long computeShaderModule = VK_NULL_HANDLE;
    private long descriptorSetLayout = VK_NULL_HANDLE;
    private long pipelineLayout = VK_NULL_HANDLE;
    private long computePipeline = VK_NULL_HANDLE;
    private long descriptorPool = VK_NULL_HANDLE;
    private long descriptorSet = VK_NULL_HANDLE;
    private long commandPool = VK_NULL_HANDLE;
    private VkCommandBuffer commandBuffer;
    private long fence = VK_NULL_HANDLE;

    // Static Validation Layer Setup
    private static final PointerBuffer VALIDATION_LAYERS;
    static {
        if (ENABLE_VALIDATION_LAYERS) {
            VALIDATION_LAYERS = memAllocPointer(1);
            VALIDATION_LAYERS.put(0, memASCII("VK_LAYER_KHRONOS_validation"));
        } else {
            VALIDATION_LAYERS = null;
        }
    }

    /**
     * Initializes the engine (equivalent to VulkanApp.initVulkan()  [cite: 3, 294-306]).
     */
    public void init() {
        System.out.println("LOG (VRT): Initializing Vulkan...");
        if (!glfwInit()) {
            throw new RuntimeException("Failed to initialize GLFW!");
        }
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();

        // Create other resources required by the pipeline
        try (MemoryStack stack = stackPush()) {
            createComputeImage(stack);
            createStagingBuffer(stack);
            // Scene-specific buffers (triangles, BVH, etc.) will be created by uploadAndSwapScene
            createComputePipelineResources(stack);
            createCommandPoolAndBuffer(stack);
            createFence(stack);
        }
        System.out.println("LOG (VRT): VulkanRenderer initialized successfully.");
    }

    /**
     * Takes new CPU data (BuiltCpuData), uploads it to the GPU,
     * and returns a new GpuSceneData instance.
     *
     * IMPORTANT: This method does NOT destroy the old GpuSceneData.
     * That is the responsibility of the RenderLoop.
     */
    public GpuSceneData uploadAndSwapScene(BuiltCpuData cpuData) {
        System.out.println("LOG (VRT): Uploading new scene data to GPU...");

        // Moved here from RenderLoop.checkQueues().
        // Ensure the GPU is idle before uploading new buffers.
        vkDeviceWaitIdle(device);

        try (MemoryStack stack = stackPush()) {
            long triangleBuffer, triangleBufferMemory;
            long materialBuffer, materialBufferMemory;
            long bvhBuffer, bvhBufferMemory;

            // 1. Upload Triangle Buffer
            if (cpuData.modelVertexData == null) throw new RuntimeException("Model vertex data was not prepared!");
            long triBufferSize = cpuData.modelVertexData.remaining() * 4L;
            LongBuffer pTriBuffer = stack.mallocLong(1);
            LongBuffer pTriMemory = stack.mallocLong(1);
            createHostVisibleBuffer(
                    stack,
                    triBufferSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    memByteBuffer(cpuData.modelVertexData),
                    pTriBuffer,
                    pTriMemory
            );
            triangleBuffer = pTriBuffer.get(0);
            triangleBufferMemory = pTriMemory.get(0);
            memFree(cpuData.modelVertexData); // Release CPU-side data

            // 2. Upload Material Buffer
            if (cpuData.modelMaterialData == null) throw new RuntimeException("Model material data was not prepared!");
            long matBufferSize = cpuData.modelMaterialData.remaining() * 4L;
            LongBuffer pMatBuffer = stack.mallocLong(1);
            LongBuffer pMatMemory = stack.mallocLong(1);
            createHostVisibleBuffer(
                    stack,
                    matBufferSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    memByteBuffer(cpuData.modelMaterialData),
                    pMatBuffer,
                    pMatMemory
            );
            materialBuffer = pMatBuffer.get(0);
            materialBufferMemory = pMatMemory.get(0);
            memFree(cpuData.modelMaterialData); // Release CPU-side data

            // 3. Upload BVH Buffer
            if (cpuData.flatBvhData == null) throw new RuntimeException("BVH data was not flattened!");
            long bvhBufferSize = cpuData.flatBvhData.remaining();
            LongBuffer pBvhBuffer = stack.mallocLong(1);
            LongBuffer pBvhMemory = stack.mallocLong(1);
            createHostVisibleBuffer(
                    stack,
                    bvhBufferSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    cpuData.flatBvhData,
                    pBvhBuffer,
                    pBvhMemory
            );
            bvhBuffer = pBvhBuffer.get(0);
            bvhBufferMemory = pBvhMemory.get(0);
            // flatBvhData (ByteBuffer) does NOT require memFree

            // 4. Update the descriptor set with the new buffers
            updateDescriptorSet(stack, triangleBuffer, materialBuffer, bvhBuffer);

            System.out.println("LOG (VRT): New scene uploaded. Triangles: " + cpuData.triangleCount);

            return new GpuSceneData(
                    triangleBuffer, triangleBufferMemory,
                    materialBuffer, materialBufferMemory,
                    bvhBuffer, bvhBufferMemory,
                    cpuData.triangleCount
            );
        }
    }

    /**
     * Main render call on the VRT.
     * Renders one frame using the given scene data.
     * Returns a ByteBuffer that contains pixel data.
     */
    public ByteBuffer renderFrame(GpuSceneData sceneData) {
        try (MemoryStack stack = stackPush()) {
            // Record commands (equivalent to VulkanApp.recordComputeCommands  [cite: 3, 853-977])
            recordComputeCommands(stack, sceneData.triangleBuffer, sceneData.materialBuffer, sceneData.bvhBuffer, sceneData.triangleCount);

            // Submit commands (equivalent to VulkanApp.submitCommands  [cite: 3, 979-992])
            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(commandBuffer));

            vkResetFences(device, fence); // Reset the fence

            vkQueueSubmit(computeQueue, submitInfo, fence);

            vkWaitForFences(device, fence, true, Long.MAX_VALUE);

            // Read image (equivalent to VulkanApp.saveImageToFile  [cite: 3, 994-1008])
            long bufferSize = WIDTH * HEIGHT * 4;
            PointerBuffer pData = stack.mallocPointer(1);
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, pData);
            ByteBuffer pixelData = pData.getByteBuffer(0, (int) bufferSize);

            // IMPORTANT: Make a copy; otherwise this buffer becomes invalid after unmap.
            ByteBuffer copyPixelData = ByteBuffer.allocateDirect((int) bufferSize);
            copyPixelData.put(pixelData);
            copyPixelData.flip();

            vkUnmapMemory(device, stagingBufferMemory);

            return copyPixelData;
        }
    }

    /**
     * Safely destroys old, now-unused GPU buffers.
     */
    public void destroyGpuSceneData(GpuSceneData oldData) {
        if (oldData == null) return;

        System.out.println("LOG (VRT): Destroying old GPU scene data...");
        // vkDeviceWaitIdle should already have been called in the RenderLoop
        vkDestroyBuffer(device, oldData.triangleBuffer, null);
        vkFreeMemory(device, oldData.triangleBufferMemory, null);
        vkDestroyBuffer(device, oldData.materialBuffer, null);
        vkFreeMemory(device, oldData.materialBufferMemory, null);
        vkDestroyBuffer(device, oldData.bvhBuffer, null);
        vkFreeMemory(device, oldData.bvhBufferMemory, null);
    }

    /**
     * Shuts down the engine (equivalent to VulkanApp.cleanup  [cite: 3, 1060-1135]).
     */
    public void destroy() {
        System.out.println("LOG (VRT): Cleaning up VulkanRenderer...");
        if (device != null) {
            try {
                vkDeviceWaitIdle(device);
            } catch (Exception e) {
                System.err.println("WARN (VRT): Exception during vkDeviceWaitIdle: " + e.getMessage());
            }
        }

        // GpuSceneData (triangle/material/bvh buffers) should be cleaned up in the RenderLoop

        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, null);
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, null);
        }
        if (computePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, computePipeline, null);
        }
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, null);
        }
        if (computeShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, computeShaderModule, null);
        }
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, null);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        }

        if (computeImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, computeImageView, null);
        }
        if (computeImage != VK_NULL_HANDLE) {
            vkDestroyImage(device, computeImage, null);
        }
        if (computeImageMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, computeImageMemory, null);
        }
        if (stagingBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, stagingBuffer, null);
        }
        if (stagingBufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, stagingBufferMemory, null);
        }

        if (device != null) {
            vkDestroyDevice(device, null);
        }
        if (debugMessenger != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        }
        if (instance != null) {
            vkDestroyInstance(instance, null);
        }
        glfwTerminate();
        System.out.println("LOG (VRT): VulkanRenderer destroyed.");
    }


    // --- 1. Initialization Methods (moved over from VulkanApp) ---
    // ... (createInstance, setupDebugMessenger, pickPhysicalDevice, etc.) ...

    private void createInstance() {
        try (MemoryStack stack = stackPush()) {
            VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                    .pApplicationName(memASCII("Vulkan Ray Tracer"))
                    .apiVersion(VK_API_VERSION_1_0);
            VkInstanceCreateInfo createInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                    .pApplicationInfo(appInfo);
            PointerBuffer glfwExtensions = glfwGetRequiredInstanceExtensions();
            if (glfwExtensions == null) {
                throw new RuntimeException("Failed to find required GLFW extensions!");
            }
            PointerBuffer requiredExtensions;
            if (ENABLE_VALIDATION_LAYERS) {
                requiredExtensions = stack.mallocPointer(glfwExtensions.capacity() + 1);
                requiredExtensions.put(glfwExtensions);
                requiredExtensions.put(memASCII(VK_EXT_DEBUG_UTILS_EXTENSION_NAME));
            } else {
                requiredExtensions = glfwExtensions;
            }
            requiredExtensions.flip();
            createInfo.ppEnabledExtensionNames(requiredExtensions);
            if (ENABLE_VALIDATION_LAYERS) {
                createInfo.ppEnabledLayerNames(VALIDATION_LAYERS);
                VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack);
                populateDebugMessengerCreateInfo(debugCreateInfo);
                createInfo.pNext(debugCreateInfo.address());
            }
            PointerBuffer pInstance = stack.mallocPointer(1);
            if (vkCreateInstance(createInfo, null, pInstance) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create Vulkan Instance!");
            }
            instance = new VkInstance(pInstance.get(0), createInfo);
        }
    }
    private void setupDebugMessenger() {
        if (!ENABLE_VALIDATION_LAYERS) return;
        try (MemoryStack stack = stackPush()) {
            VkDebugUtilsMessengerCreateInfoEXT createInfo = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack);
            populateDebugMessengerCreateInfo(createInfo);
            LongBuffer pDebugMessenger = stack.mallocLong(1);
            if (vkCreateDebugUtilsMessengerEXT(instance, createInfo, null, pDebugMessenger) != VK_SUCCESS) {
                throw new RuntimeException("Failed to set up debug messenger!");
            }
            debugMessenger = pDebugMessenger.get(0);
        }
    }
    private void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT createInfo) {
        createInfo.sType(VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
        createInfo.messageSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
        createInfo.messageType(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
        createInfo.pfnUserCallback((messageSeverity, messageType, pCallbackData, pUserData) -> {
            VkDebugUtilsMessengerCallbackDataEXT callbackData = VkDebugUtilsMessengerCallbackDataEXT.create(pCallbackData);
            System.err.println("[VULKAN VALIDATION]: " + callbackData.pMessageString());
            return VK_FALSE;
        });
    }
    private void pickPhysicalDevice() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer deviceCount = stack.ints(0);
            vkEnumeratePhysicalDevices(instance, deviceCount, null);
            if (deviceCount.get(0) == 0) {
                throw new RuntimeException("Failed to find GPUs with Vulkan support!");
            }
            PointerBuffer pPhysicalDevices = stack.mallocPointer(deviceCount.get(0));
            vkEnumeratePhysicalDevices(instance, deviceCount, pPhysicalDevices);
            for (int i = 0; i < pPhysicalDevices.capacity(); i++) {
                VkPhysicalDevice device = new VkPhysicalDevice(pPhysicalDevices.get(i), instance);
                if (isDeviceSuitable(device)) {
                    physicalDevice = device;
                    break;
                }
            }
            if (physicalDevice == null) {
                throw new RuntimeException("Failed to find a suitable GPU!");
            }
        }
    }
    private boolean isDeviceSuitable(VkPhysicalDevice device) {
        computeQueueFamilyIndex = findComputeQueueFamily(device);
        return computeQueueFamilyIndex != -1;
    }
    private int findComputeQueueFamily(VkPhysicalDevice device) {
        try (MemoryStack stack = stackPush()) {
            IntBuffer queueFamilyCount = stack.ints(0);
            vkGetPhysicalDeviceQueueFamilyProperties(device, queueFamilyCount, null);
            VkQueueFamilyProperties.Buffer queueFamilies = VkQueueFamilyProperties.malloc(queueFamilyCount.get(0), stack);
            vkGetPhysicalDeviceQueueFamilyProperties(device, queueFamilyCount, queueFamilies);
            for (int i = 0; i < queueFamilies.capacity(); i++) {
                if ((queueFamilies.get(i).queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) {
                    return i;
                }
            }
            return -1;
        }
    }
    private void createLogicalDevice() {
        try (MemoryStack stack = stackPush()) {
            FloatBuffer pQueuePriorities = stack.floats(1.0f);
            VkDeviceQueueCreateInfo.Buffer queueCreateInfo = VkDeviceQueueCreateInfo.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                    .queueFamilyIndex(computeQueueFamilyIndex)
                    .pQueuePriorities(pQueuePriorities);
            VkPhysicalDeviceFeatures deviceFeatures = VkPhysicalDeviceFeatures.calloc(stack);
            VkDeviceCreateInfo createInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                    .pQueueCreateInfos(queueCreateInfo)
                    .pEnabledFeatures(deviceFeatures);
            if (ENABLE_VALIDATION_LAYERS) {
                createInfo.ppEnabledLayerNames(VALIDATION_LAYERS);
            }
            PointerBuffer pDevice = stack.mallocPointer(1);
            if (vkCreateDevice(physicalDevice, createInfo, null, pDevice) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create logical device!");
            }
            device = new VkDevice(pDevice.get(0), physicalDevice, createInfo);
            PointerBuffer pQueue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, pQueue);
            computeQueue = new VkQueue(pQueue.get(0), device);
        }
    }


    // --- 2. Resource Creation Methods (moved over from VulkanApp) ---
    // ... (createComputeImage, createStagingBuffer, etc.) ...

    private void createComputeImage(MemoryStack stack) {
        // Unchanged from VulkanApp.java:502  [cite: 3, 502-540]
        VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .extent(e -> e.width(WIDTH).height(HEIGHT).depth(1))
                .mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        LongBuffer pImage = stack.mallocLong(1);
        vkCreateImage(device, imageInfo, null, pImage);
        computeImage = pImage.get(0);
        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, computeImage, memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        LongBuffer pImageMemory = stack.mallocLong(1);
        vkAllocateMemory(device, allocInfo, null, pImageMemory);
        computeImageMemory = pImageMemory.get(0);
        vkBindImageMemory(device, computeImage, computeImageMemory, 0);
        VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(computeImage).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        LongBuffer pImageView = stack.mallocLong(1);
        vkCreateImageView(device, viewInfo, null, pImageView);
        computeImageView = pImageView.get(0);
    }

    private void createStagingBuffer(MemoryStack stack) {
        // Unchanged from VulkanApp.java:542  [cite: 3, 542-566]
        long bufferSize = WIDTH * HEIGHT * 4;
        VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(bufferSize)
                .usage(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        LongBuffer pBuffer = stack.mallocLong(1);
        vkCreateBuffer(device, bufferInfo, null, pBuffer);
        stagingBuffer = pBuffer.get(0);
        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, stagingBuffer, memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits(),
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
        LongBuffer pBufferMemory = stack.mallocLong(1);
        vkAllocateMemory(device, allocInfo, null, pBufferMemory);
        stagingBufferMemory = pBufferMemory.get(0);
        vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);
    }

    // createTriangleBuffer, createMaterialBuffer, createBvhBuffer
    // were moved into uploadAndSwapScene.

    private void createHostVisibleBuffer(MemoryStack stack, long bufferSize, int usage,
                                         ByteBuffer data, // Source data (ByteBuffer)
                                         LongBuffer pBuffer, LongBuffer pMemory) {
        // Unchanged from VulkanApp.java:676  [cite: 3, 676-696]
        createBuffer(stack, bufferSize, usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                pBuffer, pMemory);

        long memory = pMemory.get(0);

        PointerBuffer pData = stack.mallocPointer(1);
        vkMapMemory(device, memory, 0, bufferSize, 0, pData);
        memCopy(memAddress(data), pData.get(0), bufferSize);
        vkUnmapMemory(device, memory);
    }

    private void createBuffer(MemoryStack stack, long size, int usage, int properties,
                              LongBuffer pBuffer, LongBuffer pMemory) {
        // Unchanged from VulkanApp.java:708  [cite: 3, 708-732]
        VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(size)
                .usage(usage)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        if (vkCreateBuffer(device, bufferInfo, null, pBuffer) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create buffer!");
        }

        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, pBuffer.get(0), memRequirements);

        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits(), properties));

        if (vkAllocateMemory(device, allocInfo, null, pMemory) != VK_SUCCESS) {
            throw new RuntimeException("Failed to allocate buffer memory!");
        }
        vkBindBufferMemory(device, pBuffer.get(0), pMemory.get(0), 0);
    }

    /**
     * Creates the Pipeline, Layout and Descriptor Pool.
     * These resources do not change when the scene changes.
     */
    private void createComputePipelineResources(MemoryStack stack) {
        // Based on VulkanApp.java:745  [cite: 3, 745-848], but descriptor set allocation/update moved.

        // --- Create Descriptor Set Layout (4 bindings) ---
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(4, stack);
        bindings.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(2).binding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(3).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(bindings);
        LongBuffer pSetLayout = stack.mallocLong(1);
        vkCreateDescriptorSetLayout(device, layoutInfo, null, pSetLayout);
        descriptorSetLayout = pSetLayout.get(0);

        // --- Create Pipeline Layout ---
        VkPushConstantRange.Buffer pushConstantRange = VkPushConstantRange.calloc(1, stack)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(4); // int numTriangles
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pSetLayouts(stack.longs(descriptorSetLayout))
                .pPushConstantRanges(pushConstantRange);
        LongBuffer pPipelineLayout = stack.mallocLong(1);
        vkCreatePipelineLayout(device, pipelineLayoutInfo, null, pPipelineLayout);
        pipelineLayout = pPipelineLayout.get(0);

        // --- Create Pipeline ---
        try {
            computeShaderModule = createShaderModule(loadShaderFromFile(SHADER_PATH));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load shader: " + e.getMessage());
        }
        VkPipelineShaderStageCreateInfo shaderStageInfo = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                .module(computeShaderModule)
                .pName(memASCII("main"));
        VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                .stage(shaderStageInfo)
                .layout(pipelineLayout);
        LongBuffer pComputePipeline = stack.mallocLong(1);
        vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pComputePipeline);
        computePipeline = pComputePipeline.get(0);

        // --- Create Descriptor Pool ---
        VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
        // Slightly larger to handle multiple scene swaps (e.g., 10 sets-worth of buffers)
        poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(3 * 10);

        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                .flags(VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT) // May be needed
                .pPoolSizes(poolSizes)
                .maxSets(10); // Allow up to 10 scene changes
        LongBuffer pDescriptorPool = stack.mallocLong(1);
        vkCreateDescriptorPool(device, poolInfo, null, pDescriptorPool);
        descriptorPool = pDescriptorPool.get(0);

        // --- Allocate Descriptor Set ---
        // The descriptor set is allocated here, but updated later in updateDescriptorSet().
        VkDescriptorSetAllocateInfo allocSetInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pDescriptorSet = stack.mallocLong(1);
        vkAllocateDescriptorSets(device, allocSetInfo, pDescriptorSet);
        descriptorSet = pDescriptorSet.get(0);
    }

    /**
     * Updates the allocated main descriptor set with NEW buffers.
     */
    private void updateDescriptorSet(MemoryStack stack, long triangleBuffer, long materialBuffer, long bvhBuffer) {
        VkDescriptorImageInfo.Buffer imageDescriptor = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(computeImageView);
        VkDescriptorBufferInfo.Buffer triangleBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(triangleBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer materialBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(materialBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer bvhBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(bvhBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkWriteDescriptorSet.Buffer descriptorWrites = VkWriteDescriptorSet.calloc(4, stack);

        descriptorWrites.get(0).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).pImageInfo(imageDescriptor);
        descriptorWrites.get(1).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(triangleBufferDescriptor);
        descriptorWrites.get(2).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(materialBufferDescriptor);
        descriptorWrites.get(3).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(bvhBufferDescriptor);

        vkUpdateDescriptorSets(device, descriptorWrites, null);
    }


    private void createCommandPoolAndBuffer(MemoryStack stack) {
        // Unchanged from VulkanApp.java:853  [cite: 3, 853-868]
        VkCommandPoolCreateInfo cmdPoolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(computeQueueFamilyIndex);
        LongBuffer pCmdPool = stack.mallocLong(1);
        vkCreateCommandPool(device, cmdPoolInfo, null, pCmdPool);
        commandPool = pCmdPool.get(0);
        VkCommandBufferAllocateInfo cmdAllocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
        PointerBuffer pCmdBuffer = stack.mallocPointer(1);
        vkAllocateCommandBuffers(device, cmdAllocInfo, pCmdBuffer);
        commandBuffer = new VkCommandBuffer(pCmdBuffer.get(0), device);
    }

    private void createFence(MemoryStack stack) {
        LongBuffer pFence = stack.mallocLong(1);
        VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack).sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        if (vkCreateFence(device, fenceInfo, null, pFence) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create fence");
        }
        fence = pFence.get(0);
    }

    /**
     * Called inside renderFrame().
     */
    private void recordComputeCommands(MemoryStack stack, long triangleBuffer, long materialBuffer, long bvhBuffer, int triangleCount) {
        // Based on VulkanApp.java:873  [cite: 3, 873-977]

        vkResetCommandBuffer(commandBuffer, 0); // Reset the command buffer

        VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        vkBeginCommandBuffer(commandBuffer, beginInfo);

        // 1. Barrier: ensure the image is in GENERAL layout before compute writes
        VkImageMemoryBarrier.Buffer imageBarrier1 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(0).dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_GENERAL)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier1.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        // 1b. No buffer barriers needed here: descriptor set was just updated
        // and we guarantee the host (CPU) writes have completed before submission.

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, null, null, imageBarrier1);

        // 2. Bind pipeline and descriptors
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(descriptorSet), null);

        // 3. Push Constant (number of triangles)
        ByteBuffer pushConstantData = stack.malloc(4);
        pushConstantData.putInt(0, triangleCount);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantData);

        // 4. Dispatch
        vkCmdDispatch(commandBuffer, (WIDTH + 7) / 8, (HEIGHT + 7) / 8, 1);

        // 5. Barrier: GENERAL -> TRANSFER_SRC_OPTIMAL (for copying to buffer)
        VkImageMemoryBarrier.Buffer imageBarrier2 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_GENERAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier2.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, null, null, imageBarrier2);

        // 6. Copy the VkImage (VRAM) to the VkBuffer (CPU-visible)
        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                .bufferOffset(0)
                .bufferRowLength(0).bufferImageHeight(0)
                .imageSubresource(s -> s.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1))
                .imageOffset(o -> o.x(0).y(0).z(0))
                .imageExtent(e -> e.width(WIDTH).height(HEIGHT).depth(1));

        vkCmdCopyImageToBuffer(commandBuffer, computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, region);

        // 7. Barrier: TRANSFER_SRC_OPTIMAL -> GENERAL (prepare for next frame)
        VkImageMemoryBarrier.Buffer imageBarrier3 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_TRANSFER_READ_BIT).dstAccessMask(0) // Next frame starts at top-of-pipe
                .oldLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier3.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                0, null, null, imageBarrier3);

        vkEndCommandBuffer(commandBuffer);
    }


    // --- 3. Helper Methods (moved over from VulkanApp) ---
    // ... (findMemoryType, loadShaderFromFile, createShaderModule) ...

    private int findMemoryType(int typeFilter, int properties) {
        // Unchanged from VulkanApp.java:1010  [cite: 3, 1010-1021]
        try (MemoryStack stack = stackPush()) {
            VkPhysicalDeviceMemoryProperties memProperties = VkPhysicalDeviceMemoryProperties.malloc(stack);
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, memProperties);
            for (int i = 0; i < memProperties.memoryTypeCount(); i++) {
                if ((typeFilter & (1 << i)) != 0 && (memProperties.memoryTypes(i).propertyFlags() & properties) == properties) {
                    return i;
                }
            }
        }
        throw new RuntimeException("Failed to find suitable memory type!");
    }
    private ByteBuffer loadShaderFromFile(String filepath) throws IOException {
        // Unchanged from VulkanApp.java:1022  [cite: 3, 1022-1031]
        try (FileInputStream fis = new FileInputStream(filepath);
             FileChannel fc = fis.getChannel()) {
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) fc.size());
            while (fc.read(buffer) > 0) ;
            return buffer.flip();
        }
    }
    private long createShaderModule(ByteBuffer spirvCode) {
        // Unchanged from VulkanApp.java:1032  [cite: 3, 1032-1044]
        try (MemoryStack stack = stackPush()) {
            VkShaderModuleCreateInfo createInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                    .pCode(spirvCode);
            LongBuffer pShaderModule = stack.mallocLong(1);
            if (vkCreateShaderModule(device, createInfo, null, pShaderModule) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create shader module!");
            }
            return pShaderModule.get(0);
        }
    }
}
