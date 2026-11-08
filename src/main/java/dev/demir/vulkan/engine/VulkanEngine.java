package dev.demir.vulkan.engine;

import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.scene.Camera;
import dev.demir.vulkan.scene.SceneDescription;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static dev.demir.vulkan.engine.GpuMemory.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memByteBuffer;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.VK10.*;

/**
 * Core Vulkan engine, running on its own thread (the Vulkan Render Thread, VRT).
 *
 * Renders with hardware ray tracing: a compute shader traces rays through the scene's
 * acceleration structures ({@link RayTracingScene}) with ray queries, writes the image,
 * and the frame is copied back to the CPU for the Swing UI.
 *
 * Other threads talk to it only through the submit* methods (thread-safe queues);
 * all Vulkan calls happen on the VRT. Camera / accumulation state is owned by the UI thread.
 */
public class VulkanEngine implements Runnable {

    // --- Configuration ---
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final int CAMERA_UBO_SIZE = 80; // 4 x vec3 (16-byte aligned) + 2 ints, std140

    // SPIR-V compiled at build time from src/main/shaders by the Gradle task 'compileShaders'.
    private static final String SHADER_RESOURCE = "/shaders/pathtrace.comp.spv";

    // Validation is opt-in: run with -Dvulkan.validation=true. The layer ships with the
    // Vulkan SDK, not with GPU drivers, so if it is missing we warn and run without it.
    private static final String VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    private boolean validationEnabled = false;

    // Hardware ray tracing (RT cores): acceleration structures + ray queries from the compute shader.
    private static final String[] REQUIRED_DEVICE_EXTENSIONS = {
            KHRAccelerationStructure.VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            KHRRayQuery.VK_KHR_RAY_QUERY_EXTENSION_NAME,
            KHRDeferredHostOperations.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by acceleration_structure
    };

    // Descriptor bindings (must match pathtrace.comp)
    private static final int BINDING_IMAGE = 0;
    private static final int BINDING_TLAS = 1;
    private static final int BINDING_INSTANCES = 2;
    private static final int BINDING_CAMERA = 3;

    // --- Threading & State ---
    private volatile boolean isRunning = true;
    private final Thread thread;
    private final Consumer<Throwable> onFatalError; // Called on the VRT if it crashes

    // --- Thread-Safe Queues ---
    private final AtomicReference<FrameData> frameQueue;                // VRT -> UI: latest completed frame
    private final ConcurrentLinkedQueue<SceneDescription> sceneQueue;   // UI/SRT -> VRT: new scene
    private final ConcurrentLinkedQueue<Camera> cameraQueue;            // UI -> VRT: camera updates
    private final ConcurrentLinkedQueue<Boolean> skyToggleQueue;        // UI -> VRT: sky on/off

    // --- Core Vulkan Objects ---
    private VkInstance instance;
    private long debugMessenger = VK_NULL_HANDLE;
    private VkPhysicalDevice physicalDevice;
    private VkDevice device;
    private VkQueue computeQueue;
    private int computeQueueFamilyIndex = -1;
    private GpuMemory gpuMemory;

    // --- Render target and readback ---
    private long computeImage = VK_NULL_HANDLE;
    private long computeImageMemory = VK_NULL_HANDLE;
    private long computeImageView = VK_NULL_HANDLE;
    private GpuBuffer readbackBuffer;

    // --- Camera Uniform Buffer Object (persistently mapped) ---
    private GpuBuffer cameraUbo;

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

    // --- Current State (Managed by VRT) ---
    private RayTracingScene rtScene;
    private boolean sceneLoaded = false;
    private Camera currentCamera = null; // latest camera state sent by the UI thread
    private int isSkyEnabled = 1; // 1 = true (default ON), 0 = false

    /**
     * Constructor. Prepares the engine and its communication channels.
     * @param frameQueue Thread-safe reference used to publish completed frames to the UI thread.
     * @param onFatalError Called on the VRT if initialization or rendering fails; the engine stops afterwards.
     */
    public VulkanEngine(AtomicReference<FrameData> frameQueue, Consumer<Throwable> onFatalError) {
        this.frameQueue = frameQueue;
        this.onFatalError = onFatalError;
        this.sceneQueue = new ConcurrentLinkedQueue<>();
        this.cameraQueue = new ConcurrentLinkedQueue<>();
        this.skyToggleQueue = new ConcurrentLinkedQueue<>();
        this.thread = new Thread(this, "Vulkan-Engine-Thread");
        this.thread.setDaemon(true);
    }

    // --- PUBLIC API (Called by other threads) ---

    /** Starts the Vulkan Render Thread (VRT). */
    public void start() {
        this.isRunning = true;
        this.thread.start();
    }

    /** Sends a stop signal to the VRT and waits for shutdown. */
    public void stop() {
        this.isRunning = false;
        try {
            this.thread.join(5000); // 5s graceful shutdown window
        } catch (InterruptedException e) {
            System.err.println("WARN (UI): Interrupted while stopping VulkanEngine.");
            Thread.currentThread().interrupt();
        }
    }

    /** Queues a new scene; the VRT swaps to it before its next frame. */
    public void submitScene(SceneDescription scene) {
        System.out.println("LOG (VRT-QUEUE): Received submitScene() command.");
        sceneQueue.add(scene);
    }

    /** Queues the latest camera state. */
    public void submitCameraUpdate(Camera camera) {
        cameraQueue.add(camera);
    }

    /** Queues a sky light on/off change. */
    public void submitSkyToggle(boolean isSkyOn) {
        System.out.println("LOG (VRT-QUEUE): Received submitSkyToggle(" + isSkyOn + ") command.");
        skyToggleQueue.add(isSkyOn);
    }

    // --- VULKAN RENDER THREAD (VRT) LOGIC ---

    @Override
    public void run() {
        try {
            initVulkan();
            mainLoop();
        } catch (Throwable t) { // Throwable: LWJGL reports e.g. stack overflow as an Error
            System.err.println("FATAL (VRT): VulkanEngine thread crashed!");
            t.printStackTrace();
            isRunning = false;
            onFatalError.accept(t);
        } finally {
            cleanup();
            System.out.println("LOG (VRT-LIFECYCLE): VulkanEngine shut down.");
        }
    }

    private void initVulkan() {
        System.out.println("LOG (VRT-LIFECYCLE): Starting VulkanEngine initialization...");
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();
        gpuMemory = new GpuMemory(device, computeQueue, computeQueueFamilyIndex);
        rtScene = new RayTracingScene(device, gpuMemory);

        try (MemoryStack stack = stackPush()) {
            createCommandPoolAndBuffer(stack);
            createFence(stack);
            createDescriptorSetLayout(stack);
            createPipelineLayout(stack);
            createComputePipeline(stack);
            createComputeImage(stack);
            createReadbackBuffer();
            createCameraUbo();
            createDescriptorPool(stack);
            allocateDescriptorSet(stack);
            transitionImageToGeneral();
        }
        System.out.println("LOG (VRT-LIFECYCLE): VulkanEngine initialized successfully.");
    }

    private void mainLoop() {
        while (isRunning) {
            handleCommands();

            if (!sceneLoaded || currentCamera == null) {
                sleep(16);
                continue;
            }

            internalUpdateCameraUBO();
            FrameData frame = renderFrame();
            frameQueue.set(frame);
        }
    }

    /**
     * Processes pending commands. Camera and sky queues are drained so only the latest state is used.
     */
    private void handleCommands() {
        SceneDescription newScene = sceneQueue.poll();
        if (newScene != null) {
            internalSwapScene(newScene);
        }

        Camera newCamera = null;
        Camera polledCamera;
        while ((polledCamera = cameraQueue.poll()) != null) {
            newCamera = polledCamera;
        }
        if (newCamera != null) {
            this.currentCamera = newCamera;
        }

        Boolean skyState = null;
        Boolean polledSkyState;
        while ((polledSkyState = skyToggleQueue.poll()) != null) {
            skyState = polledSkyState;
        }
        if (skyState != null) {
            int newSkyState = skyState ? 1 : 0;
            if (this.isSkyEnabled != newSkyState) {
                System.out.println("LOG (VRT-HANDLE): Sky state set to: " + newSkyState);
                this.isSkyEnabled = newSkyState;
            }
        }
    }

    /**
     * Makes the GPU scene match the description and points the descriptor set at it.
     */
    private void internalSwapScene(SceneDescription scene) {
        vkDeviceWaitIdle(device); // resources of the previous scene may still be in use
        rtScene.update(scene);
        try (MemoryStack stack = stackPush()) {
            updateDescriptorSet(stack);
        }
        sceneLoaded = true;
    }

    /** Writes the camera into the persistently mapped UBO (std140 layout, see CAMERA_UBO_SIZE). */
    private void internalUpdateCameraUBO() {
        ByteBuffer ubo = cameraUbo.mapped();
        currentCamera.getOrigin().store(0, ubo);
        currentCamera.getLowerLeft().store(16, ubo);
        currentCamera.getHorizontal().store(32, ubo);
        currentCamera.getVertical().store(48, ubo);
        ubo.putInt(64, currentCamera.getFrameCount());
        ubo.putInt(68, this.isSkyEnabled);
    }

    /** Renders one frame and returns a copy of its pixels (RGBA8). */
    private FrameData renderFrame() {
        try (MemoryStack stack = stackPush()) {
            recordComputeCommands(stack);

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType$Default()
                    .pCommandBuffers(stack.pointers(commandBuffer));
            vkResetFences(device, fence);
            check(vkQueueSubmit(computeQueue, submitInfo, fence), "vkQueueSubmit");
            vkWaitForFences(device, fence, true, Long.MAX_VALUE);

            // Copy out of the mapped readback buffer; it is overwritten by the next frame.
            int bufferSize = WIDTH * HEIGHT * 4;
            ByteBuffer copyPixelData = ByteBuffer.allocateDirect(bufferSize);
            copyPixelData.put(memByteBuffer(readbackBuffer.mappedAddress(), bufferSize));
            copyPixelData.flip();
            return new FrameData(copyPixelData);
        }
    }

    /** Records: dispatch the ray tracing shader, then copy the image into the readback buffer. */
    private void recordComputeCommands(MemoryStack stack) {
        vkResetCommandBuffer(commandBuffer, 0);
        vkBeginCommandBuffer(commandBuffer, VkCommandBufferBeginInfo.calloc(stack)
                .sType$Default()
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));

        // Barrier 1: previous frame's transfer read must finish before this frame's shader write.
        imageBarrier(stack, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                stack.longs(descriptorSet), null);
        vkCmdDispatch(commandBuffer, (WIDTH + 7) / 8, (HEIGHT + 7) / 8, 1);

        // Barrier 2: shader write -> transfer read, GENERAL -> TRANSFER_SRC_OPTIMAL.
        imageBarrier(stack, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                .bufferOffset(0)
                .bufferRowLength(0).bufferImageHeight(0)
                .imageSubresource(s -> s.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1))
                .imageOffset(o -> o.x(0).y(0).z(0))
                .imageExtent(e -> e.width(WIDTH).height(HEIGHT).depth(1));
        vkCmdCopyImageToBuffer(commandBuffer, computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                readbackBuffer.buffer(), region);

        // Barrier 3: back to GENERAL for the next frame.
        imageBarrier(stack, VK_ACCESS_TRANSFER_READ_BIT, 0,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    private void imageBarrier(MemoryStack stack, int srcAccess, int dstAccess, int oldLayout, int newLayout,
                              int srcStage, int dstStage) {
        VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                .sType$Default()
                .srcAccessMask(srcAccess)
                .dstAccessMask(dstAccess)
                .oldLayout(oldLayout)
                .newLayout(newLayout)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                        .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, null, null, barrier);
    }

    private void cleanup() {
        System.out.println("LOG (VRT-LIFECYCLE): Cleaning up VulkanEngine resources...");
        if (device != null) {
            try {
                vkDeviceWaitIdle(device);
            } catch (Exception e) {
                System.err.println("WARN (VRT-LIFECYCLE): Exception during vkDeviceWaitIdle: " + e.getMessage());
            }
        }
        if (rtScene != null) rtScene.destroy();
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, null);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, null);
        if (computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, computePipeline, null);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, null);
        if (computeShaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, computeShaderModule, null);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, null);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        if (computeImageView != VK_NULL_HANDLE) vkDestroyImageView(device, computeImageView, null);
        if (computeImage != VK_NULL_HANDLE) vkDestroyImage(device, computeImage, null);
        if (computeImageMemory != VK_NULL_HANDLE) vkFreeMemory(device, computeImageMemory, null);
        if (gpuMemory != null) {
            gpuMemory.destroy(readbackBuffer);
            gpuMemory.destroy(cameraUbo);
            gpuMemory.destroy();
        }
        if (device != null) vkDestroyDevice(device, null);
        if (debugMessenger != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        if (instance != null) vkDestroyInstance(instance, null);
    }

    // --- Instance and device setup ---

    /**
     * Creates the Vulkan instance. The renderer is compute-only and reads frames back to the CPU,
     * so no window-system (surface) extensions are needed; only debug utils when validating.
     */
    private void createInstance() {
        validationEnabled = Boolean.getBoolean("vulkan.validation") && isValidationLayerAvailable();
        System.out.println("LOG (VRT-LIFECYCLE): Validation layers " + (validationEnabled ? "ENABLED" : "disabled")
                + " (enable with -Dvulkan.validation=true).");

        try (MemoryStack stack = stackPush()) {
            VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
                    .sType$Default()
                    .pApplicationName(stack.UTF8("Vulkan Ray Tracer"))
                    .apiVersion(VK12.VK_API_VERSION_1_2); // 1.2: buffer device address, needed for ray tracing
            VkInstanceCreateInfo createInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pApplicationInfo(appInfo);
            if (validationEnabled) {
                createInfo.ppEnabledExtensionNames(stack.pointers(stack.UTF8(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)));
                createInfo.ppEnabledLayerNames(stack.pointers(stack.UTF8(VALIDATION_LAYER_NAME)));
                VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack);
                populateDebugMessengerCreateInfo(debugCreateInfo);
                createInfo.pNext(debugCreateInfo.address());
            }
            PointerBuffer pInstance = stack.mallocPointer(1);
            int result = vkCreateInstance(createInfo, null, pInstance);
            if (result != VK_SUCCESS) {
                throw new RuntimeException("Failed to create Vulkan Instance! VkResult = " + result);
            }
            instance = new VkInstance(pInstance.get(0), createInfo);
        }
    }

    /** Returns true if the Khronos validation layer is installed; logs a warning otherwise. */
    private static boolean isValidationLayerAvailable() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer layerCount = stack.ints(0);
            vkEnumerateInstanceLayerProperties(layerCount, null);
            VkLayerProperties.Buffer layers = VkLayerProperties.malloc(layerCount.get(0), stack);
            vkEnumerateInstanceLayerProperties(layerCount, layers);
            for (VkLayerProperties layer : layers) {
                if (VALIDATION_LAYER_NAME.equals(layer.layerNameString())) {
                    return true;
                }
            }
        }
        System.err.println("WARN (VRT-LIFECYCLE): Validation requested, but " + VALIDATION_LAYER_NAME
                + " is not installed (it comes with the Vulkan SDK). Continuing without validation.");
        return false;
    }

    private void setupDebugMessenger() {
        if (!validationEnabled) return;
        try (MemoryStack stack = stackPush()) {
            VkDebugUtilsMessengerCreateInfoEXT createInfo = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack);
            populateDebugMessengerCreateInfo(createInfo);
            LongBuffer pDebugMessenger = stack.mallocLong(1);
            check(vkCreateDebugUtilsMessengerEXT(instance, createInfo, null, pDebugMessenger), "vkCreateDebugUtilsMessengerEXT");
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

    /**
     * Picks the GPU: it must support hardware ray tracing (see {@link #REQUIRED_DEVICE_EXTENSIONS}
     * and {@link #findMissingRequirement}); among those, a discrete GPU wins over an integrated one.
     */
    private void pickPhysicalDevice() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer deviceCount = stack.ints(0);
            vkEnumeratePhysicalDevices(instance, deviceCount, null);
            if (deviceCount.get(0) == 0) {
                throw new RuntimeException("No Vulkan-capable GPU found!");
            }
            PointerBuffer pPhysicalDevices = stack.mallocPointer(deviceCount.get(0));
            vkEnumeratePhysicalDevices(instance, deviceCount, pPhysicalDevices);

            List<String> rejected = new ArrayList<>();
            int bestScore = -1;
            String bestName = null;
            for (int i = 0; i < pPhysicalDevices.capacity(); i++) {
                VkPhysicalDevice candidate = new VkPhysicalDevice(pPhysicalDevices.get(i), instance);
                VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
                vkGetPhysicalDeviceProperties(candidate, properties);
                String name = properties.deviceNameString();

                String missing = findMissingRequirement(candidate, properties);
                if (missing != null) {
                    rejected.add(name + " (" + missing + ")");
                    System.out.println("LOG (VRT-LIFECYCLE): Skipping GPU " + name + ": " + missing);
                    continue;
                }
                int score = switch (properties.deviceType()) {
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU -> 2;
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU -> 1;
                    default -> 0;
                };
                if (score > bestScore) {
                    bestScore = score;
                    bestName = name;
                    physicalDevice = candidate;
                }
            }
            if (physicalDevice == null) {
                throw new RuntimeException("No GPU with hardware ray tracing support found. This renderer needs "
                        + "Vulkan 1.2 with VK_KHR_acceleration_structure and VK_KHR_ray_query "
                        + "(e.g. NVIDIA RTX 20+, AMD RX 6000+, Intel Arc). Checked: " + rejected);
            }
            computeQueueFamilyIndex = findComputeQueueFamily(physicalDevice);
            System.out.println("LOG (VRT-LIFECYCLE): Using GPU: " + bestName);
        }
    }

    /** Returns why the device cannot run the renderer, or null if it can. */
    private String findMissingRequirement(VkPhysicalDevice device, VkPhysicalDeviceProperties properties) {
        if (VK_API_VERSION_MAJOR(properties.apiVersion()) == 1 && VK_API_VERSION_MINOR(properties.apiVersion()) < 2) {
            return "Vulkan 1.2 not supported";
        }
        if (findComputeQueueFamily(device) == -1) {
            return "no compute queue";
        }
        try (MemoryStack stack = stackPush()) {
            IntBuffer extensionCount = stack.ints(0);
            vkEnumerateDeviceExtensionProperties(device, (String) null, extensionCount, null);
            VkExtensionProperties.Buffer extensions = VkExtensionProperties.malloc(extensionCount.get(0), stack);
            vkEnumerateDeviceExtensionProperties(device, (String) null, extensionCount, extensions);
            Set<String> available = new HashSet<>();
            for (VkExtensionProperties extension : extensions) {
                available.add(extension.extensionNameString());
            }
            for (String required : REQUIRED_DEVICE_EXTENSIONS) {
                if (!available.contains(required)) {
                    return "missing " + required;
                }
            }

            VkPhysicalDeviceRayQueryFeaturesKHR rayQuery = VkPhysicalDeviceRayQueryFeaturesKHR.calloc(stack).sType$Default();
            VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure =
                    VkPhysicalDeviceAccelerationStructureFeaturesKHR.calloc(stack).sType$Default().pNext(rayQuery.address());
            VkPhysicalDeviceVulkan12Features vulkan12 =
                    VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default().pNext(accelerationStructure.address());
            VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default().pNext(vulkan12.address());
            VK11.vkGetPhysicalDeviceFeatures2(device, features2);

            if (!rayQuery.rayQuery()) return "rayQuery feature not supported";
            if (!accelerationStructure.accelerationStructure()) return "accelerationStructure feature not supported";
            if (!vulkan12.bufferDeviceAddress()) return "bufferDeviceAddress feature not supported";
            if (!features2.features().shaderInt64()) return "shaderInt64 feature not supported";
        }
        return null;
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
                    .sType$Default()
                    .queueFamilyIndex(computeQueueFamilyIndex)
                    .pQueuePriorities(pQueuePriorities);

            // Feature chain for hardware ray tracing (all checked in findMissingRequirement).
            VkPhysicalDeviceRayQueryFeaturesKHR rayQuery = VkPhysicalDeviceRayQueryFeaturesKHR.calloc(stack)
                    .sType$Default()
                    .rayQuery(true);
            VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure =
                    VkPhysicalDeviceAccelerationStructureFeaturesKHR.calloc(stack)
                            .sType$Default()
                            .pNext(rayQuery.address())
                            .accelerationStructure(true);
            VkPhysicalDeviceVulkan12Features vulkan12 = VkPhysicalDeviceVulkan12Features.calloc(stack)
                    .sType$Default()
                    .pNext(accelerationStructure.address())
                    .bufferDeviceAddress(true);
            VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType$Default()
                    .pNext(vulkan12.address());
            features2.features().shaderInt64(true);

            PointerBuffer extensionNames = stack.mallocPointer(REQUIRED_DEVICE_EXTENSIONS.length);
            for (String extension : REQUIRED_DEVICE_EXTENSIONS) {
                extensionNames.put(stack.UTF8(extension));
            }
            extensionNames.flip();

            VkDeviceCreateInfo createInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pNext(features2.address()) // features go through pNext, so pEnabledFeatures stays null
                    .pQueueCreateInfos(queueCreateInfo)
                    .ppEnabledExtensionNames(extensionNames);
            // (Device-level layers are deprecated in Vulkan; the instance layer covers the device.)
            PointerBuffer pDevice = stack.mallocPointer(1);
            int result = vkCreateDevice(physicalDevice, createInfo, null, pDevice);
            if (result != VK_SUCCESS) {
                throw new RuntimeException("Failed to create logical device! VkResult = " + result);
            }
            device = new VkDevice(pDevice.get(0), physicalDevice, createInfo, VK12.VK_API_VERSION_1_2);
            PointerBuffer pQueue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, pQueue);
            computeQueue = new VkQueue(pQueue.get(0), device);
        }
    }

    // --- Render target, readback and UBO ---

    /** The image the shader writes (storage) and the frame is copied from (transfer source). */
    private void createComputeImage(MemoryStack stack) {
        VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                .sType$Default()
                .imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .extent(e -> e.width(WIDTH).height(HEIGHT).depth(1))
                .mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        LongBuffer pImage = stack.mallocLong(1);
        check(vkCreateImage(device, imageInfo, null, pImage), "vkCreateImage");
        computeImage = pImage.get(0);

        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, computeImage, memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType$Default()
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(gpuMemory.findMemoryType(memRequirements.memoryTypeBits(),
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0));
        LongBuffer pImageMemory = stack.mallocLong(1);
        check(vkAllocateMemory(device, allocInfo, null, pImageMemory), "vkAllocateMemory (image)");
        computeImageMemory = pImageMemory.get(0);
        vkBindImageMemory(device, computeImage, computeImageMemory, 0);

        VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                .sType$Default()
                .image(computeImage).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        LongBuffer pImageView = stack.mallocLong(1);
        check(vkCreateImageView(device, viewInfo, null, pImageView), "vkCreateImageView");
        computeImageView = pImageView.get(0);
    }

    private void createReadbackBuffer() {
        readbackBuffer = gpuMemory.createBuffer((long) WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    }

    private void createCameraUbo() {
        cameraUbo = gpuMemory.createBuffer(CAMERA_UBO_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    }

    private void transitionImageToGeneral() {
        gpuMemory.submitAndWait(cmd -> {
            try (MemoryStack stack = stackPush()) {
                VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                        .sType$Default()
                        .srcAccessMask(0)
                        .dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                        .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                        .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .image(computeImage)
                        .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                                .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, null, null, barrier);
            }
        });
    }

    // --- Pipeline and descriptors ---

    private void createDescriptorSetLayout(MemoryStack stack) {
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(4, stack);
        bindings.get(0).binding(BINDING_IMAGE).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(1).binding(BINDING_TLAS).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(2).binding(BINDING_INSTANCES).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(3).binding(BINDING_CAMERA).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType$Default()
                .pBindings(bindings);
        LongBuffer pSetLayout = stack.mallocLong(1);
        check(vkCreateDescriptorSetLayout(device, layoutInfo, null, pSetLayout), "vkCreateDescriptorSetLayout");
        descriptorSetLayout = pSetLayout.get(0);
    }

    private void createPipelineLayout(MemoryStack stack) {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType$Default()
                .pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pPipelineLayout = stack.mallocLong(1);
        check(vkCreatePipelineLayout(device, pipelineLayoutInfo, null, pPipelineLayout), "vkCreatePipelineLayout");
        pipelineLayout = pPipelineLayout.get(0);
    }

    private void createComputePipeline(MemoryStack stack) {
        try {
            computeShaderModule = createShaderModule(loadShaderResource(SHADER_RESOURCE));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load shader " + SHADER_RESOURCE + ": " + e.getMessage(), e);
        }
        VkPipelineShaderStageCreateInfo shaderStageInfo = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default()
                .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                .module(computeShaderModule)
                .pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack)
                .sType$Default()
                .stage(shaderStageInfo)
                .layout(pipelineLayout);
        LongBuffer pComputePipeline = stack.mallocLong(1);
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pComputePipeline), "vkCreateComputePipelines");
        computePipeline = pComputePipeline.get(0);
    }

    private void createDescriptorPool(MemoryStack stack) {
        VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(4, stack);
        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
        poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
        poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);
        poolSizes.get(3).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(1);

        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType$Default()
                .pPoolSizes(poolSizes)
                .maxSets(1);
        LongBuffer pDescriptorPool = stack.mallocLong(1);
        check(vkCreateDescriptorPool(device, poolInfo, null, pDescriptorPool), "vkCreateDescriptorPool");
        descriptorPool = pDescriptorPool.get(0);
    }

    private void allocateDescriptorSet(MemoryStack stack) {
        VkDescriptorSetAllocateInfo allocSetInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType$Default()
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pDescriptorSet = stack.mallocLong(1);
        check(vkAllocateDescriptorSets(device, allocSetInfo, pDescriptorSet), "vkAllocateDescriptorSets");
        descriptorSet = pDescriptorSet.get(0);
    }

    /** Points the descriptor set at the current TLAS and instance buffer (called after each scene swap). */
    private void updateDescriptorSet(MemoryStack stack) {
        VkDescriptorImageInfo.Buffer imageInfo = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(computeImageView);
        VkWriteDescriptorSetAccelerationStructureKHR tlasInfo = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                .sType$Default()
                .pAccelerationStructures(stack.longs(rtScene.topLevelAccelerationStructure()));
        VkDescriptorBufferInfo.Buffer instanceInfo = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(rtScene.instanceInfoBuffer().buffer()).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer cameraInfo = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(cameraUbo.buffer()).offset(0).range(VK_WHOLE_SIZE);

        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(4, stack);
        writes.get(0).sType$Default().dstSet(descriptorSet).dstBinding(BINDING_IMAGE)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).pImageInfo(imageInfo);
        writes.get(1).sType$Default().pNext(tlasInfo.address()).dstSet(descriptorSet).dstBinding(BINDING_TLAS)
                .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
        writes.get(2).sType$Default().dstSet(descriptorSet).dstBinding(BINDING_INSTANCES)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(instanceInfo);
        writes.get(3).sType$Default().dstSet(descriptorSet).dstBinding(BINDING_CAMERA)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(1).pBufferInfo(cameraInfo);
        vkUpdateDescriptorSets(device, writes, null);
    }

    private void createCommandPoolAndBuffer(MemoryStack stack) {
        VkCommandPoolCreateInfo cmdPoolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType$Default()
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(computeQueueFamilyIndex);
        LongBuffer pCmdPool = stack.mallocLong(1);
        check(vkCreateCommandPool(device, cmdPoolInfo, null, pCmdPool), "vkCreateCommandPool");
        commandPool = pCmdPool.get(0);
        VkCommandBufferAllocateInfo cmdAllocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType$Default()
                .commandPool(commandPool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
        PointerBuffer pCmdBuffer = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, cmdAllocInfo, pCmdBuffer), "vkAllocateCommandBuffers");
        commandBuffer = new VkCommandBuffer(pCmdBuffer.get(0), device);
    }

    private void createFence(MemoryStack stack) {
        LongBuffer pFence = stack.mallocLong(1);
        check(vkCreateFence(device, VkFenceCreateInfo.calloc(stack).sType$Default(), null, pFence), "vkCreateFence");
        fence = pFence.get(0);
    }

    /** Reads a SPIR-V binary from the classpath into a direct buffer (Vulkan needs off-heap memory). */
    private ByteBuffer loadShaderResource(String resource) throws IOException {
        try (InputStream in = VulkanEngine.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IOException("not found on the classpath (run the Gradle build to compile shaders)");
            }
            byte[] bytes = in.readAllBytes();
            return ByteBuffer.allocateDirect(bytes.length).put(bytes).flip();
        }
    }

    private long createShaderModule(ByteBuffer spirvCode) {
        try (MemoryStack stack = stackPush()) {
            VkShaderModuleCreateInfo createInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default()
                    .pCode(spirvCode);
            LongBuffer pShaderModule = stack.mallocLong(1);
            check(vkCreateShaderModule(device, createInfo, null, pShaderModule), "vkCreateShaderModule");
            return pShaderModule.get(0);
        }
    }

    private void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
