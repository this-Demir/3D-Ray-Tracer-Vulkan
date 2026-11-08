package dev.demir.vulkan.engine;

import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.scene.CameraState;
import dev.demir.vulkan.scene.SceneDescription;
import dev.demir.vulkan.util.Vec3;
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
import static org.lwjgl.system.MemoryUtil.memIntBuffer;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.VK10.*;

/**
 * Core Vulkan engine, running on its own thread (the Vulkan Render Thread, VRT).
 *
 * Renders with hardware ray tracing: a compute shader traces rays through the scene's
 * acceleration structures ({@link RayTracingScene}) with ray queries, and each frame is
 * copied back to the CPU for the Swing UI. Two frames are kept in flight, so the GPU
 * renders frame N while the CPU reads back frame N-1.
 *
 * Other threads only talk to it through the submit* methods (latest value wins) and read
 * {@link #stats()}; all Vulkan calls happen on the VRT. The VRT owns the accumulation
 * counter and restarts it whenever the scene, camera or settings change.
 */
public class VulkanEngine implements Runnable {

    // --- Configuration ---
    public static final int WIDTH = 1280;
    public static final int HEIGHT = 720;
    private static final int FRAMES_IN_FLIGHT = 2;

    /** Frame UBO (std140, must match FrameUBO in pathtrace.comp). */
    private static final int FRAME_UBO_SIZE = 96;

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
    private static final int BINDING_FRAME = 3;
    private static final int BINDING_ACCUMULATION = 4;
    private static final int BINDING_LIGHTS = 5;

    /** Per-frame-in-flight resources. */
    private static final class FrameSlot {
        final int index;
        VkCommandBuffer commandBuffer;
        long fence = VK_NULL_HANDLE;
        long descriptorSet = VK_NULL_HANDLE;
        GpuBuffer frameUbo;
        GpuBuffer readback;
        boolean submitted;

        FrameSlot(int index) {
            this.index = index;
        }
    }

    // --- Threading & State ---
    private volatile boolean isRunning = true;
    private final Thread thread;
    private final Consumer<Throwable> onFatalError; // Called on the VRT if it crashes

    // --- Cross-thread communication (latest value wins) ---
    private final AtomicReference<FrameData> frameQueue;                       // VRT -> UI
    private final AtomicReference<SceneDescription> pendingScene = new AtomicReference<>();
    private final AtomicReference<CameraState> pendingCamera = new AtomicReference<>();
    private final AtomicReference<RenderSettings> pendingSettings = new AtomicReference<>();
    private final ConcurrentLinkedQueue<int[]> freePixelArrays = new ConcurrentLinkedQueue<>();
    private volatile EngineStats stats = EngineStats.NONE;

    // --- Core Vulkan Objects ---
    private VkInstance instance;
    private long debugMessenger = VK_NULL_HANDLE;
    private VkPhysicalDevice physicalDevice;
    private VkDevice device;
    private VkQueue computeQueue;
    private int computeQueueFamilyIndex = -1;
    private GpuMemory gpuMemory;

    // --- Render target (RGBA8, displayed) and accumulation buffer (RGBA32F, running average) ---
    private long computeImage = VK_NULL_HANDLE;
    private long computeImageMemory = VK_NULL_HANDLE;
    private long computeImageView = VK_NULL_HANDLE;
    private long accumImage = VK_NULL_HANDLE;
    private long accumImageMemory = VK_NULL_HANDLE;
    private long accumImageView = VK_NULL_HANDLE;

    // --- Pipeline Objects ---
    private long computeShaderModule = VK_NULL_HANDLE;
    private long descriptorSetLayout = VK_NULL_HANDLE;
    private long pipelineLayout = VK_NULL_HANDLE;
    private long computePipeline = VK_NULL_HANDLE;
    private long descriptorPool = VK_NULL_HANDLE;
    private long commandPool = VK_NULL_HANDLE;
    private final FrameSlot[] slots = new FrameSlot[FRAMES_IN_FLIGHT];

    // --- GPU timing ---
    private long queryPool = VK_NULL_HANDLE; // two timestamps per slot
    private double timestampPeriodNs = 0;     // 0 = timestamps not supported

    // --- Current State (VRT only) ---
    private RayTracingScene rtScene;
    private boolean sceneLoaded = false;
    private CameraState camera;
    private RenderSettings settings = RenderSettings.DEFAULT;
    private long sampleIndex = 0; // samples accumulated since the last change
    private long frameIndex = 0;  // frames submitted in total

    // --- Statistics (VRT only) ---
    private long statsWindowStart = System.nanoTime();
    private int statsFrames = 0;
    private double statsGpuNs = 0;

    /**
     * @param frameQueue   receives the latest finished frame for the UI thread
     * @param onFatalError called on the VRT if initialization or rendering fails; the engine stops afterwards
     */
    public VulkanEngine(AtomicReference<FrameData> frameQueue, Consumer<Throwable> onFatalError) {
        this.frameQueue = frameQueue;
        this.onFatalError = onFatalError;
        this.thread = new Thread(this, "Vulkan-Engine-Thread");
        this.thread.setDaemon(true);
    }

    // --- PUBLIC API (called by other threads) ---

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

    /** Replaces the scene before the next frame (only the latest submitted scene is used). */
    public void submitScene(SceneDescription scene) {
        pendingScene.set(scene);
    }

    /** Replaces the camera before the next frame. */
    public void submitCamera(CameraState cameraState) {
        pendingCamera.set(cameraState);
    }

    /** Replaces the render settings before the next frame. */
    public void submitSettings(RenderSettings renderSettings) {
        pendingSettings.set(renderSettings);
    }

    /** Gives a displayed frame's pixel array back for reuse. */
    public void recycleFrame(FrameData frame) {
        if (frame != null && freePixelArrays.size() < 4) {
            freePixelArrays.add(frame.pixels());
        }
    }

    /** Latest statistics (updated about once per second). */
    public EngineStats stats() {
        return stats;
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
            createCommandPool(stack);
            createDescriptorSetLayout(stack);
            createPipelineLayout(stack);
            createComputePipeline(stack);
            long[] output = createStorageImage(stack, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            computeImage = output[0];
            computeImageMemory = output[1];
            computeImageView = output[2];
            long[] accumulation = createStorageImage(stack, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
            accumImage = accumulation[0];
            accumImageMemory = accumulation[1];
            accumImageView = accumulation[2];
            createDescriptorPool(stack);
            createQueryPool(stack);
            for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
                slots[i] = createFrameSlot(stack, i);
            }
            transitionImageToGeneral();
        }
        System.out.println("LOG (VRT-LIFECYCLE): VulkanEngine initialized successfully.");
    }

    private void mainLoop() {
        while (isRunning) {
            // The slot we are about to reuse holds the oldest frame in flight: publish it first.
            FrameSlot slot = slots[(int) (frameIndex % FRAMES_IN_FLIGHT)];
            if (slot.submitted) {
                waitAndPublish(slot);
            }

            if (applyPendingChanges()) {
                sampleIndex = 0;
            }

            boolean done = settings.maxSamples() > 0 && sampleIndex >= settings.maxSamples();
            if (!sceneLoaded || camera == null || done) {
                drainFramesInFlight(); // show the last frames before idling
                updateStats();
                sleep(5);
                continue;
            }

            writeFrameUbo(slot);
            recordFrame(slot);
            submitFrame(slot);
            sampleIndex++;
            frameIndex++;
            statsFrames++;
            updateStats();
        }
        drainFramesInFlight();
    }

    /** Applies scene/camera/settings changes from other threads. Returns true if anything changed. */
    private boolean applyPendingChanges() {
        boolean changed = false;
        SceneDescription newScene = pendingScene.getAndSet(null);
        if (newScene != null) {
            drainFramesInFlight();
            vkDeviceWaitIdle(device); // resources of the previous scene may still be in use
            rtScene.update(newScene);
            try (MemoryStack stack = stackPush()) {
                for (FrameSlot slot : slots) {
                    updateDescriptorSet(stack, slot);
                }
            }
            sceneLoaded = true;
            changed = true;
        }
        CameraState newCamera = pendingCamera.getAndSet(null);
        if (newCamera != null) {
            camera = newCamera;
            changed = true;
        }
        RenderSettings newSettings = pendingSettings.getAndSet(null);
        if (newSettings != null) {
            settings = newSettings;
            changed = true;
        }
        return changed;
    }

    /** Waits for every frame still on the GPU and publishes them, oldest first. */
    private void drainFramesInFlight() {
        for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
            FrameSlot slot = slots[(int) ((frameIndex + i) % FRAMES_IN_FLIGHT)];
            if (slot.submitted) {
                waitAndPublish(slot);
            }
        }
    }

    /** Writes the frame UBO for this slot (std140 layout, see FRAME_UBO_SIZE). */
    private void writeFrameUbo(FrameSlot slot) {
        ByteBuffer ubo = slot.frameUbo.mapped();
        putVec4(ubo, 0, camera.origin());
        putVec4(ubo, 16, camera.lowerLeft());
        putVec4(ubo, 32, camera.horizontal());
        putVec4(ubo, 48, camera.vertical());
        ubo.putInt(64, (int) sampleIndex);
        ubo.putInt(68, settings.maxBounces());
        ubo.putInt(72, settings.skyEnabled() ? 1 : 0);
        ubo.putInt(76, rtScene.lightCount());
        ubo.putFloat(80, settings.exposure());
        ubo.putFloat(84, 0f); // time
    }

    private static void putVec4(ByteBuffer buffer, int offset, Vec3 v) {
        buffer.putFloat(offset, (float) v.x);
        buffer.putFloat(offset + 4, (float) v.y);
        buffer.putFloat(offset + 8, (float) v.z);
        buffer.putFloat(offset + 12, 0f);
    }

    /** Records: timestamps, dispatch of the ray tracing shader, copy of the image into the slot's readback buffer. */
    private void recordFrame(FrameSlot slot) {
        VkCommandBuffer cmd = slot.commandBuffer;
        try (MemoryStack stack = stackPush()) {
            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack)
                    .sType$Default()
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
            if (queryPool != VK_NULL_HANDLE) {
                vkCmdResetQueryPool(cmd, queryPool, slot.index * 2, 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, slot.index * 2);
            }

            // The previous frame's copy (same queue) must finish reading before this frame writes.
            imageBarrier(stack, cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // The previous frame's accumulation write must be visible to this frame's read-modify-write.
            barrier(stack, cmd, accumImage, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                    stack.longs(slot.descriptorSet), null);
            vkCmdDispatch(cmd, (WIDTH + 7) / 8, (HEIGHT + 7) / 8, 1);

            imageBarrier(stack, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                    .imageSubresource(s -> s.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1))
                    .imageExtent(e -> e.width(WIDTH).height(HEIGHT).depth(1));
            vkCmdCopyImageToBuffer(cmd, computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.readback.buffer(), region);
            imageBarrier(stack, cmd, VK_ACCESS_TRANSFER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            // Make the copied pixels visible to the CPU once the fence signals.
            VkBufferMemoryBarrier.Buffer hostBarrier = VkBufferMemoryBarrier.calloc(1, stack)
                    .sType$Default()
                    .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_HOST_READ_BIT)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .buffer(slot.readback.buffer())
                    .size(VK_WHOLE_SIZE);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, null, hostBarrier, null);

            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, slot.index * 2 + 1);
            }
            check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        }
    }

    private void submitFrame(FrameSlot slot) {
        try (MemoryStack stack = stackPush()) {
            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType$Default()
                    .pCommandBuffers(stack.pointers(slot.commandBuffer));
            vkResetFences(device, slot.fence);
            check(vkQueueSubmit(computeQueue, submitInfo, slot.fence), "vkQueueSubmit");
            slot.submitted = true;
        }
    }

    /** Waits for the slot's frame, reads its GPU time and hands its pixels to the UI. */
    private void waitAndPublish(FrameSlot slot) {
        vkWaitForFences(device, slot.fence, true, Long.MAX_VALUE);
        slot.submitted = false;

        if (queryPool != VK_NULL_HANDLE) {
            try (MemoryStack stack = stackPush()) {
                LongBuffer timestamps = stack.mallocLong(2);
                int result = vkGetQueryPoolResults(device, queryPool, slot.index * 2, 2, timestamps, Long.BYTES,
                        VK_QUERY_RESULT_64_BIT);
                if (result == VK_SUCCESS) {
                    statsGpuNs += (timestamps.get(1) - timestamps.get(0)) * timestampPeriodNs;
                }
            }
        }

        int[] pixels = freePixelArrays.poll();
        if (pixels == null) {
            pixels = new int[WIDTH * HEIGHT];
        }
        memIntBuffer(slot.readback.mappedAddress(), WIDTH * HEIGHT).get(pixels);
        recycleFrame(frameQueue.getAndSet(new FrameData(pixels, WIDTH, HEIGHT))); // drop an unshown frame
    }

    private void updateStats() {
        long now = System.nanoTime();
        long elapsed = now - statsWindowStart;
        if (elapsed >= 1_000_000_000L) {
            double fps = statsFrames * 1e9 / elapsed;
            double gpuMs = statsFrames > 0 ? statsGpuNs / statsFrames / 1e6 : 0;
            stats = new EngineStats(fps, gpuMs, sampleIndex);
            statsFrames = 0;
            statsGpuNs = 0;
            statsWindowStart = now;
        } else if (stats.samples() != sampleIndex && statsFrames == 0) {
            stats = new EngineStats(stats.renderFps(), stats.gpuMs(), sampleIndex);
        }
    }

    private void imageBarrier(MemoryStack stack, VkCommandBuffer cmd, int srcAccess, int dstAccess,
                              int oldLayout, int newLayout, int srcStage, int dstStage) {
        barrier(stack, cmd, computeImage, srcAccess, dstAccess, oldLayout, newLayout, srcStage, dstStage);
    }

    private void barrier(MemoryStack stack, VkCommandBuffer cmd, long image, int srcAccess, int dstAccess,
                         int oldLayout, int newLayout, int srcStage, int dstStage) {
        VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                .sType$Default()
                .srcAccessMask(srcAccess)
                .dstAccessMask(dstAccess)
                .oldLayout(oldLayout)
                .newLayout(newLayout)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                        .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, barrier);
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
        for (FrameSlot slot : slots) {
            if (slot == null) continue;
            if (slot.fence != VK_NULL_HANDLE) vkDestroyFence(device, slot.fence, null);
            gpuMemory.destroy(slot.frameUbo);
            gpuMemory.destroy(slot.readback);
        }
        if (queryPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, queryPool, null);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, null);
        if (computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, computePipeline, null);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, null);
        if (computeShaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, computeShaderModule, null);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, null);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        if (computeImageView != VK_NULL_HANDLE) vkDestroyImageView(device, computeImageView, null);
        if (computeImage != VK_NULL_HANDLE) vkDestroyImage(device, computeImage, null);
        if (computeImageMemory != VK_NULL_HANDLE) vkFreeMemory(device, computeImageMemory, null);
        if (accumImageView != VK_NULL_HANDLE) vkDestroyImageView(device, accumImageView, null);
        if (accumImage != VK_NULL_HANDLE) vkDestroyImage(device, accumImage, null);
        if (accumImageMemory != VK_NULL_HANDLE) vkFreeMemory(device, accumImageMemory, null);
        if (gpuMemory != null) gpuMemory.destroy();
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
                    timestampPeriodNs = properties.limits().timestampPeriod();
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

    // --- Render target, frame slots, timing ---

    /** Creates a WIDTH x HEIGHT VRAM image the shader can read/write. Returns {image, memory, view}. */
    private long[] createStorageImage(MemoryStack stack, int format, int usage) {
        VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                .sType$Default()
                .imageType(VK_IMAGE_TYPE_2D).format(format)
                .extent(e -> e.width(WIDTH).height(HEIGHT).depth(1))
                .mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(usage)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        LongBuffer pImage = stack.mallocLong(1);
        check(vkCreateImage(device, imageInfo, null, pImage), "vkCreateImage");
        long image = pImage.get(0);

        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, image, memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType$Default()
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(gpuMemory.findMemoryType(memRequirements.memoryTypeBits(),
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0));
        LongBuffer pImageMemory = stack.mallocLong(1);
        check(vkAllocateMemory(device, allocInfo, null, pImageMemory), "vkAllocateMemory (image)");
        long memory = pImageMemory.get(0);
        vkBindImageMemory(device, image, memory, 0);

        VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                .sType$Default()
                .image(image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(format)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        LongBuffer pImageView = stack.mallocLong(1);
        check(vkCreateImageView(device, viewInfo, null, pImageView), "vkCreateImageView");
        return new long[]{image, memory, pImageView.get(0)};
    }

    private FrameSlot createFrameSlot(MemoryStack stack, int index) {
        FrameSlot slot = new FrameSlot(index);

        VkCommandBufferAllocateInfo cmdAllocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType$Default()
                .commandPool(commandPool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
        PointerBuffer pCmdBuffer = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, cmdAllocInfo, pCmdBuffer), "vkAllocateCommandBuffers");
        slot.commandBuffer = new VkCommandBuffer(pCmdBuffer.get(0), device);

        LongBuffer pFence = stack.mallocLong(1);
        check(vkCreateFence(device, VkFenceCreateInfo.calloc(stack).sType$Default(), null, pFence), "vkCreateFence");
        slot.fence = pFence.get(0);

        // CPU reads the frame back: cached host memory makes that read ~10x faster than uncached.
        slot.readback = gpuMemory.createBuffer((long) WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        slot.frameUbo = gpuMemory.createBuffer(FRAME_UBO_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);

        VkDescriptorSetAllocateInfo allocSetInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType$Default()
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pDescriptorSet = stack.mallocLong(1);
        check(vkAllocateDescriptorSets(device, allocSetInfo, pDescriptorSet), "vkAllocateDescriptorSets");
        slot.descriptorSet = pDescriptorSet.get(0);
        return slot;
    }

    private void createQueryPool(MemoryStack stack) {
        VkQueueFamilyProperties.Buffer families;
        IntBuffer count = stack.ints(0);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, count, null);
        families = VkQueueFamilyProperties.malloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, count, families);
        if (families.get(computeQueueFamilyIndex).timestampValidBits() == 0 || timestampPeriodNs <= 0) {
            System.out.println("LOG (VRT-LIFECYCLE): GPU timestamps not supported; GPU time will show as 0.");
            return;
        }
        VkQueryPoolCreateInfo poolInfo = VkQueryPoolCreateInfo.calloc(stack)
                .sType$Default()
                .queryType(VK_QUERY_TYPE_TIMESTAMP)
                .queryCount(FRAMES_IN_FLIGHT * 2);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateQueryPool(device, poolInfo, null, pPool), "vkCreateQueryPool");
        queryPool = pPool.get(0);
    }

    private void transitionImageToGeneral() {
        gpuMemory.submitAndWait(cmd -> {
            try (MemoryStack stack = stackPush()) {
                for (long image : new long[]{computeImage, accumImage}) {
                    barrier(stack, cmd, image, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                }
            }
        });
    }

    // --- Pipeline and descriptors ---

    private void createDescriptorSetLayout(MemoryStack stack) {
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(6, stack);
        bindings.get(0).binding(BINDING_IMAGE).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(1).binding(BINDING_TLAS).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(2).binding(BINDING_INSTANCES).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(3).binding(BINDING_FRAME).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(4).binding(BINDING_ACCUMULATION).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(5).binding(BINDING_LIGHTS).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
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
        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2 * FRAMES_IN_FLIGHT);
        poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(FRAMES_IN_FLIGHT);
        poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(2 * FRAMES_IN_FLIGHT);
        poolSizes.get(3).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType$Default()
                .pPoolSizes(poolSizes)
                .maxSets(FRAMES_IN_FLIGHT);
        LongBuffer pDescriptorPool = stack.mallocLong(1);
        check(vkCreateDescriptorPool(device, poolInfo, null, pDescriptorPool), "vkCreateDescriptorPool");
        descriptorPool = pDescriptorPool.get(0);
    }

    /** Points a slot's descriptor set at the current scene and its own frame UBO (after each scene swap). */
    private void updateDescriptorSet(MemoryStack stack, FrameSlot slot) {
        VkDescriptorImageInfo.Buffer imageInfo = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(computeImageView);
        VkWriteDescriptorSetAccelerationStructureKHR tlasInfo = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                .sType$Default()
                .pAccelerationStructures(stack.longs(rtScene.topLevelAccelerationStructure()));
        VkDescriptorBufferInfo.Buffer instanceInfo = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(rtScene.instanceInfoBuffer().buffer()).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer frameInfo = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(slot.frameUbo.buffer()).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorImageInfo.Buffer accumInfo = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(accumImageView);
        VkDescriptorBufferInfo.Buffer lightInfo = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(rtScene.lightBuffer().buffer()).offset(0).range(VK_WHOLE_SIZE);

        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(6, stack);
        writes.get(4).sType$Default().dstSet(slot.descriptorSet).dstBinding(BINDING_ACCUMULATION)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).pImageInfo(accumInfo);
        writes.get(5).sType$Default().dstSet(slot.descriptorSet).dstBinding(BINDING_LIGHTS)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(lightInfo);
        writes.get(0).sType$Default().dstSet(slot.descriptorSet).dstBinding(BINDING_IMAGE)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1).pImageInfo(imageInfo);
        writes.get(1).sType$Default().pNext(tlasInfo.address()).dstSet(slot.descriptorSet).dstBinding(BINDING_TLAS)
                .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
        writes.get(2).sType$Default().dstSet(slot.descriptorSet).dstBinding(BINDING_INSTANCES)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(instanceInfo);
        writes.get(3).sType$Default().dstSet(slot.descriptorSet).dstBinding(BINDING_FRAME)
                .descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(1).pBufferInfo(frameInfo);
        vkUpdateDescriptorSets(device, writes, null);
    }

    private void createCommandPool(MemoryStack stack) {
        VkCommandPoolCreateInfo cmdPoolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType$Default()
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(computeQueueFamilyIndex);
        LongBuffer pCmdPool = stack.mallocLong(1);
        check(vkCreateCommandPool(device, cmdPoolInfo, null, pCmdPool), "vkCreateCommandPool");
        commandPool = pCmdPool.get(0);
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
