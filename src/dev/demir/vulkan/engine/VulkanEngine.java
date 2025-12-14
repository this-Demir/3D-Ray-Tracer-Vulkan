package dev.demir.vulkan.engine; // New package for the engine

// LWJGL/Vulkan imports
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;
import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwTerminate;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.VK10.*;

// Data classes from our 'renderer' package
import dev.demir.vulkan.renderer.BuiltCpuData;
import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.renderer.GpuSceneData;

// New Camera class
import dev.demir.vulkan.scene.Camera;

// Java utilities
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Core Vulkan Engine.
 *
 * (3-THREAD-RACE-CONDITION-FIX): This class is now a "dumb" renderer.
 * It MUST NOT modify the Camera state (e.g., resetAccumulation).
 * All state management is the responsibility of the UI Thread (VulkanApp).
 * The handleCommands() method now drains the queues to get the *latest* state.
 */
public class VulkanEngine implements Runnable {

    // --- Configuration ---
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;

    // Use the shader name you defined
    private static final String SHADER_PATH = "shaders_spv/compute_with_dynamic_light_source.spv";


    private static final boolean ENABLE_VALIDATION_LAYERS = true;

    // --- Threading & State ---
    private volatile boolean isRunning = true;
    private final Thread thread;

    // --- Thread-Safe Queues (UI -> VRT Commands) ---
    private final AtomicReference<FrameData> frameQueue; // VRT -> UI: Publishes the completed frame
    private final ConcurrentLinkedQueue<BuiltCpuData> sceneQueue;  // UI -> VRT: Sends a new scene to load
    private final ConcurrentLinkedQueue<Camera> cameraQueue; // UI -> VRT: Sends camera updates
    // --- NEW (Phase 5): Sky toggle queue ---
    private final ConcurrentLinkedQueue<Boolean> skyToggleQueue;

    // --- Core Vulkan Objects ---
    private VkInstance instance;
    private long debugMessenger = VK_NULL_HANDLE;
    private VkPhysicalDevice physicalDevice;
    private VkDevice device;
    private VkQueue computeQueue;
    private int computeQueueFamilyIndex = -1;

    // --- Dummy Buffer for NULL descriptors ---
    private long dummyBuffer = VK_NULL_HANDLE;
    private long dummyBufferMemory = VK_NULL_HANDLE;

    // --- Compute Resources ---
    private long computeImage = VK_NULL_HANDLE;
    private long computeImageMemory = VK_NULL_HANDLE;
    private long computeImageView = VK_NULL_HANDLE;
    private long stagingBuffer = VK_NULL_HANDLE;
    private long stagingBufferMemory = VK_NULL_HANDLE;

    // --- Camera Uniform Buffer Object (UBO) ---
    private long cameraUbo = VK_NULL_HANDLE;
    private long cameraUboMemory = VK_NULL_HANDLE;
    private ByteBuffer cameraUboMapped = null; // Persistently mapped for fast updates

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
    private GpuSceneData currentScene = null;
    private Camera currentCamera = null; // This is a *copy* of the state from the UI thread
    private int isSkyEnabled = 1; // 1 = true (default ON), 0 = false

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
     * Constructor. Prepares the engine and its communication channels.
     * @param frameQueue Thread-safe reference used to publish completed frames to the UI thread.
     */
    public VulkanEngine(AtomicReference<FrameData> frameQueue) {
        this.frameQueue = frameQueue;
        this.sceneQueue = new ConcurrentLinkedQueue<>();
        this.cameraQueue = new ConcurrentLinkedQueue<>();
        this.skyToggleQueue = new ConcurrentLinkedQueue<>(); // <-- NEW
        this.thread = new Thread(this, "Vulkan-Engine-Thread");
        this.thread.setDaemon(true);
    }

    // --- PUBLIC API (Called by the UI Thread) ---

    /**
     * Starts the Vulkan Render Thread (VRT).
     */
    public void start() {
        this.isRunning = true;
        this.thread.start();
    }

    /**
     * Sends a stop signal to the VRT and waits for shutdown.
     */
    public void stop() {
        this.isRunning = false;
        try {
            this.thread.join(5000); // 5s graceful shutdown window
        } catch (InterruptedException e) {
            System.err.println("WARN (UI): Interrupted while stopping VulkanEngine.");
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Thread-safe method for the UI thread to submit a new scene to load.
     * @param sceneData CPU-side data produced by SceneBuilder.
     */
    public void submitScene(BuiltCpuData sceneData) {
        // --- 3-THREAD-FIX ---
        // DO NOT reset accumulation here. The UI thread (VulkanApp)
        // is responsible for resetting its own camera state.
        System.out.println("LOG (VRT-QUEUE): Received submitScene() command.");
        sceneQueue.add(sceneData);
    }

    /**
     * Thread-safe method for the UI thread to submit a camera update.
     * @param camera New camera state.
     */
    public void submitCameraUpdate(Camera camera) {
        // Camera state is managed by VulkanApp.
        // We just queue the latest state.
        // System.out.println("LOG (VRT-QUEUE): Received submitCameraUpdate(fc=" + camera.getFrameCount() + ") command.");
        cameraQueue.add(camera);
    }

    /**
     * --- NEW (Phase 5): Thread-safe method for UI to toggle sky.
     * @param isSkyOn true if sky light is enabled, false for black.
     */
    public void submitSkyToggle(boolean isSkyOn) {
        // --- 3-THREAD-FIX ---
        // DO NOT reset accumulation here. The UI thread (VulkanApp)
        // is responsible for resetting its own camera state.
        System.out.println("LOG (VRT-QUEUE): Received submitSkyToggle(" + isSkyOn + ") command.");
        skyToggleQueue.add(isSkyOn);
    }


    // --- VULKAN RENDER THREAD (VRT) LOGIC ---

    /**
     * Main entry point for the Vulkan Render Thread.
     */
    @Override
    public void run() {
        try {
            initVulkan();
            mainLoop();
        } catch (Exception e) {
            System.err.println("FATAL (VRT): VulkanEngine thread crashed!");
            e.printStackTrace();
            isRunning = false;
        } finally {
            cleanup();
            System.out.println("LOG (VRT-LIFECYCLE): VulkanEngine shut down.");
        }
    }

    /**
     * Initializes all core Vulkan components, pipeline, and static resources.
     */
    private void initVulkan() {
        System.out.println("LOG (VRT-LIFECYCLE): Starting VulkanEngine initialization...");
        if (!glfwInit()) {
            throw new RuntimeException("GLFW could not be initialized!");
        }

        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();

        try (MemoryStack stack = stackPush()) {
            createCommandPoolAndBuffer(stack);
            createFence(stack);
            createDescriptorSetLayout(stack); // Now has 6 bindings
            createPipelineLayout(stack);
            createComputePipeline(stack);
            createComputeImage(stack); // Now has STORAGE_BIT usage
            createStagingBuffer(stack);
            createCameraUbo(stack); // UBO size is now 80 bytes
            createDummyBuffer(stack);
            createDescriptorPool(stack); // Now allocates 2 storage images
            allocateDescriptorSet(stack);
            // Transition image layout to GENERAL once at the start
            transitionImageLayout(computeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        }
        System.out.println("LOG (VRT-LIFECYCLE): VulkanEngine initialized successfully.");
    }

    /**
     * Main render loop. Runs on the VRT.
     * UPDATED for Phase 5.
     */
    private void mainLoop() {
        while (isRunning) {
            // 1. Process all pending commands
            handleCommands();

            // 2. Wait if scene or camera are not ready
            if (currentScene == null || currentCamera == null) {
                // System.out.println("LOG (VRT-LOOP): No scene or camera. Sleeping.");
                sleep(16); // ~60 FPS sleep
                continue;
            }

            // 3. (NEW) Update the UBO every single frame.
            // This sends the incrementing frameCount and current sky state to the shader.
            internalUpdateCameraUBO();

            // 4. Render one frame
            FrameData frame = renderFrame();

            // 5. Publish the completed frame to the UI thread
            frameQueue.set(frame);

            // 6. (RACE CONDITION FIX)
            // The VRT MUST NOT increment the frame count.
            // This is now handled by the UI Thread (VulkanApp).
            // currentCamera.incrementFrameCount(); // <-- REMOVED
        }
    }

    /**
     * Processes all pending commands from the queues (scene/camera/sky).
     * (3-THREAD-FIX): This now *drains* the queues to get the latest state.
     */
    private void handleCommands() {
        String logPrefix = "LOG (VRT-HANDLE): ";

        // 1. Check for a new scene
        BuiltCpuData newSceneData = sceneQueue.poll();
        if (newSceneData != null) {
            System.out.println(logPrefix + "Found new scene in queue. Swapping...");
            internalSwapScene(newSceneData);
        }

        // 2. Check for a camera update
        // We *must* drain the queue to get the *latest* camera state
        Camera newCamera = null;
        Camera polledCamera;
        while ((polledCamera = cameraQueue.poll()) != null) {
            newCamera = polledCamera;
        }
        if (newCamera != null) {
            // System.out.println(logPrefix + "Found new camera state in queue. Using latest fc=" + newCamera.getFrameCount());
            this.currentCamera = newCamera;
        }

        // 3. (NEW) Check for a sky toggle
        // We *must* drain the queue to get the *latest* sky state
        Boolean skyState = null;
        Boolean polledSkyState;
        while ((polledSkyState = skyToggleQueue.poll()) != null) {
            skyState = polledSkyState;
        }
        if (skyState != null) {
            int newSkyState = skyState ? 1 : 0;
            if (this.isSkyEnabled != newSkyState) {
                System.out.println(logPrefix + "Found new sky state in queue. Setting to: " + newSkyState);
                this.isSkyEnabled = newSkyState;
            }
        }
    }

    /**
     * Destroys the old scene (if any) and loads the new one safely.
     */
    private void internalSwapScene(BuiltCpuData cpuData) {
        String logPrefix = "LOG (VRT-SWAP): ";
        System.out.println(logPrefix + "Waiting for device idle...");
        vkDeviceWaitIdle(device);
        System.out.println(logPrefix + "Device idle.");

        if (currentScene != null) {
            System.out.println(logPrefix + "Destroying old GPU scene data...");
            destroyGpuSceneData(currentScene);
            currentScene = null;
        }

        try (MemoryStack stack = stackPush()) {
            long triBuffer = VK_NULL_HANDLE, triMem = VK_NULL_HANDLE;
            long matBuffer = VK_NULL_HANDLE, matMem = VK_NULL_HANDLE;
            long bvhBuffer = VK_NULL_HANDLE, bvhMem = VK_NULL_HANDLE;

            System.out.println(logPrefix + "Uploading buffers...");

            long triBufferSize = cpuData.modelVertexData.remaining() * 4L;
            if (triBufferSize > 0) {
                LongBuffer pBuf = stack.mallocLong(1); LongBuffer pMem = stack.mallocLong(1);
                createHostVisibleBuffer(stack, triBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memByteBuffer(cpuData.modelVertexData), pBuf, pMem);
                triBuffer = pBuf.get(0); triMem = pMem.get(0);
            }
            memFree(cpuData.modelVertexData);

            long matBufferSize = cpuData.modelMaterialData.remaining() * 4L;
            if (matBufferSize > 0) {
                LongBuffer pBuf = stack.mallocLong(1); LongBuffer pMem = stack.mallocLong(1);
                createHostVisibleBuffer(stack, matBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memByteBuffer(cpuData.modelMaterialData), pBuf, pMem);
                matBuffer = pBuf.get(0); matMem = pMem.get(0);
            }
            memFree(cpuData.modelMaterialData);

            long bvhBufferSize = cpuData.flatBvhData.remaining();
            if (bvhBufferSize > 0) {
                LongBuffer pBuf = stack.mallocLong(1); LongBuffer pMem = stack.mallocLong(1);
                createHostVisibleBuffer(stack, bvhBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cpuData.flatBvhData, pBuf, pMem);
                bvhBuffer = pBuf.get(0); bvhMem = pMem.get(0);
            }
            System.out.println(logPrefix + "Buffers uploaded.");

            // Use dummy buffers if scene was empty
            long validTriBuffer = (triBuffer == VK_NULL_HANDLE) ? dummyBuffer : triBuffer;
            long validMatBuffer = (matBuffer == VK_NULL_HANDLE) ? dummyBuffer : matBuffer;
            long validBvhBuffer = (bvhBuffer == VK_NULL_HANDLE) ? dummyBuffer : bvhBuffer;

            System.out.println(logPrefix + "Updating descriptor set...");
            // Update the single descriptor set to point to the new buffers
            updateDescriptorSet(stack, validTriBuffer, validMatBuffer, validBvhBuffer, cameraUbo);

            currentScene = new GpuSceneData(triBuffer, triMem, matBuffer, matMem, bvhBuffer, bvhMem, cpuData.triangleCount);
            System.out.println(logPrefix + "New scene loaded. Triangle count: " + cpuData.triangleCount); // Use the actual count
        }
    }

    /**
     * Updates the persistently mapped camera UBO with new data.
     */
    private void internalUpdateCameraUBO() {
        if (cameraUboMapped == null || currentCamera == null) return;

        int frameCount = currentCamera.getFrameCount();

        // System.out.println("LOG (VRT-UBO): Writing to UBO. frameCount=" + frameCount + ", isSkyEnabled=" + this.isSkyEnabled);

        // Structure matching std140 layout in the shader:
        // vec3s (16-byte aligned)
        currentCamera.getOrigin().store(0, cameraUboMapped);                // offset 0
        currentCamera.getLowerLeft().store(16, cameraUboMapped); // offset 16
        currentCamera.getHorizontal().store(32, cameraUboMapped);           // offset 32
        currentCamera.getVertical().store(48, cameraUboMapped);             // offset 48

        // --- NEW (Phase 5): Write int values ---
        // We write them at offset 64 (the next 16-byte alignment block)
        cameraUboMapped.putInt(64, frameCount);
        cameraUboMapped.putInt(68, this.isSkyEnabled); // Use the VRT's stored state
    }

    /**
     * Renders a single frame and returns pixel data in a FrameData object.
     */
    private FrameData renderFrame() {
        // System.out.println("LOG (VRT-RENDER): Starting renderFrame()...");
        try (MemoryStack stack = stackPush()) {
            recordComputeCommands(stack, currentScene);

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(commandBuffer));

            vkResetFences(device, fence);
            // System.out.println("LOG (VRT-RENDER): Submitting queue...");
            vkQueueSubmit(computeQueue, submitInfo, fence);
            vkWaitForFences(device, fence, true, Long.MAX_VALUE);
            // System.out.println("LOG (VRT-RENDER): Fence wait complete. Mapping memory...");

            long bufferSize = WIDTH * HEIGHT * 4;
            PointerBuffer pData = stack.mallocPointer(1);
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, pData);
            ByteBuffer pixelData = pData.getByteBuffer(0, (int) bufferSize);

            // Create a copy, as the mapped buffer will be unmapped
            ByteBuffer copyPixelData = ByteBuffer.allocateDirect((int)bufferSize);
            copyPixelData.put(pixelData);
            copyPixelData.flip();

            vkUnmapMemory(device, stagingBufferMemory);
            // System.out.println("LOG (VRT-RENDER): Memory unmapped. Returning frame.");

            return new FrameData(copyPixelData);
        }
    }

    /**
     * Records all Vulkan commands for a single frame.
     * UPDATED for Phase 5 (Image barriers for accumulation).
     */
    private void recordComputeCommands(MemoryStack stack, GpuSceneData scene) {
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        vkBeginCommandBuffer(commandBuffer, beginInfo);

        // Barrier 1: Ensure image is ready for read/write
        // Wait for previous frame's READ (from binding 5) to finish
        // before this frame's WRITE (to binding 0).
        VkImageMemoryBarrier.Buffer imageBarrier1 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT) // Allow both read (b5) and write (b0)
                .oldLayout(VK_IMAGE_LAYOUT_GENERAL)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier1.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Wait for last compute
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Start in compute
                0, null, null, imageBarrier1);

        // --- Bind Pipeline and Descriptors ---
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(descriptorSet), null);

        // --- Push Constants ---
        ByteBuffer pushConstantData = stack.malloc(4);
        pushConstantData.putInt(0, scene.triangleCount); // Use the correct triangle count
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantData);

        // --- Dispatch ---
        vkCmdDispatch(commandBuffer, (WIDTH + 7) / 8, (HEIGHT + 7) / 8, 1);

        // Barrier 2: Transition for copy
        // Wait for this frame's WRITE to finish before TRANSFER_READ.
        VkImageMemoryBarrier.Buffer imageBarrier2 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_GENERAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier2.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Wait for compute
                VK_PIPELINE_STAGE_TRANSFER_BIT,       // Start in transfer
                0, null, null, imageBarrier2);

        // --- Copy Image to Staging Buffer ---
        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                .bufferOffset(0)
                .bufferRowLength(0).bufferImageHeight(0)
                .imageSubresource(s -> s.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1))
                .imageOffset(o -> o.x(0).y(0).z(0))
                .imageExtent(e -> e.width(WIDTH).height(HEIGHT).depth(1));
        vkCmdCopyImageToBuffer(commandBuffer, computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, region);

        // Barrier 3: Transition back to GENERAL
        // Wait for TRANSFER_READ to finish before preparing for next frame's SHADER_READ.
        VkImageMemoryBarrier.Buffer imageBarrier3 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_TRANSFER_READ_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT) // Read for next frame
                .oldLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier3.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,       // Wait for transfer
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Start in compute (for next frame)
                0, null, null, imageBarrier3);

        vkEndCommandBuffer(commandBuffer);
    }

    /**
     * Cleans up all Vulkan resources.
     */
    private void cleanup() {
        System.out.println("LOG (VRT-LIFECYCLE): Cleaning up VulkanEngine resources...");
        if (device != null) {
            try {
                vkDeviceWaitIdle(device);
            } catch (Exception e) {
                System.err.println("WARN (VRT-LIFECYCLE): Exception during vkDeviceWaitIdle: " + e.getMessage());
            }
        }
        if (cameraUboMapped != null) vkUnmapMemory(device, cameraUboMemory);
        if (currentScene != null) destroyGpuSceneData(currentScene);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, null);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, null);
        if (computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, computePipeline, null);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, null);
        if (computeShaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, computeShaderModule, null);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, null);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        if (dummyBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, dummyBuffer, null);
        if (dummyBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, dummyBufferMemory, null);
        if (computeImageView != VK_NULL_HANDLE) vkDestroyImageView(device, computeImageView, null);
        if (computeImage != VK_NULL_HANDLE) vkDestroyImage(device, computeImage, null);
        if (computeImageMemory != VK_NULL_HANDLE) vkFreeMemory(device, computeImageMemory, null);
        if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, stagingBuffer, null);
        if (stagingBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, stagingBufferMemory, null);
        if (cameraUbo != VK_NULL_HANDLE) vkDestroyBuffer(device, cameraUbo, null);
        if (cameraUboMemory != VK_NULL_HANDLE) vkFreeMemory(device, cameraUboMemory, null);
        if (device != null) vkDestroyDevice(device, null);
        if (debugMessenger != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        if (instance != null) vkDestroyInstance(instance, null);
        glfwTerminate();
    }

    /**
     * Helper to destroy GpuSceneData buffers.
     */
    private void destroyGpuSceneData(GpuSceneData scene) {
        if (scene.triangleBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, scene.triangleBuffer, null);
        if (scene.triangleBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, scene.triangleBufferMemory, null);
        if (scene.materialBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, scene.materialBuffer, null);
        if (scene.materialBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, scene.materialBufferMemory, null);
        if (scene.bvhBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, scene.bvhBuffer, null);
        if (scene.bvhBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, scene.bvhBufferMemory, null);
    }

    // --- VULKAN HELPERS ---
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
                throw new RuntimeException("Required GLFW extensions not found!");
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
                throw new RuntimeException("No Vulkan-capable GPU found!");
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
                throw new RuntimeException("No suitable GPU found!");
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

    /**
     * Creates the compute image.
     * UPDATED for Phase 5.
     */
    private void createComputeImage(MemoryStack stack) {
        VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .extent(e -> e.width(WIDTH).height(HEIGHT).depth(1))
                .mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                // --- NEW (Phase 5): Need to read from this image (binding 5)
                // and write to it (binding 0). STORAGE_BIT covers both.
                .usage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        LongBuffer pImage = stack.mallocLong(1);
        if (vkCreateImage(device, imageInfo, null, pImage) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create compute image!");
        }
        computeImage = pImage.get(0);
        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, computeImage, memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        LongBuffer pImageMemory = stack.mallocLong(1);
        if (vkAllocateMemory(device, allocInfo, null, pImageMemory) != VK_SUCCESS) {
            throw new RuntimeException("Failed to allocate image memory!");
        }
        computeImageMemory = pImageMemory.get(0);
        vkBindImageMemory(device, computeImage, computeImageMemory, 0);
        VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(computeImage).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        LongBuffer pImageView = stack.mallocLong(1);
        if (vkCreateImageView(device, viewInfo, null, pImageView) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create image view!");
        }
        computeImageView = pImageView.get(0);
    }

    private void createStagingBuffer(MemoryStack stack) {
        long bufferSize = WIDTH * HEIGHT * 4;
        VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(bufferSize)
                .usage(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        LongBuffer pBuffer = stack.mallocLong(1);
        if (vkCreateBuffer(device, bufferInfo, null, pBuffer) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create staging buffer!");
        }
        stagingBuffer = pBuffer.get(0);
        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, stagingBuffer, memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits(),
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
        LongBuffer pBufferMemory = stack.mallocLong(1);
        if (vkAllocateMemory(device, allocInfo, null, pBufferMemory) != VK_SUCCESS) {
            throw new RuntimeException("Failed to allocate staging buffer memory!");
        }
        stagingBufferMemory = pBufferMemory.get(0);
        vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);
    }

    /**
     * Creates the camera UBO.
     * UPDATED for Phase 5.
     */
    private void createCameraUbo(MemoryStack stack) {
        // 4 vec3s (16-byte aligned) + 2 ints (16-byte aligned block)
        // (4 * 16) + 16 = 80 bytes
        long bufferSize = 80;

        LongBuffer pBuffer = stack.mallocLong(1);
        LongBuffer pMemory = stack.mallocLong(1);

        createBuffer(stack, bufferSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                pBuffer, pMemory);

        cameraUbo = pBuffer.get(0);
        cameraUboMemory = pMemory.get(0);

        // Persistently map the buffer
        PointerBuffer pData = stack.mallocPointer(1);
        vkMapMemory(device, cameraUboMemory, 0, bufferSize, 0, pData);
        cameraUboMapped = pData.getByteBuffer(0, (int)bufferSize);

        System.out.println("LOG (VRT-LIFECYCLE): Camera UBO created and persistently mapped.");
    }

    /**
     * Creates a tiny dummy buffer.
     * (No changes from Phase 4)
     */
    private void createDummyBuffer(MemoryStack stack) {
        long bufferSize = 4; // 4 bytes is a minimal valid size
        LongBuffer pBuffer = stack.mallocLong(1);
        LongBuffer pMemory = stack.mallocLong(1);
        createBuffer(stack, bufferSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                pBuffer, pMemory);
        dummyBuffer = pBuffer.get(0);
        dummyBufferMemory = pMemory.get(0);
    }

    private void createHostVisibleBuffer(MemoryStack stack, long bufferSize, int usage,
                                         ByteBuffer data, LongBuffer pBuffer, LongBuffer pMemory) {
        createBuffer(stack, bufferSize, usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                pBuffer, pMemory);
        PointerBuffer pData = stack.mallocPointer(1);
        vkMapMemory(device, pMemory.get(0), 0, bufferSize, 0, pData);
        memCopy(memAddress(data), pData.get(0), bufferSize);
        vkUnmapMemory(device, pMemory.get(0));
    }

    private void createBuffer(MemoryStack stack, long size, int usage, int properties,
                              LongBuffer pBuffer, LongBuffer pMemory) {
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
     * Creates the DescriptorSetLayout.
     * UPDATED for Phase 5 (Added binding 5 for accumulation image).
     */
    private void createDescriptorSetLayout(MemoryStack stack) {
        // --- UPDATED: 6 bindings total ---
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(6, stack);

        bindings.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT); // resultImage (Write)
        bindings.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT); // triBuf
        bindings.get(2).binding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT); // matBuf
        bindings.get(3).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT); // bvhBuf
        bindings.get(4).binding(4).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT); // cameraUbo

        // --- NEW (Phase 5): Binding 5 for previous frame (accumulation) ---
        bindings.get(5).binding(5).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT); // prevResultImage (Read)

        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(bindings);
        LongBuffer pSetLayout = stack.mallocLong(1);
        vkCreateDescriptorSetLayout(device, layoutInfo, null, pSetLayout);
        descriptorSetLayout = pSetLayout.get(0);
    }

    /**
     * Creates only the PipelineLayout.
     * (No changes from Phase 4)
     */
    private void createPipelineLayout(MemoryStack stack) {
        VkPushConstantRange.Buffer pushConstantRange = VkPushConstantRange.calloc(1, stack)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(4); // int numTriangles
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pSetLayouts(stack.longs(descriptorSetLayout))
                .pPushConstantRanges(pushConstantRange);
        LongBuffer pPipelineLayout = stack.mallocLong(1);
        vkCreatePipelineLayout(device, pipelineLayoutInfo, null, pPipelineLayout);
        pipelineLayout = pPipelineLayout.get(0);
    }

    /**
     * Creates only the ComputePipeline.
     * (No changes from Phase 4)
     */
    private void createComputePipeline(MemoryStack stack) {
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pComputePipeline) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create compute pipeline!");
        }
        computePipeline = pComputePipeline.get(0);
    }

    /**
     * Creates only the DescriptorPool.
     * UPDATED for Phase 5 (More storage images).
     */
    private void createDescriptorPool(MemoryStack stack) {
        VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(3, stack);
        // --- UPDATED: 2 storage images (current frame + prev frame) ---
        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2);
        poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(3); // Tri, Mat, BVH
        poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(1); // Camera

        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                .pPoolSizes(poolSizes)
                .maxSets(1); // We only need one descriptor set
        LongBuffer pDescriptorPool = stack.mallocLong(1);
        if (vkCreateDescriptorPool(device, poolInfo, null, pDescriptorPool) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create descriptor pool!");
        }
        descriptorPool = pDescriptorPool.get(0);
    }

    /**
     * Allocates our single descriptor set.
     * (No changes from Phase 4)
     */
    private void allocateDescriptorSet(MemoryStack stack) {
        VkDescriptorSetAllocateInfo allocSetInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pDescriptorSet = stack.mallocLong(1);
        if (vkAllocateDescriptorSets(device, allocSetInfo, pDescriptorSet) != VK_SUCCESS) {
            throw new RuntimeException("Failed to allocate descriptor set!");
        }
        descriptorSet = pDescriptorSet.get(0);
    }

    /**
     * Updates our single reusable descriptor set.
     * UPDATED for Phase 5 (Added binding 5 for accumulation image).
     */
    private void updateDescriptorSet(MemoryStack stack, long triangleBuffer, long materialBuffer, long bvhBuffer, long cameraBuffer) {

        // --- UPDATED: 2 image descriptors ---
        VkDescriptorImageInfo.Buffer imageDescriptor = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(computeImageView);

        // (NEW) The 'previous frame' image is the *same* image,
        // just read from a different binding.
        VkDescriptorImageInfo.Buffer prevImageDescriptor = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(computeImageView);

        // Scene buffers
        VkDescriptorBufferInfo.Buffer triangleBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(triangleBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer materialBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(materialBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer bvhBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(bvhBuffer).offset(0).range(VK_WHOLE_SIZE);

        // Camera buffer
        VkDescriptorBufferInfo.Buffer cameraBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(cameraBuffer).offset(0).range(VK_WHOLE_SIZE);

        // --- UPDATED: 6 descriptor writes ---
        VkWriteDescriptorSet.Buffer descriptorWrites = VkWriteDescriptorSet.calloc(6, stack);

        descriptorWrites.get(0).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).pImageInfo(imageDescriptor); // Current Frame (Write)
        descriptorWrites.get(1).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(triangleBufferDescriptor);
        descriptorWrites.get(2).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(materialBufferDescriptor);
        descriptorWrites.get(3).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(bvhBufferDescriptor);
        descriptorWrites.get(4).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(4).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1).pBufferInfo(cameraBufferDescriptor);

        // --- NEW (Phase 5): Binding 5 (Prev Frame) ---
        descriptorWrites.get(5).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(5).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).pImageInfo(prevImageDescriptor); // Prev Frame (Read)

        vkUpdateDescriptorSets(device, descriptorWrites, null);
    }

    private void createCommandPoolAndBuffer(MemoryStack stack) {
        VkCommandPoolCreateInfo cmdPoolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(computeQueueFamilyIndex);
        LongBuffer pCmdPool = stack.mallocLong(1);
        if (vkCreateCommandPool(device, cmdPoolInfo, null, pCmdPool) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create command pool!");
        }
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
            throw new RuntimeException("Failed to create fence!");
        }
        fence = pFence.get(0);
    }

    /**
     * One-time command helper for image layout transitions.
     */
    private void transitionImageLayout(long image, int oldLayout, int newLayout) {
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandPool(commandPool)
                    .commandBufferCount(1);
            PointerBuffer pCmdBuffer = stack.mallocPointer(1);
            vkAllocateCommandBuffers(device, allocInfo, pCmdBuffer);
            VkCommandBuffer cmdBuffer = new VkCommandBuffer(pCmdBuffer.get(0), device);

            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            vkBeginCommandBuffer(cmdBuffer, beginInfo);

            VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                    .oldLayout(oldLayout)
                    .newLayout(newLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                            .baseMipLevel(0).levelCount(1)
                            .baseArrayLayer(0).layerCount(1));

            int sourceStage, destStage;
            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
                // --- UPDATED (Phase 5): Access mask must allow both read and write ---
                barrier.srcAccessMask(0);
                barrier.dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            } else {
                throw new IllegalArgumentException("Unsupported layout transition!");
            }

            vkCmdPipelineBarrier(cmdBuffer, sourceStage, destStage, 0, null, null, barrier);
            vkEndCommandBuffer(cmdBuffer);

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(pCmdBuffer);
            vkQueueSubmit(computeQueue, submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(computeQueue); // Wait for completion
            vkFreeCommandBuffers(device, commandPool, pCmdBuffer);

            System.out.println("LOG (VRT-LIFECYCLE): Initial image layout transitioned UNDEFINED -> GENERAL.");
        }
    }

    private int findMemoryType(int typeFilter, int properties) {
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
        try (FileInputStream fis = new FileInputStream(filepath);
             FileChannel fc = fis.getChannel()) {
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) fc.size());
            while (fc.read(buffer) > 0) ;
            return buffer.flip();
        }
    }

    private long createShaderModule(ByteBuffer spirvCode) {
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

    private void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}