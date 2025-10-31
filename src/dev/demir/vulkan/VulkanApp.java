package dev.demir.vulkan;

import org.lwjgl.PointerBuffer;
import org.lwjgl.assimp.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import dev.demir.vulkan.bvh.AABB;
import dev.demir.vulkan.bvh.BVHBuilder;
import dev.demir.vulkan.bvh.BVHFlattener;
import dev.demir.vulkan.bvh.BVHNode;
import dev.demir.vulkan.scene.Hittable;
import dev.demir.vulkan.scene.Triangle;
import dev.demir.vulkan.util.Vec3;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.function.LongConsumer;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import static org.lwjgl.assimp.Assimp.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.VK10.*;


/**
 * Implements "Step 5: BVH Integration" (CPU-side).
 *
 * This version integrates all the new 'bvh' and 'scene' packages.
 * 1. Loads model with Assimp.
 * 2. Builds the recursive BVH tree on the CPU using 'BVHBuilder'.
 * 3. "Flattens" the tree into a linear ByteBuffer using 'BVHFlattener'.
 * 4. Re-orders the triangle and material buffers to match the BVH leaf order.
 * 5. Uploads FOUR buffers to the GPU: Image, Triangles, Materials, and the new BVH data.
 *
 * (NOTE: The shader is still the old O(N) one.
 * Updating the shader to *use* the BVH buffer is the final step.)
 */
public class VulkanApp implements Runnable {

    // --- Configuration (Hardcoded) ---
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String SHADER_PATH = "shaders_spv/compute.spv";

    private static final String MODEL_PATH = "Moon.obj"; //

    private static final String OUTPUT_FILENAME = "step5_bvh_cpu_build.png";
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

    private long triangleBuffer = VK_NULL_HANDLE;
    private long triangleBufferMemory = VK_NULL_HANDLE;
    private long materialBuffer = VK_NULL_HANDLE;
    private long materialBufferMemory = VK_NULL_HANDLE;

    // --- NEW: Step 5 BVH Objects ---
    private long bvhBuffer = VK_NULL_HANDLE;
    private long bvhBufferMemory = VK_NULL_HANDLE;
    private int triangleCount = 0;

    // --- CPU-side Scene Data ---
    private BVHNode bvhRootNode;
    private ByteBuffer flatBvhData;
    private FloatBuffer modelVertexData;
    private FloatBuffer modelMaterialData;

    // --- Pipeline Objects ---
    private long computeShaderModule = VK_NULL_HANDLE;
    private long descriptorSetLayout = VK_NULL_HANDLE;
    private long pipelineLayout = VK_NULL_HANDLE;
    private long computePipeline = VK_NULL_HANDLE;
    private long descriptorPool = VK_NULL_HANDLE;
    private long descriptorSet = VK_NULL_HANDLE;
    private long commandPool = VK_NULL_HANDLE;
    private VkCommandBuffer commandBuffer;

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

    public static void main(String[] args) {
        new VulkanApp().run();
    }

    @Override
    public void run() {
        System.out.println("===========================================================");
        System.out.println("====================== RAY TRACER 3D ======================");
        System.out.println("===========================================================");
        System.out.println("LOG: Vulkan 'Step 5: BVH Integration' (CPU-side) starting...");
        try {
            loadModelAndBuildBVH();

            flattenSceneAndPrepareBuffers();

            initVulkan();

            runComputeTask();

        } catch (Exception e) {
            System.err.println("FATAL: A critical error occurred: " + e.getMessage());
            e.printStackTrace();
        } finally {
            cleanup();
            System.out.println("LOG: Vulkan resources cleaned up. Application exited.");
        }
        System.out.println("===========================================================");
    }

    /**
     * MODIFIED: Step 5.1 - Load Model AND Build BVH
     * Loads model vertices with Assimp and feeds them into our new
     * 'BVHBuilder' to create the CPU-side recursive tree.
     */
    private void loadModelAndBuildBVH() {
        System.out.println("LOG: Loading model: " + MODEL_PATH);

        List<Hittable> hittableTriangles = new ArrayList<>(); // For BVHBuilder

        float r = 1.0f, g = 0.0f, b = 1.0f; // Error Color
        if (MODEL_PATH.equalsIgnoreCase("Moon.obj")) {
            r = 0.8f; g = 0.8f; b = 0.8f;
        } else if (MODEL_PATH.equalsIgnoreCase("cube.obj")) {
            r = 1.0f; g = 0.0f; b = 0.0f;
        }
        System.out.println(String.format("LOG: Using base color (%.1f, %.1f, %.1f)", r, g, b));

        float matType = 0.0f;

        try (MemoryStack stack = stackPush()) {
            AIScene scene = aiImportFile(MODEL_PATH, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
            if (scene == null || (scene.mFlags() & AI_SCENE_FLAGS_INCOMPLETE) != 0 || scene.mRootNode() == null) {
                throw new RuntimeException("Failed to load model: " + aiGetErrorString());
            }

            int meshCount = scene.mNumMeshes();
            System.out.println("LOG: Model contains " + meshCount + " mesh(es).");
            triangleCount = 0;

            for (int i = 0; i < meshCount; i++) {
                AIMesh mesh = AIMesh.create(scene.mMeshes().get(i));
                int faceCount = mesh.mNumFaces();
                AIVector3D.Buffer aiVertices = mesh.mVertices();

                for (int j = 0; j < faceCount; j++) {
                    AIFace face = mesh.mFaces().get(j);
                    if (face.mNumIndices() != 3) continue;

                    AIVector3D aiV0 = aiVertices.get(face.mIndices().get(0));
                    AIVector3D aiV1 = aiVertices.get(face.mIndices().get(1));
                    AIVector3D aiV2 = aiVertices.get(face.mIndices().get(2));

                    // 1. Create Vec3 objects for CPU-side BVH build
                    Vec3 v0 = new Vec3(aiV0.x(), aiV0.y(), aiV0.z());
                    Vec3 v1 = new Vec3(aiV1.x(), aiV1.y(), aiV1.z());
                    Vec3 v2 = new Vec3(aiV2.x(), aiV2.y(), aiV2.z());

                    int materialIndex = triangleCount;
                    int vertexIndex = triangleCount * 3;

                    // 2. Add the new Triangle object to the list for the BVH builder
                    hittableTriangles.add(new Triangle(v0, v1, v2, materialIndex, vertexIndex, r, g, b, matType));

                    triangleCount++;
                }
            }
            aiReleaseImport(scene);
            System.out.println("LOG: Model loaded. Total triangles: " + triangleCount);

            if (hittableTriangles.isEmpty()) {
                throw new RuntimeException("Model loaded, but 0 triangles found.");
            }
            this.bvhRootNode = BVHBuilder.build(hittableTriangles);

        } catch (Exception e) {
            throw new RuntimeException("Failed during model loading or BVH build: " + e.getMessage(), e);
        }
    }

    /**
     * NEW: Step 5.2 - Flatten BVH and Prepare Buffers
     * Runs the BVHFlattener and re-orders the vertex/material
     * data to match the BVH leaf order.
     */
    private void flattenSceneAndPrepareBuffers() {
        if (this.bvhRootNode == null) {
            throw new RuntimeException("BVH Root node is null. Build must happen first.");
        }

        System.out.println("LOG: Flattening BVH tree for GPU...");
        BVHFlattener flattener = new BVHFlattener();

        // 1. Düzleştirme işlemini çalıştır
        this.flatBvhData = flattener.flatten(this.bvhRootNode);

        // 2. Düzleştiriciden *yeniden sıralanmış* üçgen listesini al
        List<Triangle> orderedTriangles = flattener.flattenedTriangles;
        int numFlattenedTriangles = orderedTriangles.size();

        System.out.println("LOG: BVH flattened. Triangle list re-ordered.");

        // 3. GPU tamponlarını bu *yeni sıralanmış* listeye göre doldur
        // (Her üçgen = 3 vertex * 4 float) + (1 materyal * 4 float)
        modelVertexData = memAllocFloat(numFlattenedTriangles * 3 * 4);
        modelMaterialData = memAllocFloat(numFlattenedTriangles * 4);

        for(Triangle tri : orderedTriangles) {
            // Vertex verisini ekle (v0, v1, v2)
            modelVertexData.put((float)tri.v0.x).put((float)tri.v0.y).put((float)tri.v0.z).put(0.0f); // pad
            modelVertexData.put((float)tri.v1.x).put((float)tri.v1.y).put((float)tri.v1.z).put(0.0f); // pad
            modelVertexData.put((float)tri.v2.x).put((float)tri.v2.y).put((float)tri.v2.z).put(0.0f); // pad

            // Materyal verisini ekle
            modelMaterialData.put(tri.r).put(tri.g).put(tri.b).put(tri.materialType);
        }

        modelVertexData.flip();
        modelMaterialData.flip();

        // Push constant için üçgen sayısını güncelle
        // (Düzleştirilmiş listedeki sayı, Assimp'ten gelenle aynı olmalı)
        if (this.triangleCount != numFlattenedTriangles) {
            System.err.println("WARN: Triangle count mismatch! Assimp: " + this.triangleCount + ", Flattener: " + numFlattenedTriangles);
            this.triangleCount = numFlattenedTriangles;
        }

        System.out.println("LOG: Vertex, Material, and BVH buffers are ready for upload.");
    }

    // --- 1. Initialization  ---
    private void initVulkan() {
        System.out.println("LOG: Initializing Vulkan...");
        if (!glfwInit()) {
            throw new RuntimeException("Failed to initialize GLFW!");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        System.out.println("LOG: GLFW initialized successfully (Vulkan mode).");
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();
    }
    private void createInstance() {
        try (MemoryStack stack = stackPush()) {
            VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                    .pApplicationName(memASCII("Step 5 BVH Build"))
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
            System.out.println("LOG: Vulkan Instance created.");
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
            System.out.println("LOG: Vulkan Debug Messenger created.");
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
            VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
            vkGetPhysicalDeviceProperties(physicalDevice, properties);
            System.out.println("LOG: Picked GPU: " + properties.deviceNameString());
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
            System.out.println("LOG: Logical Device and Compute Queue created.");
        }
    }

    // --- 2. Compute Task  ---

    private void runComputeTask() throws IOException {
        System.out.println("LOG: Starting Vulkan resource creation...");
        try (MemoryStack stack = stackPush()) {
            createComputeImage(stack);
            createStagingBuffer(stack);


            createTriangleBuffer(stack);
            createMaterialBuffer(stack);
            createBvhBuffer(stack);

            createComputePipeline(stack);

            createCommandPoolAndBuffer(stack);
            recordComputeCommands(stack);
            submitCommands();
            saveImageToFile(stack);
        }
    }

    private void createComputeImage(MemoryStack stack) {
        VkImageCreateInfo imageInfo = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .extent(e -> e.width(WIDTH).height(HEIGHT).depth(1))
                .mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
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
        System.out.println("LOG: Compute VkImage and VkImageView created.");
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
        System.out.println("LOG: Staging buffer created.");
    }


    /**
     * MODIFIED:  'modelVertexData--->  CPU uses.
     */
    private void createTriangleBuffer(MemoryStack stack) {
        if (modelVertexData == null) throw new RuntimeException("Model vertex data was not prepared!");
        long bufferSize = modelVertexData.remaining() * 4L;
        if (bufferSize == 0) throw new RuntimeException("Model has 0 vertices!");

        createHostVisibleBuffer(stack, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                modelVertexData,
                pBuffer -> triangleBuffer = pBuffer,
                pMemory -> triangleBufferMemory = pMemory);

        memFree(modelVertexData);
        modelVertexData = null;
        System.out.println("LOG: Triangle buffer created (from flattened list).");
    }

    /**
     * MODIFIED:  'modelVertexData--->  CPU uses.
     */
    private void createMaterialBuffer(MemoryStack stack) {
        if (modelMaterialData == null) throw new RuntimeException("Model material data was not prepared!");
        long bufferSize = modelMaterialData.remaining() * 4L;
        if (bufferSize == 0) throw new RuntimeException("Model has 0 materials!");

        createHostVisibleBuffer(stack, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                modelMaterialData,
                pBuffer -> materialBuffer = pBuffer,
                pMemory -> materialBufferMemory = pMemory);

        memFree(modelMaterialData);
        modelMaterialData = null;
        System.out.println("LOG: Material buffer created (from flattened list).");
    }

    /**
     * NEW: Step 5.3 - Create BVH Buffer
     * Creates a GPU buffer and uploads the flattened BVH data into it.
     */
    private void createBvhBuffer(MemoryStack stack) {
        if (flatBvhData == null) throw new RuntimeException("BVH data was not flattened!");
        long bufferSize = flatBvhData.remaining();
        if (bufferSize == 0) throw new RuntimeException("BVH data is empty!");

        createHostVisibleBuffer(stack, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                flatBvhData, // (ByteBuffer)
                pBuffer -> bvhBuffer = pBuffer,
                pMemory -> bvhBufferMemory = pMemory);

        flatBvhData = null;
        System.out.println("LOG: BVH buffer created and uploaded.");
    }


    /**
     * Helper: Creates a long buffer
     */
    private void createHostVisibleBuffer(MemoryStack stack, long bufferSize, int usage,
                                         ByteBuffer data, // Veri (ByteBuffer)
                                         LongConsumer bufferConsumer, LongConsumer memoryConsumer) {

        LongBuffer pBuffer = stack.mallocLong(1);
        LongBuffer pMemory = stack.mallocLong(1);

        createBuffer(stack, bufferSize, usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                pBuffer, pMemory);

        long buffer = pBuffer.get(0);
        long memory = pMemory.get(0);

        PointerBuffer pData = stack.mallocPointer(1);
        vkMapMemory(device, memory, 0, bufferSize, 0, pData);
        memCopy(memAddress(data), pData.get(0), bufferSize);
        vkUnmapMemory(device, memory);

        bufferConsumer.accept(buffer);
        memoryConsumer.accept(memory);
    }

    // FloatBuffer  (overload) helper
    private void createHostVisibleBuffer(MemoryStack stack, long bufferSize, int usage,
                                         FloatBuffer data,
                                         LongConsumer bufferConsumer, LongConsumer memoryConsumer) {
        createHostVisibleBuffer(stack, bufferSize, usage, memByteBuffer(data), bufferConsumer, memoryConsumer);
    }

    /**
     * YARDIMCI METOD: Genel buffer oluşturma ve bellek ayırma
     */
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
     * MODIFIED: Step 5.4 - Create Compute Pipeline
     * Now creates a DescriptorSetLayout for FOUR bindings:
     * binding 0 = Storage Image
     * binding 1 = Storage Buffer (Triangles)
     * binding 2 = Storage Buffer (Materials)
     * binding 3 = Storage Buffer (BVH Nodes)
     */
    private void createComputePipeline(MemoryStack stack) {
        // --- Create Descriptor Set Layout (for 4 bindings) ---
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(4, stack);

        // Binding 0: Output Image
        bindings.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        // Binding 1: Input Triangle Buffer
        bindings.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        // Binding 2: Input Material Buffer
        bindings.get(2).binding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        // Binding 3: Input BVH Buffer
        bindings.get(3).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(bindings);

        LongBuffer pSetLayout = stack.mallocLong(1);
        if (vkCreateDescriptorSetLayout(device, layoutInfo, null, pSetLayout) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create descriptor set layout!");
        }
        descriptorSetLayout = pSetLayout.get(0);

        // --- Create Pipeline Layout  ---
        VkPushConstantRange.Buffer pushConstantRange = VkPushConstantRange.calloc(1, stack)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(4); // int numTriangles

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pSetLayouts(stack.longs(descriptorSetLayout))
                .pPushConstantRanges(pushConstantRange);

        LongBuffer pPipelineLayout = stack.mallocLong(1);
        if (vkCreatePipelineLayout(device, pipelineLayoutInfo, null, pPipelineLayout) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create pipeline layout!");
        }
        pipelineLayout = pPipelineLayout.get(0);

        // --- Create Pipeline  ---
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

        // --- Create Descriptor Pool  ---
        VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
        poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(3); // 3 Storage Buffers (Tri, Mat, BVH)

        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                .pPoolSizes(poolSizes)
                .maxSets(1);

        LongBuffer pDescriptorPool = stack.mallocLong(1);
        if (vkCreateDescriptorPool(device, poolInfo, null, pDescriptorPool) != VK_SUCCESS) {
            throw new RuntimeException("Failed to create descriptor pool!");
        }
        descriptorPool = pDescriptorPool.get(0);

        // --- Allocate Descriptor Set ---
        VkDescriptorSetAllocateInfo allocSetInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));

        LongBuffer pDescriptorSet = stack.mallocLong(1);
        if (vkAllocateDescriptorSets(device, allocSetInfo, pDescriptorSet) != VK_SUCCESS) {
            throw new RuntimeException("Failed to allocate descriptor set!");
        }
        descriptorSet = pDescriptorSet.get(0);

        // --- Update Descriptor Set (with 4 writes) ---
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

        // Binding 0 (Image)
        descriptorWrites.get(0).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).pImageInfo(imageDescriptor);
        // Binding 1 (Triangle Buffer)
        descriptorWrites.get(1).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(triangleBufferDescriptor);
        // Binding 2 (Material Buffer)
        descriptorWrites.get(2).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(materialBufferDescriptor);
        //  Binding 3 (BVH Buffer)
        descriptorWrites.get(3).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).pBufferInfo(bvhBufferDescriptor);

        vkUpdateDescriptorSets(device, descriptorWrites, null);

        System.out.println("LOG: Compute Pipeline and Descriptors created for 4 bindings.");
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
        System.out.println("LOG: Command Pool and Buffer created.");
    }

    /**
     * MODIFIED: Records barriers for the image and ALL THREE buffers.
     */
    private void recordComputeCommands(MemoryStack stack) {
        VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        vkBeginCommandBuffer(commandBuffer, beginInfo);

        // 1. Barrier: Transition Image from UNDEFINED -> GENERAL
        VkImageMemoryBarrier.Buffer imageBarrier1 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(0).dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier1.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        // 1b. Barrier: Make sure Host (CPU) writes to ALL THREE buffers are visible
        VkBufferMemoryBarrier.Buffer bufferBarriers = VkBufferMemoryBarrier.calloc(3, stack);

        // Triangle Buffer Barrier
        bufferBarriers.get(0).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_HOST_WRITE_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .buffer(triangleBuffer).offset(0).size(VK_WHOLE_SIZE);
        bufferBarriers.get(0).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        // Material Buffer Barrier
        bufferBarriers.get(1).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_HOST_WRITE_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .buffer(materialBuffer).offset(0).size(VK_WHOLE_SIZE);
        bufferBarriers.get(1).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        //  BVH Buffer Barrier
        bufferBarriers.get(2).sType(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_HOST_WRITE_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .buffer(bvhBuffer).offset(0).size(VK_WHOLE_SIZE);
        bufferBarriers.get(2).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, null, bufferBarriers, imageBarrier1); // Add ALL barriers

        // 2. Bind pipeline and descriptors
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(descriptorSet), null);

        // 3. Send Push Constant (number of triangles)
        ByteBuffer pushConstantData = stack.malloc(4);
        pushConstantData.putInt(0, triangleCount);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantData);

        // 4. Dispatch the shader
        vkCmdDispatch(commandBuffer, (WIDTH + 7) / 8, (HEIGHT + 7) / 8, 1);

        // 5. Barrier: Transition Image from GENERAL -> TRANSFER_SRC_OPTIMAL (for copying)
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

        vkEndCommandBuffer(commandBuffer);

        System.out.println("LOG: Compute commands recorded (with BVH).");
    }

    private void submitCommands() {
        try (MemoryStack stack = stackPush()) {
            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(commandBuffer));
            LongBuffer pFence = stack.mallocLong(1);
            VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack).sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
            if (vkCreateFence(device, fenceInfo, null, pFence) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create fence");
            }
            long fence = pFence.get(0);
            System.out.println("LOG: Submitting compute task to GPU...");
            vkQueueSubmit(computeQueue, submitInfo, fence);
            vkWaitForFences(device, fence, true, Long.MAX_VALUE);
            vkDestroyFence(device, fence, null);
            System.out.println("LOG: GPU task finished.");
        }
    }
    private void saveImageToFile(MemoryStack stack) throws IOException {
        System.out.println("LOG: Reading data back from GPU...");
        long bufferSize = WIDTH * HEIGHT * 4;
        PointerBuffer pData = stack.mallocPointer(1);
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, pData);
        ByteBuffer pixelData = pData.getByteBuffer(0, (int) bufferSize);
        BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);
        byte[] imageData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        for (int i = 0; i < (WIDTH * HEIGHT); i++) {
            imageData[i * 4 + 0] = pixelData.get(i * 4 + 3); // A
            imageData[i * 4 + 1] = pixelData.get(i * 4 + 2); // B
            imageData[i * 4 + 2] = pixelData.get(i * 4 + 1); // G
            imageData[i * 4 + 3] = pixelData.get(i * 4 + 0); // R
        }
        vkUnmapMemory(device, stagingBufferMemory);
        try (FileOutputStream fos = new FileOutputStream(OUTPUT_FILENAME)) {
            ImageIO.write(image, "png", fos);
        }
        System.out.println("LOG: Success! Image saved to " + OUTPUT_FILENAME);
    }

    // --- 3. Cleanup  ---

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
        System.out.println("LOG: Loading shader from file: " + filepath);
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

    /**
     * TODO : HIGH COMPLEXITY -> !
     * MODIFIED: Cleans up the new bvhBuffer and its memory.
     */
    private void cleanup() {
        System.out.println("LOG: Cleaning up Vulkan objects...");
        if (device != null) {
            try {
                vkDeviceWaitIdle(device);
            } catch (Exception e) {
                System.err.println("WARN: Exception during vkDeviceWaitIdle: " + e.getMessage());
            }
        }



        // --- Destroy Step 5 Objects ---
        if (bvhBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, bvhBuffer, null);
        }
        if (bvhBufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, bvhBufferMemory, null);
        }

        // --- Destroy Step 4.5 Objects ---
        if (materialBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, materialBuffer, null);
        }
        if (materialBufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, materialBufferMemory, null);
        }

        // --- Destroy Step 4 Objects ---
        if (triangleBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, triangleBuffer, null);
        }
        if (triangleBufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, triangleBufferMemory, null);
        }

        // --- Destroy Step 2 Objects ---
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

        // --- Destroy Step 1 Objects ---
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
        if (device != null) {
            vkDestroyDevice(device, null);
            System.out.println("LOG: Logical Device destroyed.");
        }
        if (debugMessenger != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
            System.out.println("LOG: Debug Messenger destroyed.");
        }
        if (instance != null) {
            vkDestroyInstance(instance, null);
            System.out.println("LOG: Vulkan Instance destroyed.");
        }
        glfwTerminate();
        System.out.println("LOG: GLFW terminated.");
    }

}


//End Of VulkanApp.java