package dev.demir.vulkan.engine; // Motor için yeni bir paket

// LWJGL/Vulkan importları
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

// 'renderer' paketimizdeki veri sınıfları
import dev.demir.vulkan.renderer.BuiltCpuData;
import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.renderer.GpuSceneData;

// Yeni Camera sınıfımız
import dev.demir.vulkan.scene.Camera;

// Java yardımcı programları
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
 * Çekirdek Vulkan Motoru.
 * Bu sınıf Runnable arayüzünü uygular ve kendi adanmış iş parçacığında (VRT) çalışır.
 * Kendi Vulkan kaynaklarını ve yaşam döngüsünü yönetir.
 * UI iş parçacığından (Swing/EDT) gelen komutları thread-safe kuyruklar aracılığıyla alır.
 * Bu mimari, senkronizasyon sorunlarını çözer, layout hatalarını düzeltir
 * ve render mantığını UI'dan ayırır.
 */
public class VulkanEngine implements Runnable {

    // --- Konfigürasyon ---
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    // Kamera için UBO kullanacak yeni shader'a işaret ediyoruz
    private static final String SHADER_PATH = "shaders_spv/compute_dynamic.spv";
    private static final boolean ENABLE_VALIDATION_LAYERS = true;

    // --- Threading ve Durum ---
    private volatile boolean isRunning = true;
    private final Thread thread;

    // --- Thread-Safe Kuyruklar (UI -> VRT Komutları) ---
    // UI thread'i (EDT) bu kuyruklara veri *koyar*.
    // VRT (bu sınıf) bu kuyruklardan veriyi `run()` döngüsünde *çeker*.
    private final AtomicReference<FrameData> frameQueue; // VRT -> UI: Tamamlanan kareyi yayınlar
    private final ConcurrentLinkedQueue<BuiltCpuData> sceneQueue;  // UI -> VRT: Yüklenecek new sahneyi gönderir
    private final ConcurrentLinkedQueue<Camera> cameraQueue; // UI -> VRT: Kamera güncellemesini gönderir

    // --- Vulkan Çekirdek Nesneleri ---
    private VkInstance instance;
    private long debugMessenger = VK_NULL_HANDLE;
    private VkPhysicalDevice physicalDevice;
    private VkDevice device;
    private VkQueue computeQueue;
    private int computeQueueFamilyIndex = -1;

    // --- Compute Kaynakları ---
    private long computeImage = VK_NULL_HANDLE;
    private long computeImageMemory = VK_NULL_HANDLE;
    private long computeImageView = VK_NULL_HANDLE;
    private long stagingBuffer = VK_NULL_HANDLE;
    private long stagingBufferMemory = VK_NULL_HANDLE;

    // --- YENİ: Kamera Uniform Buffer Object (UBO) ---
    private long cameraUbo = VK_NULL_HANDLE;
    private long cameraUboMemory = VK_NULL_HANDLE;
    private ByteBuffer cameraUboMapped = null; // Hızlı güncellemeler için kalıcı olarak haritalanmış tampon

    // --- Pipeline Nesneleri ---
    private long computeShaderModule = VK_NULL_HANDLE;
    private long descriptorSetLayout = VK_NULL_HANDLE;
    private long pipelineLayout = VK_NULL_HANDLE;
    private long computePipeline = VK_NULL_HANDLE;
    private long descriptorPool = VK_NULL_HANDLE;
    private long descriptorSet = VK_NULL_HANDLE;
    private long commandPool = VK_NULL_HANDLE;
    private VkCommandBuffer commandBuffer;
    private long fence = VK_NULL_HANDLE;

    // --- Mevcut Durum ---
    private GpuSceneData currentScene = null;

    // Statik Doğrulama Katmanı Kurulumu
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
     * Constructor. Motoru ve iletişim kanallarını hazırlar.
     * @param frameQueue Tamamlanan kareleri UI thread'ine yayınlamak için thread-safe referans.
     */
    public VulkanEngine(AtomicReference<FrameData> frameQueue) {
        this.frameQueue = frameQueue;
        this.sceneQueue = new ConcurrentLinkedQueue<>();
        this.cameraQueue = new ConcurrentLinkedQueue<>();
        this.thread = new Thread(this, "Vulkan-Engine-Thread");
        this.thread.setDaemon(true);
    }

    // --- PUBLIC API (UI Thread tarafından çağrılır) ---

    /**
     * Vulkan Motor Thread'ini (VRT) başlatır.
     */
    public void start() {
        this.isRunning = true;
        this.thread.start();
    }

    /**
     * Vulkan Motor Thread'ine durma sinyali gönderir ve kapanmasını bekler.
     */
    public void stop() {
        this.isRunning = false;
        try {
            this.thread.join(5000); // 5 saniye güvenli kapanma süresi
        } catch (InterruptedException e) {
            System.err.println("WARN (UI): VulkanEngine durdurulurken kesinti yaşandı.");
            Thread.currentThread().interrupt();
        }
    }

    /**
     * UI thread'inin yüklenecek yeni bir sahneyi thread-safe olarak göndermesi için.
     * Motor bu sahneyi bir sonraki uygun karede yükleyecektir.
     * @param sceneData SceneBuilder tarafından oluşturulan CPU-tarafı verisi.
     */
    public void submitScene(BuiltCpuData sceneData) {
        sceneQueue.add(sceneData);
    }

    /**
     * UI thread'inin bir kamera güncellemesini thread-safe olarak göndermesi için.
     * Motor bu güncellemeyi bir sonraki uygun karede UBO'ya yazacaktır.
     * @param camera Yeni kamera durumu.
     */
    public void submitCameraUpdate(Camera camera) {
        cameraQueue.add(camera);
    }

    // --- VULKAN RENDER THREAD (VRT) MANTIĞI ---

    /**
     * Vulkan Motor Thread'i için ana giriş noktası.
     */
    @Override
    public void run() {
        try {
            initVulkan();
            mainLoop();
        } catch (Exception e) {
            System.err.println("FATAL (VRT): VulkanEngine thread'i çöktü!");
            e.printStackTrace();
            isRunning = false;
        } finally {
            cleanup();
            System.out.println("LOG (VRT): VulkanEngine kapatıldı.");
        }
    }

    /**
     * Tüm çekirdek Vulkan bileşenlerini, pipeline'ı ve statik kaynakları başlatır.
     */
    private void initVulkan() {
        System.out.println("LOG (VRT): VulkanEngine başlatılıyor...");
        if (!glfwInit()) {
            throw new RuntimeException("GLFW başlatılamadı!");
        }

        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();

        try (MemoryStack stack = stackPush()) {
            // Çekirdek kaynakları oluştur
            createCommandPoolAndBuffer(stack);
            createFence(stack);

            // Layout'a bağlı kaynakları oluştur
            createDescriptorSetLayout(stack); // Artık Kamera UBO için binding 4 içeriyor
            createPipelineLayout(stack);
            createComputePipeline(stack);

            // Render için kaynakları oluştur
            createComputeImage(stack);
            createStagingBuffer(stack);
            createCameraUbo(stack); // Yeni kamera tamponunu oluştur

            // Descriptor pool ve yeniden kullanılabilir tek descriptor set'i oluştur
            createDescriptorPool(stack);
            allocateDescriptorSet(stack);

            // ** HATA DÜZELTMESİ **: Compute image'ı başlangıçta *bir kez* GENERAL layout'una geçir.
            // Bu, 39 FPS ve bozuk görüntüye neden olan
            // VK_IMAGE_LAYOUT_UNDEFINED doğrulama hatasını düzeltir.
            transitionImageLayout(computeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

            // Descriptor set'i statik kaynaklarla güncelle (görüntü, kamera UBO)
            // Sahne tamponları (üçgenler, bvh) sahne yüklendiğinde güncellenecek.
            updateDescriptorSet(stack, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, cameraUbo);
        }
        System.out.println("LOG (VRT): VulkanEngine başarıyla başlatıldı.");
    }

    /**
     * Ana render döngüsü. VRT üzerinde çalışır.
     */
    private void mainLoop() {
        while (isRunning) {
            // UI thread'inden gelen tüm bekleyen komutları işle
            handleCommands();

            // Henüz sahne yüklenmediyse bekle.
            if (currentScene == null) {
                sleep(16); // ~60 FPS uyku
                continue;
            }

            // Bir kare render et
            FrameData frame = renderFrame();

            // Tamamlanan kareyi UI thread'ine yayınla
            frameQueue.set(frame);
        }
    }

    /**
     * UI kuyruklarından (sahne/kamera) gelen tüm bekleyen komutları işler.
     */
    private void handleCommands() {
        // 1. Yeni sahne var mı diye kontrol et
        BuiltCpuData newSceneData = sceneQueue.poll();
        if (newSceneData != null) {
            internalSwapScene(newSceneData);
        }

        // 2. Kamera güncellemesi var mı diye kontrol et
        Camera newCamera = cameraQueue.poll();
        if (newCamera != null) {
            internalUpdateCameraUBO(newCamera);
        }
    }

    /**
     * Varsa eski sahneyi güvenle yok eder ve yenisini yükler.
     */
    private void internalSwapScene(BuiltCpuData cpuData) {
        System.out.println("LOG (VRT): Sahne değiştiriliyor...");

        // Kaynakları değiştirmeden önce GPU'nun boşta olduğundan emin ol
        vkDeviceWaitIdle(device);

        // 1. Eski sahneyi yok et (eğer varsa)
        if (currentScene != null) {
            destroyGpuSceneData(currentScene);
            currentScene = null;
        }

        // 2. Yeni sahneyi yükle
        try (MemoryStack stack = stackPush()) {
            long triBuffer = VK_NULL_HANDLE, triMem = VK_NULL_HANDLE;
            long matBuffer = VK_NULL_HANDLE, matMem = VK_NULL_HANDLE;
            long bvhBuffer = VK_NULL_HANDLE, bvhMem = VK_NULL_HANDLE;

            // 2a. Üçgenleri Yükle
            long triBufferSize = cpuData.modelVertexData.remaining() * 4L;
            if (triBufferSize > 0) {
                LongBuffer pBuf = stack.mallocLong(1); LongBuffer pMem = stack.mallocLong(1);
                createHostVisibleBuffer(stack, triBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memByteBuffer(cpuData.modelVertexData), pBuf, pMem);
                triBuffer = pBuf.get(0); triMem = pMem.get(0);
            }
            memFree(cpuData.modelVertexData);

            // 2b. Materyalleri Yükle
            long matBufferSize = cpuData.modelMaterialData.remaining() * 4L;
            if (matBufferSize > 0) {
                LongBuffer pBuf = stack.mallocLong(1); LongBuffer pMem = stack.mallocLong(1);
                createHostVisibleBuffer(stack, matBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memByteBuffer(cpuData.modelMaterialData), pBuf, pMem);
                matBuffer = pBuf.get(0); matMem = pMem.get(0);
            }
            memFree(cpuData.modelMaterialData);

            // 2c. BVH Yükle
            long bvhBufferSize = cpuData.flatBvhData.remaining();
            if (bvhBufferSize > 0) {
                LongBuffer pBuf = stack.mallocLong(1); LongBuffer pMem = stack.mallocLong(1);
                createHostVisibleBuffer(stack, bvhBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cpuData.flatBvhData, pBuf, pMem);
                bvhBuffer = pBuf.get(0); bvhMem = pMem.get(0);
            }

            // 3. Descriptor set'i yeni tamponları gösterecek şekilde güncelle
            updateDescriptorSet(stack, triBuffer, matBuffer, bvhBuffer, cameraUbo);

            // 4. Yeni sahne verisini sakla
            currentScene = new GpuSceneData(triBuffer, triMem, matBuffer, matMem, bvhBuffer, bvhMem, cpuData.triangleCount);

            System.out.println("LOG (VRT): Yeni sahne yüklendi. Üçgen sayısı: " + currentScene.triangleCount);
        }
    }

    /**
     * YENİ: Kalıcı olarak haritalanmış kamera UBO'sunu yeni verilerle günceller.
     */
    private void internalUpdateCameraUBO(Camera camera) {
        if (cameraUboMapped == null) return;

        // Shader'daki std140 layout'u ile eşleşen yapı:
        // vec3 12 byte alır, 4 byte padding ile 16 byte'a tamamlanır.
        camera.getOrigin().store(0, cameraUboMapped);      // offset 0 (0-11 kullanılır)
        camera.getLowerLeft().store(16, cameraUboMapped);  // offset 16 (16-27 kullanılır)
        camera.getHorizontal().store(32, cameraUboMapped); // offset 32 (32-43 kullanılır)
        camera.getVertical().store(48, cameraUboMapped);   // offset 48 (48-59 kullanılır)

        // Bu veri artık bir sonraki renderFrame() için GPU tarafından görülebilir.
    }

    /**
     * Tek bir kare render eder ve piksel verisini bir FrameData nesnesinde döndürür.
     */
    private FrameData renderFrame() {
        try (MemoryStack stack = stackPush()) {
            // 1. Komutları kaydet
            recordComputeCommands(stack, currentScene);

            // 2. GPU'ya gönder ve bekle
            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(commandBuffer));

            vkResetFences(device, fence);
            vkQueueSubmit(computeQueue, submitInfo, fence);
            vkWaitForFences(device, fence, true, Long.MAX_VALUE); // ~16ms

            // 3. Görüntüyü staging buffer'dan oku
            long bufferSize = WIDTH * HEIGHT * 4;
            PointerBuffer pData = stack.mallocPointer(1);
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, pData);
            ByteBuffer pixelData = pData.getByteBuffer(0, (int) bufferSize);

            // 4. Unmap yapmadan *önce* veriyi yeni bir direct buffer'a kopyala
            ByteBuffer copyPixelData = ByteBuffer.allocateDirect((int)bufferSize);
            copyPixelData.put(pixelData);
            copyPixelData.flip();

            vkUnmapMemory(device, stagingBufferMemory);

            return new FrameData(copyPixelData);
        }
    }

    /**
     * Tek bir kare için tüm Vulkan komutlarını kaydeder.
     */
    private void recordComputeCommands(MemoryStack stack, GpuSceneData scene) {
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        vkBeginCommandBuffer(commandBuffer, beginInfo);

        // **HATA DÜZELTMESİ**: Bu bariyer artık layout'un GENERAL olduğunu varsayıyor.
        VkImageMemoryBarrier.Buffer imageBarrier1 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(0).dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_GENERAL) // UNDEFINED idi, hataya neden oluyordu
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier1.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, null, null, imageBarrier1); // Buffer barrier'ları vkDeviceWaitIdle hallettiği için kaldırıldı

        // 2. Pipeline ve descriptor'ları bağla
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(descriptorSet), null);

        // 3. Push Constant (üçgen sayısı) gönder
        ByteBuffer pushConstantData = stack.malloc(4);
        pushConstantData.putInt(0, scene.triangleCount);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantData);

        // 4. Shader'ı çalıştır
        vkCmdDispatch(commandBuffer, (WIDTH + 7) / 8, (HEIGHT + 7) / 8, 1);

        // 5. Bariyer: Görüntüyü kopyalama için GENERAL -> TRANSFER_SRC_OPTIMAL yap
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

        // 6. Görüntüyü VkImage'dan (VRAM) Staging Buffer'a (CPU-görünür) kopyala
        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                .bufferOffset(0)
                .bufferRowLength(0).bufferImageHeight(0)
                .imageSubresource(s -> s.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1))
                .imageOffset(o -> o.x(0).y(0).z(0))
                .imageExtent(e -> e.width(WIDTH).height(HEIGHT).depth(1));

        vkCmdCopyImageToBuffer(commandBuffer, computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, region);

        // 7. **HATA DÜZELTMESİ**: Görüntüyü *bir sonraki* kare için tekrar GENERAL'e döndür.
        VkImageMemoryBarrier.Buffer imageBarrier3 = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_TRANSFER_READ_BIT).dstAccessMask(0)
                .oldLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL) // UNDEFINED idi, bu yasadışı
                .image(computeImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        imageBarrier3.srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                0, null, null, imageBarrier3);

        vkEndCommandBuffer(commandBuffer);
    }

    /**
     * Tüm Vulkan kaynaklarını temizler.
     */
    private void cleanup() {
        System.out.println("LOG (VRT): VulkanEngine kaynakları temizleniyor...");
        if (device != null) {
            try {
                vkDeviceWaitIdle(device);
            } catch (Exception e) {
                System.err.println("WARN (VRT): vkDeviceWaitIdle sırasında istisna: " + e.getMessage());
            }
        }

        // UBO haritasını kaldır
        if (cameraUboMapped != null) {
            vkUnmapMemory(device, cameraUboMemory);
        }

        // Mevcut sahneyi yok et
        if (currentScene != null) {
            destroyGpuSceneData(currentScene);
        }

        // Pipeline kaynaklarını yok et
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, null);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, null);
        if (computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, computePipeline, null);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, null);
        if (computeShaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, computeShaderModule, null);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, null);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);

        // Görüntü/tampon kaynaklarını yok et
        if (computeImageView != VK_NULL_HANDLE) vkDestroyImageView(device, computeImageView, null);
        if (computeImage != VK_NULL_HANDLE) vkDestroyImage(device, computeImage, null);
        if (computeImageMemory != VK_NULL_HANDLE) vkFreeMemory(device, computeImageMemory, null);
        if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, stagingBuffer, null);
        if (stagingBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, stagingBufferMemory, null);
        if (cameraUbo != VK_NULL_HANDLE) vkDestroyBuffer(device, cameraUbo, null);
        if (cameraUboMemory != VK_NULL_HANDLE) vkFreeMemory(device, cameraUboMemory, null);

        // Çekirdek bileşenleri yok et
        if (device != null) vkDestroyDevice(device, null);
        if (debugMessenger != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        if (instance != null) vkDestroyInstance(instance, null);

        glfwTerminate();
    }

    /**
     * GpuSceneData tamponlarını yok etmek için yardımcı metot.
     */
    private void destroyGpuSceneData(GpuSceneData scene) {
        vkDestroyBuffer(device, scene.triangleBuffer, null);
        vkFreeMemory(device, scene.triangleBufferMemory, null);
        vkDestroyBuffer(device, scene.materialBuffer, null);
        vkFreeMemory(device, scene.materialBufferMemory, null);
        vkDestroyBuffer(device, scene.bvhBuffer, null);
        vkFreeMemory(device, scene.bvhBufferMemory, null);
    }

    // --- VULKAN YARDIMCI METOTLARI ---
    // (Eski VulkanRenderer'dan kopyalandı)

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
                throw new RuntimeException("Gerekli GLFW eklentileri bulunamadı!");
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
                throw new RuntimeException("Vulkan Instance oluşturulamadı!");
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
                throw new RuntimeException("Debug messenger kurulamadı!");
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
                throw new RuntimeException("Vulkan destekli GPU bulunamadı!");
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
                throw new RuntimeException("Uygun bir GPU bulunamadı!");
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
                throw new RuntimeException("Mantıksal cihaz oluşturulamadı!");
            }
            device = new VkDevice(pDevice.get(0), physicalDevice, createInfo);
            PointerBuffer pQueue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, pQueue);
            computeQueue = new VkQueue(pQueue.get(0), device);
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
            throw new RuntimeException("Compute image oluşturulamadı!");
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
            throw new RuntimeException("Image belleği ayrılamadı!");
        }
        computeImageMemory = pImageMemory.get(0);
        vkBindImageMemory(device, computeImage, computeImageMemory, 0);
        VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(computeImage).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1));
        LongBuffer pImageView = stack.mallocLong(1);
        if (vkCreateImageView(device, viewInfo, null, pImageView) != VK_SUCCESS) {
            throw new RuntimeException("Image view oluşturulamadı!");
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
            throw new RuntimeException("Staging buffer oluşturulamadı!");
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
            throw new RuntimeException("Staging buffer belleği ayrılamadı!");
        }
        stagingBufferMemory = pBufferMemory.get(0);
        vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);
    }

    /**
     * YENİ: Kamera UBO'sunu oluşturur.
     * Hızlı, kilitlenmesiz güncellemeler için kalıcı olarak haritalanır.
     */
    private void createCameraUbo(MemoryStack stack) {
        long bufferSize = 4 * 16; // 4 adet vec3, her biri vec4 (16 byte) olarak hizalanmış

        LongBuffer pBuffer = stack.mallocLong(1);
        LongBuffer pMemory = stack.mallocLong(1);

        createBuffer(stack, bufferSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                pBuffer, pMemory);

        cameraUbo = pBuffer.get(0);
        cameraUboMemory = pMemory.get(0);

        // Tamponu kalıcı olarak haritala
        PointerBuffer pData = stack.mallocPointer(1);
        vkMapMemory(device, cameraUboMemory, 0, bufferSize, 0, pData);
        cameraUboMapped = pData.getByteBuffer(0, (int)bufferSize);

        System.out.println("LOG (VRT): Kamera UBO'su oluşturuldu ve kalıcı olarak haritalandı.");
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
            throw new RuntimeException("Buffer oluşturulamadı!");
        }
        VkMemoryRequirements memRequirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, pBuffer.get(0), memRequirements);
        VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(memRequirements.size())
                .memoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits(), properties));
        if (vkAllocateMemory(device, allocInfo, null, pMemory) != VK_SUCCESS) {
            throw new RuntimeException("Buffer belleği ayrılamadı!");
        }
        vkBindBufferMemory(device, pBuffer.get(0), pMemory.get(0), 0);
    }

    /**
     * YENİDEN DÜZENLENDİ: Sadece DescriptorSetLayout'u oluşturur.
     * Artık Kamera UBO'su için binding 4'ü içerir.
     */
    private void createDescriptorSetLayout(MemoryStack stack) {
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(5, stack);

        bindings.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(2).binding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        bindings.get(3).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        // YENİ Binding 4: Kamera UBO
        bindings.get(4).binding(4).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(bindings);
        LongBuffer pSetLayout = stack.mallocLong(1);
        vkCreateDescriptorSetLayout(device, layoutInfo, null, pSetLayout);
        descriptorSetLayout = pSetLayout.get(0);
    }

    /**
     * YENİDEN DÜZENLENDİ: Sadece PipelineLayout'u oluşturur.
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
     * YENİDEN DÜZENLENDİ: Sadece ComputePipeline'ı oluşturur.
     */
    private void createComputePipeline(MemoryStack stack) {
        try {
            computeShaderModule = createShaderModule(loadShaderFromFile(SHADER_PATH));
        } catch (IOException e) {
            throw new RuntimeException("Shader yüklenemedi: " + e.getMessage());
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
            throw new RuntimeException("Compute pipeline oluşturulamadı!");
        }
        computePipeline = pComputePipeline.get(0);
    }

    /**
     * YENİDEN DÜZENLENDİ: Sadece DescriptorPool'u oluşturur.
     */
    private void createDescriptorPool(MemoryStack stack) {
        VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(3, stack);
        poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1);
        poolSizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(3); // Tri, Mat, BVH
        poolSizes.get(2).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).descriptorCount(1); // Kamera

        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                .pPoolSizes(poolSizes)
                .maxSets(1); // Sadece bir descriptor set'e ihtiyacımız var
        LongBuffer pDescriptorPool = stack.mallocLong(1);
        if (vkCreateDescriptorPool(device, poolInfo, null, pDescriptorPool) != VK_SUCCESS) {
            throw new RuntimeException("Descriptor pool oluşturulamadı!");
        }
        descriptorPool = pDescriptorPool.get(0);
    }

    /**
     * YENİDEN DÜZENLENDİ: Tek descriptor set'imizi ayırır (allocate).
     */
    private void allocateDescriptorSet(MemoryStack stack) {
        VkDescriptorSetAllocateInfo allocSetInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(descriptorPool)
                .pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pDescriptorSet = stack.mallocLong(1);
        if (vkAllocateDescriptorSets(device, allocSetInfo, pDescriptorSet) != VK_SUCCESS) {
            throw new RuntimeException("Descriptor set ayrılamadı!");
        }
        descriptorSet = pDescriptorSet.get(0);
    }

    /**
     * Yeniden kullanılabilir tek descriptor set'imizi günceller.
     * Bu artık init() ve internalSwapScene() tarafından çağrılır.
     */
    private void updateDescriptorSet(MemoryStack stack, long triangleBuffer, long materialBuffer, long bvhBuffer, long cameraBuffer) {

        VkDescriptorImageInfo.Buffer imageDescriptor = VkDescriptorImageInfo.calloc(1, stack)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .imageView(computeImageView);

        // Sahne tamponları
        VkDescriptorBufferInfo.Buffer triangleBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(triangleBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer materialBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(materialBuffer).offset(0).range(VK_WHOLE_SIZE);
        VkDescriptorBufferInfo.Buffer bvhBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(bvhBuffer).offset(0).range(VK_WHOLE_SIZE);

        // Kamera tamponu
        VkDescriptorBufferInfo.Buffer cameraBufferDescriptor = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(cameraBuffer).offset(0).range(VK_WHOLE_SIZE);

        VkWriteDescriptorSet.Buffer descriptorWrites = VkWriteDescriptorSet.calloc(5, stack);

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
        // YENİ Binding 4 (Kamera UBO)
        descriptorWrites.get(4).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET).dstSet(descriptorSet)
                .dstBinding(4).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                .descriptorCount(1).pBufferInfo(cameraBufferDescriptor);

        vkUpdateDescriptorSets(device, descriptorWrites, null);
    }

    private void createCommandPoolAndBuffer(MemoryStack stack) {
        VkCommandPoolCreateInfo cmdPoolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(computeQueueFamilyIndex);
        LongBuffer pCmdPool = stack.mallocLong(1);
        if (vkCreateCommandPool(device, cmdPoolInfo, null, pCmdPool) != VK_SUCCESS) {
            throw new RuntimeException("Command pool oluşturulamadı!");
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
            throw new RuntimeException("Fence oluşturulamadı");
        }
        fence = pFence.get(0);
    }

    /**
     * YENİ: Görüntü düzeni geçişleri için tek seferlik komut gönderen yardımcı metot.
     * Bu, UNDEFINED layout hatasını düzeltir.
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
                barrier.srcAccessMask(0);
                barrier.dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT);
                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            } else {
                throw new IllegalArgumentException("Desteklenmeyen layout geçişi!");
            }

            vkCmdPipelineBarrier(cmdBuffer, sourceStage, destStage, 0, null, null, barrier);
            vkEndCommandBuffer(cmdBuffer);

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(pCmdBuffer);
            vkQueueSubmit(computeQueue, submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(computeQueue); // Komutun bitmesini bekle
            vkFreeCommandBuffers(device, commandPool, pCmdBuffer);

            System.out.println("LOG (VRT): Başlangıç görüntü layout'u UNDEFINED -> GENERAL olarak değiştirildi.");
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
        throw new RuntimeException("Uygun bellek türü bulunamadı!");
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
                throw new RuntimeException("Shader module oluşturulamadı!");
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