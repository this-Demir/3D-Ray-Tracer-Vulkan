package dev.demir.vulkan.renderer;

import java.nio.ByteBuffer;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;

// Bu import, VulkanRenderer.java dosyanızın içinde olmalı.
// Eğer VulkanRenderer'da 'device' public değilse (olmamalı),
// bu importu VulkanRenderer'a taşıyıp orada vkDeviceWaitIdle çağırmalısınız.
// Şimdilik basitlik adına burada bırakıyorum, ancak ideal olanı renderer'dadır.
import static org.lwjgl.vulkan.VK10.vkDeviceWaitIdle;

/**
 * Bu sınıf VRT'yi (Vulkan Render Thread) temsil eder.
 * 'RenderService' tarafından ayrı bir Thread'de başlatılır.
 * VulkanRenderer'ı yönetir ve thread-safe kuyruklar aracılığıyla
 * FXAT ve SRT ile iletişim kurar.
 */
public class RenderLoop implements Runnable {

    // Render motoru (Sadece bu thread tarafından kullanılır)
    private final VulkanRenderer renderer = new VulkanRenderer();

    // --- Thread-Safe İletişim Kuyrukları ---

    // VRT -> FXAT: Tamamlanan kareleri (frame) iletmek için.
    private final AtomicReference<FrameData> latestFrame;

    // FXAT -> VRT: Yeni kamera/ayarları iletmek için.
    // private final ConcurrentLinkedQueue<Camera> cameraQueue = new ConcurrentLinkedQueue<>(); (Phase 3'te eklenecek)
    // private final ConcurrentLinkedQueue<RenderSettings> settingsQueue = new ConcurrentLinkedQueue<>(); (Phase 3'te eklenecek)

    // SRT -> VRT: Yüklenmeye hazır yeni sahne verisini iletmek için.
    private final ConcurrentLinkedQueue<BuiltCpuData> sceneSwapQueue = new ConcurrentLinkedQueue<>();


    // Durum (State)
    private volatile boolean isRunning = true;
    private GpuSceneData currentGpuData = null;

    /**
     * @param latestFrame VRT'nin tamamlanan kareleri FXAT'ye iletmesi için thread-safe referans.
     */
    public RenderLoop(AtomicReference<FrameData> latestFrame) {
        this.latestFrame = latestFrame;
    }

    @Override
    public void run() {
        try {
            // 1. Motoru Başlat
            renderer.init();

            // 2. Ana Render Döngüsü
            while (isRunning) {

                // 2a. Kuyrukları kontrol et (Yeni sahne var mı?)
                checkQueues();

                // 2b. Sahne verisi hazırsa render al
                if (currentGpuData != null) {
                    ByteBuffer frameBytes = renderer.renderFrame(currentGpuData);

                    // 2c. Yeni kareyi FXAT için yayınla
                    // (Önceki kareyi eziyoruz, FXAT'nin her zaman en son kareyi alması için)
                    latestFrame.set(new FrameData(frameBytes));
                } else {
                    // Sahne yüklenene kadar bekle (ilk kare)
                    Thread.sleep(16);
                }
            }

        } catch (Exception e) {
            System.err.println("FATAL (VRT): Render thread crashed!");
            e.printStackTrace();
            isRunning = false;
        } finally {
            // 3. Temizlik
            // Son bir kez daha kuyrukları kontrol edip kalanları temizle
            checkQueues();
            // Kalan GPU verisini temizle
            if (currentGpuData != null) {
                renderer.destroyGpuSceneData(currentGpuData);
            }
            // Motoru yok et
            renderer.destroy();
        }
    }

    /**
     * Döngünün başında VRT tarafından çağrılır.
     * Diğer thread'lerden gelen istekleri işler.
     */
    private void checkQueues() {
        // 1. Sahne Değişim İsteği (SRT -> VRT)
        BuiltCpuData newCpuData = sceneSwapQueue.poll();
        if (newCpuData != null) {

            // Cihazın boşta olduğundan emin ol
            // Bu satır VulkanRenderer.java'ya taşınmalı (uploadAndSwapScene içine)
            // vkDeviceWaitIdle(renderer.device);
            // Şimdilik, renderer.device'ın public olduğunu varsayıyoruz (kötü pratik)
            // Ya da daha iyisi, renderer'da vkDeviceWaitIdle çağıran bir metot yapalım.
            // Şimdilik bu kısmı renderer.uploadAndSwapScene içine taşıyalım.

            // Eski GPU verisini yok et
            if (currentGpuData != null) {
                renderer.destroyGpuSceneData(currentGpuData);
            }

            // Yeni veriyi GPU'ya yükle ve 'current' olarak ayarla
            currentGpuData = renderer.uploadAndSwapScene(newCpuData);
        }

        // 2. Kamera Güncelleme İsteği (FXAT -> VRT)
        // (Phase 4'te eklenecek)
        // Camera newCam = cameraQueue.poll();
        // if (newCam != null) {
        //     renderer.updateCameraUBO(newCam);
        // }
    }

    // --- Diğer Thread'ler Tarafından Çağrılan Metodlar ---
    // (RenderService aracılığıyla)

    /**
     * (SRT'den çağrılır -> RenderService aracılığıyla)
     * VRT'ye yeni bir sahne verisinin yükleneceğini bildirir.
     */
    public void submitNewScene(BuiltCpuData cpuData) {
        sceneSwapQueue.add(cpuData);
    }

    /**
     * (FXAT'den çağrılır)
     * Render döngüsünü güvenli bir şekilde durdurur.
     */
    public void stop() {
        this.isRunning = false;
    }
}