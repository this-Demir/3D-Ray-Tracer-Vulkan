package dev.demir.vulkan;

import dev.demir.vulkan.engine.VulkanEngine; // Yeni akıllı motoru import et
import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.renderer.SceneBuilder;
import dev.demir.vulkan.scene.Camera;
import dev.demir.vulkan.util.Vec3;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Ana Uygulama Sınıfı (UI Thread - EDT).
 * Artık Vulkan mantığı içermez. Sadece Java Swing arayüzünü yönetir
 * ve VulkanEngine'i (VRT) ve SceneBuilder'ı (SRT) koordine eder.
 */
public class VulkanApp {

    // --- Konfigürasyon ---
    private static final int RENDER_WIDTH = 1280;
    private static final int RENDER_HEIGHT = 720;
    private static final String MODEL_PATH = "plane.obj";

    // --- Swing UI ---
    private JFrame frame;
    private JLabel imageLabel;
    private BufferedImage bufferedImage;
    private byte[] swizzleBuffer; // Vulkan (RGBA) -> Swing (BGR)

    // --- Motor Çekirdeği ---
    private final AtomicReference<FrameData> latestFrame = new AtomicReference<>();
    private final VulkanEngine vulkanEngine; // Yeni motorumuz
    private final SceneBuilder sceneBuilder = new SceneBuilder();
    private final Camera camera; // Dinamik kamera nesnemiz

    // --- Durum (State) ---
    private long lastFrameTime = System.nanoTime();
    private int frameCount = 0;
    private volatile boolean sceneBuildInProgress = false;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                new VulkanApp().run();
            } catch (Exception e) {
                e.printStackTrace();
                JOptionPane.showMessageDialog(null, "Kritik Hata: " + e.getMessage(), "Hata", JOptionPane.ERROR_MESSAGE);
                System.exit(1);
            }
        });
    }

    public VulkanApp() {
        // 1. Motoru oluştur, iletişim kuyruğunu (latestFrame) ver
        this.vulkanEngine = new VulkanEngine(latestFrame);

        // 2. Kamerayı oluştur
        this.camera = new Camera(
                new Vec3(0, 0, 15), // origin
                new Vec3(0, 0, 0),  // lookAt
                new Vec3(0, 1, 0),  // vUp
                20.0,               // vfov
                (double)RENDER_WIDTH / RENDER_HEIGHT // aspect ratio
        );
    }

    public void run() {
        // 1. Swing UI'ı kur
        frame = new JFrame("Vulkan BVH Ray Tracer (Akıllı Motor)");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        bufferedImage = new BufferedImage(RENDER_WIDTH, RENDER_HEIGHT, BufferedImage.TYPE_3BYTE_BGR);
        imageLabel = new JLabel(new ImageIcon(bufferedImage));
        frame.getContentPane().add(imageLabel, BorderLayout.CENTER);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        swizzleBuffer = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();

        // 2. VulkanEngine Thread'ini (VRT) başlat
        System.out.println("LOG (UI): VulkanEngine başlatılıyor...");
        vulkanEngine.start();

        // 3. Başlangıç sahne inşasını (SRT) tetikle
        rebuildSceneAsync(MODEL_PATH);

        // 4. Başlangıç kamera durumunu motora gönder
        vulkanEngine.submitCameraUpdate(camera);

        // 5. UI Timer'ı (Swing'in 'AnimationTimer'ı) başlat
        Timer swingTimer = new Timer(16, (e) -> updateUI()); // ~60 FPS
        swingTimer.start();

        // 6. Klavye Kontrollerini ayarla
        setupControls();

        // 7. Pencere kapanışını yönet
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                System.out.println("LOG (UI): Motor durduruluyor...");
                swingTimer.stop();
                vulkanEngine.stop(); // Motora temizlik yapmasını söyle
            }
        });
    }

    /**
     * Swing Timer tarafından çalıştırılan ana UI döngüsü.
     */
    private void updateUI() {
        FrameData frameData = latestFrame.getAndSet(null);

        if (frameData != null) {
            updateBufferedImage(frameData.pixelData);
            imageLabel.repaint();
            frameCount++;
        }

        long now = System.nanoTime();
        if (now - lastFrameTime >= 1_000_000_000) {
            frame.setTitle(String.format("Vulkan BVH Ray Tracer (Akıllı Motor) | %d FPS", frameCount));
            frameCount = 0;
            lastFrameTime = now;
        }
    }

    /**
     * SceneBuilder'a (SRT) bir sahne inşa etme görevi verir.
     */
    public void rebuildSceneAsync(String modelPath) {
        if (sceneBuildInProgress) return;
        sceneBuildInProgress = true;
        System.out.println("LOG (UI): Asenkron sahne inşası (SRT) tetikleniyor...");

        CompletableFuture
                .supplyAsync(() -> sceneBuilder.buildScene(modelPath))
                .whenCompleteAsync((builtCpuData, error) -> {
                    if (error != null) {
                        error.printStackTrace();
                        JOptionPane.showMessageDialog(frame, "Sahne inşası başarısız: " + error.getMessage(), "SRT Hatası", JOptionPane.ERROR_MESSAGE);
                    } else {
                        System.out.println("LOG (UI): SRT bitti. Yeni sahne motora gönderiliyor.");
                        vulkanEngine.submitScene(builtCpuData);
                    }
                    sceneBuildInProgress = false;
                }, SwingUtilities::invokeLater); // Geri bildirimi Swing thread'inde çalıştır
    }

    /**
     * Kamera için klavye kontrollerini ayarlar.
     */
    private void setupControls() {
        frame.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                boolean cameraChanged = false;
                Vec3 origin = camera.getOrigin();

                switch(e.getKeyCode()) {
                    case KeyEvent.VK_W:
                        camera.setOrigin(origin.add(new Vec3(0, 0, -6.5))); // İleri
                        cameraChanged = true;
                        break;
                    case KeyEvent.VK_S:
                        camera.setOrigin(origin.add(new Vec3(0, 0, 15))); // Geri
                        cameraChanged = true;
                        break;
                    case KeyEvent.VK_A:
                        camera.setOrigin(origin.add(new Vec3(-5.5, 0, 0))); // Sola
                        cameraChanged = true;
                        break;
                    case KeyEvent.VK_D:
                        camera.setOrigin(origin.add(new Vec3(5.5, 0, 0))); // Sağa
                        cameraChanged = true;
                        break;
                    case KeyEvent.VK_SPACE:
                        camera.setOrigin(origin.add(new Vec3(0, 0.5, 0))); // Yukarı
                        cameraChanged = true;
                        break;
                    case KeyEvent.VK_CONTROL:
                        camera.setOrigin(origin.add(new Vec3(0, -0.5, 0))); // Aşağı
                        cameraChanged = true;
                        break;
                }

                if (cameraChanged) {
                    // Güncellenmiş kamera durumunu motora gönder
                    vulkanEngine.submitCameraUpdate(camera);
                }
            }
        });
        frame.setFocusable(true); // JFrame'in tuş olaylarını alabilmesini sağla
    }

    /**
     * Vulkan'ın ByteBuffer'ını (RGBA) Swing'in BufferedImage'ına (BGR) kopyalar.
     */
    private void updateBufferedImage(ByteBuffer pixelData) {
        pixelData.rewind();
        for (int i = 0; i < (RENDER_WIDTH * RENDER_HEIGHT); i++) {
            // Vulkan (RGBA) -> Swing (BGR)
            swizzleBuffer[i * 3 + 0] = pixelData.get(i * 4 + 2); // Mavi
            swizzleBuffer[i * 3 + 1] = pixelData.get(i * 4 + 1); // Yeşil
            swizzleBuffer[i * 3 + 2] = pixelData.get(i * 4 + 0); // Kırmızı
            // Alfa (i*4+3) atlanır
        }
    }
}