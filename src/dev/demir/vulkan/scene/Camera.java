package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Sahnedeki sanal kamerayı temsil eder.
 * Pozisyonuna (origin) ve yönelimine (lookAt) göre viewport
 * vektörlerini (lower-left, horizontal, vertical) hesaplar.
 *
 * Bu veri, compute shader tarafından kullanılmak üzere bir
 * Uniform Buffer Object (UBO) aracılığıyla GPU'ya gönderilir.
 *
 * [cite_start]Hesaplama mantığı, compute.comp [cite: 1, 91-97] içindeki orijinal
 * [cite_start]hardcoded değerlere ve raytracer-java [cite: 2, 1-52] ilkelerine dayanmaktadır.
 */
public class Camera {

    // --- UBO için Önbelleğe Alınan Vektörler ---
    // Bunlar GPU'ya gönderilen nihai değerlerdir.
    private Vec3 origin;
    private Vec3 lowerLeft;
    private Vec3 horizontal;
    private Vec3 vertical;

    // --- Kamera Durum Özellikleri ---
    // Bunlar UBO vektörlerini yeniden hesaplamak için kullanılır
    private Vec3 lookAt;
    private final Vec3 vUp;
    private final double vfov; // Derece cinsinden dikey görüş alanı
    private final double aspectRatio;
    private final double focusDist;

    /**
     * Yeni bir dinamik kamera oluşturur.
     *
     * @param origin      Kamera pozisyonu
     * @param lookAt      Kameranın baktığı nokta
     * @param vUp         Kamera için "yukarı" yönü (örn: (0, 1, 0))
     * @param vfov        Dikey görüş alanı (derece cinsinden)
     * @param aspectRatio En-boy oranı (genişlik / yükseklik)
     */
    public Camera(Vec3 origin, Vec3 lookAt, Vec3 vUp, double vfov, double aspectRatio) {
        this.origin = origin;
        this.lookAt = lookAt;
        this.vUp = vUp;
        this.vfov = vfov;
        this.aspectRatio = aspectRatio;
        this.focusDist = 10.0; // compute.comp [cite: 1, 95] içindeki hardcoded focus_dist ile eşleşir

        recalculate();
    }

    /**
     * Kameranın mevcut durumuna (origin, lookAt, vb.) göre viewport
     * vektörlerini (horizontal, vertical, lowerLeft) yeniden hesaplar.
     * [cite_start]Bu mantık doğrudan compute.comp [cite: 1, 91-97] dosyasından alınmıştır.
     */
    private void recalculate() {
        double theta = Math.toRadians(this.vfov);
        double h = Math.tan(theta / 2.0);
        double viewportHeight = 2.0 * h;
        double viewportWidth = this.aspectRatio * viewportHeight;

        // Standart "lookAt" kamera modelini takip eder
        Vec3 w = origin.sub(lookAt).normalize();

        Vec3 u = Vec3.cross(vUp, w).normalize();
        Vec3 v = Vec3.cross(w, u);

        this.horizontal = u.mul(viewportWidth).mul(focusDist);
        this.vertical = v.mul(viewportHeight).mul(focusDist);
        this.lowerLeft = origin.sub(horizontal.div(2.0))
                .sub(vertical.div(2.0))
                .sub(w.mul(focusDist));
    }

    // --- VulkanEngine UBO için Getters ---
    // Bunlar VulkanEngine tarafından GPU tamponunu güncellemek için çağrılır.

    public Vec3 getOrigin() {
        return origin;
    }

    public Vec3 getLowerLeft() {
        return lowerLeft;
    }

    public Vec3 getHorizontal() {
        return horizontal;
    }

    public Vec3 getVertical() {
        return vertical;
    }

    // --- UI Kontrolleri için Setters ---
    // Bunlar VulkanApp (Swing UI) tarafından bir tuşa basıldığında çağrılır.

    /**
     * Kameranın pozisyonunu (origin) ayarlar ve viewport'u yeniden hesaplar.
     * @param newOrigin Kameranın yeni (x, y, z) pozisyonu.
     */
    public void setOrigin(Vec3 newOrigin) {
        this.origin = newOrigin;
        recalculate(); // Pozisyon değiştiğinde viewport vektörleri güncellenmelidir
    }

    /**
     * Kameranın baktığı noktayı ayarlar ve viewport'u yeniden hesaplar.
     * @param newLookAt Yeni (x, y, z) hedef.
     */
    public void setLookAt(Vec3 newLookAt) {
        this.lookAt = newLookAt;
        recalculate();
    }
}