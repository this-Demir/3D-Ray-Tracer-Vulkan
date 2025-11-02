package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Advanced Phase 2.5: Updated Camera
 * This class now calculates the viewport vectors (lower_left_corner,
 * horizontal, vertical) that the 'raytracer-in-one-weekend' style
 * shader (the one you provided) expects.
 */
public class Camera {

    // --- Mevcut Durum ---
    private Vec3 origin;
    private Vec3 lookAt;
    private Vec3 vUp;
    private double vfov; // vertical field-of-view in degrees
    private double aspect_ratio;

    // --- YENİ: Shader'ın İhtiyaç Duyduğu Hesaplanmış Vektörler ---
    private Vec3 lower_left_corner;
    private Vec3 horizontal;
    private Vec3 vertical;

    public Camera(Vec3 origin, Vec3 lookAt, Vec3 vUp, double vfov, double aspect_ratio) {
        this.origin = origin;
        this.lookAt = lookAt;
        this.vUp = vUp;
        this.vfov = vfov;
        this.aspect_ratio = aspect_ratio;

        // Kamerayı oluştururken bu değerleri hesapla
        recalculateViewport();
    }

    /**
     * NEW: Calculates the viewport vectors based on the camera's state.
     */
    private void recalculateViewport() {
        double theta = Math.toRadians(vfov);
        double h = Math.tan(theta / 2.0);
        double viewport_height = 2.0 * h;
        double viewport_width = aspect_ratio * viewport_height;

        // Kameranın 3D eksenlerini hesapla (u, v, w)
        // w: Kameranın baktığı yönün tersi (Z ekseni)
        Vec3 w = Vec3.unitVector(origin.sub(lookAt));
        // u: Kameranın "sağ" vektörü (X ekseni)
        Vec3 u = Vec3.unitVector(Vec3.cross(vUp, w));
        // v: Kameranın "yukarı" vektörü (Y ekseni)
        Vec3 v = Vec3.cross(w, u);

        // Viewport'u (görüş alanı) oluşturan vektörleri hesapla
        this.horizontal = u.multiply(viewport_width);
        this.vertical = v.multiply(viewport_height);

        // Viewport'un sol alt köşesini hesapla
        // origin - (horizontal/2) - (vertical/2) - w
        this.lower_left_corner = origin
                .sub(horizontal.div(2.0))
                .sub(vertical.div(2.0))
                .sub(w);

        // Orijini de güncellememiz gerekiyor (bu modelde origin sabittir)
        // Not: Bu hesaplama 'origin'i değiştirmez, sadece LL köşe için kullanır.
    }

    // --- Getters (Shader'ın ihtiyacı olanlar) ---

    public Vec3 getLowerLeft() { return lower_left_corner; }
    public Vec3 getHorizontal() { return horizontal; }
    public Vec3 getVertical() { return vertical; }


    // --- Setters (Kamerayı hareket ettirmek için) ---

    public Vec3 getOrigin() {
        return origin;
    }

    /**
     * Kameranın pozisyonunu günceller ve viewport'u yeniden hesaplar.
     */
    public void setOrigin(Vec3 origin) {
        this.origin = origin;
        // Kamera hareket ettiğinde, viewport'u yeniden hesaplamamız gerekir!
        recalculateViewport();
    }

    // (Gelecekte lookAt'i de güncelleyebiliriz, şimdilik bu yeterli)
}