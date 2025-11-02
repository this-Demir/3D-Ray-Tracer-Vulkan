package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Gelişmiş Phase 1: Sahne Nesnesi Modeli
 * Bu sınıf, sahnemizdeki tek bir nesnenin (bir .obj modelinin
 * bir "örneği" - instance'ı) tüm durumunu tutar.
 * Hangi modelin, nerede, ne boyutta ve ne renk olduğunu tanımlar.
 */
public class ModelInstance {

    // UI'da (örn. JList) gösterilecek isim
    private String displayName;

    // Yüklenecek .obj dosyasının yolu
    private final String modelPath;

    // Dönüşüm (Transform) bilgileri
    private Vec3 position;
    private Vec3 scale;
    // Not: Rotasyon (Dönüş) daha karmaşık matrisler gerektirdiğinden
    // şimdilik atlıyoruz, varsayılan (0,0,0) olacak.

    // Materyal bilgileri
    private Vec3 color;
    private float materialType;

    /**
     * Yeni bir model nesnesi oluşturur.
     * @param modelPath .obj dosyasının yolu (örn. "car.obj")
     * @param displayName UI'da görünecek isim (örn. "Araba")
     */
    public ModelInstance(String modelPath, String displayName) {
        this.modelPath = modelPath;
        this.displayName = displayName;

        // Varsayılan değerleri ayarla
        this.position = new Vec3(0, 0, 0);
        this.scale = new Vec3(1, 1, 1);
        this.color = new Vec3(0.8, 0.8, 0.8); // Varsayılan gri
        this.materialType = 0.0f; // 0.0f = Diffuse (Mat)
    }

    // --- Getters ---

    public String getDisplayName() { return displayName; }
    public String getModelPath() { return modelPath; }
    public Vec3 getPosition() { return position; }
    public Vec3 getScale() { return scale; }
    public Vec3 getColor() { return color; }
    public float getMaterialType() { return materialType; }

    // --- Setters (Bunları UI'dan güncellemek için kullanacağız) ---

    public void setDisplayName(String displayName) { this.displayName = displayName; }
    public void setPosition(Vec3 position) { this.position = position; }
    public void setScale(Vec3 scale) { this.scale = scale; }
    public void setColor(Vec3 color) { this.color = color; }
    public void setMaterialType(float materialType) { this.materialType = materialType; }

    /**
     * JList'te düzgün görünmesi için.
     */
    @Override
    public String toString() {
        return displayName;
    }
}