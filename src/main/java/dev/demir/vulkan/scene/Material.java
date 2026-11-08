package dev.demir.vulkan.scene;

/**
 * Surface material of an instance (immutable).
 *
 * @param type      how light interacts with the surface
 * @param albedo    r, g, b reflectance (or tint for glass), each 0..1
 * @param roughness metal only: 0 = perfect mirror, higher = blurrier reflections (0..1)
 * @param emission  emissive only: brightness multiplier for albedo (the light's color)
 * @param ior       glass only: index of refraction (1.5 for glass, 1.33 for water)
 */
public record Material(Type type, float[] albedo, float roughness, float emission, float ior) {

    /** Must match the MAT_* constants in pathtrace.comp. */
    public enum Type {
        DIFFUSE("Matte"),
        METAL("Metal"),
        GLASS("Glass"),
        EMISSIVE("Light (emissive)");

        private final String label;

        Type(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public Material {
        albedo = albedo.clone();
    }

    public static Material diffuse(float r, float g, float b) {
        return new Material(Type.DIFFUSE, new float[]{r, g, b}, 0f, 0f, 1.5f);
    }

    public static Material metal(float r, float g, float b, float roughness) {
        return new Material(Type.METAL, new float[]{r, g, b}, roughness, 0f, 1.5f);
    }

    public static Material mirror() {
        return metal(0.95f, 0.95f, 0.95f, 0f);
    }

    public static Material glass(float ior) {
        return new Material(Type.GLASS, new float[]{1f, 1f, 1f}, 0f, 0f, ior);
    }

    public static Material light(float r, float g, float b, float strength) {
        return new Material(Type.EMISSIVE, new float[]{r, g, b}, 0f, strength, 1.5f);
    }

    public Material withAlbedo(float r, float g, float b) {
        return new Material(type, new float[]{r, g, b}, roughness, emission, ior);
    }

    public Material withType(Type newType) {
        return new Material(newType, albedo, roughness, emission, ior);
    }

    public Material withRoughness(float value) {
        return new Material(type, albedo, value, emission, ior);
    }

    public Material withEmission(float value) {
        return new Material(type, albedo, roughness, value, ior);
    }

    public Material withIor(float value) {
        return new Material(type, albedo, roughness, emission, value);
    }
}
