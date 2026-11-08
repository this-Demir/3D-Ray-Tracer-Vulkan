package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * One object in the editable scene: which geometry (a model file or a built-in primitive),
 * where it is (position, rotation, scale) and what it is made of (material).
 * Owned and mutated by the UI thread; the renderer only sees immutable snapshots.
 */
public class ModelInstance {

    private String displayName;
    private final String modelPath; // file path, or SphereGeometry.KEY for the analytic sphere

    private Vec3 position = new Vec3(0, 0, 0);
    private Vec3 rotation = new Vec3(0, 0, 0); // Euler angles in degrees, applied X, then Y, then Z
    private Vec3 scale = new Vec3(1, 1, 1);
    private Material material = Material.diffuse(0.8f, 0.8f, 0.8f);

    /**
     * @param modelPath   .obj file path, or {@link SphereGeometry#KEY}
     * @param displayName name shown in the UI
     */
    public ModelInstance(String modelPath, String displayName) {
        this.modelPath = modelPath;
        this.displayName = displayName;
    }

    /** Copy with the same geometry, placement and material. */
    public ModelInstance copy(String newName) {
        ModelInstance copy = new ModelInstance(modelPath, newName);
        copy.position = position;
        copy.rotation = rotation;
        copy.scale = scale;
        copy.material = material;
        return copy;
    }

    // --- Fluent setup (used by scene presets) ---

    public ModelInstance at(double x, double y, double z) {
        position = new Vec3(x, y, z);
        return this;
    }

    public ModelInstance rotated(double x, double y, double z) {
        rotation = new Vec3(x, y, z);
        return this;
    }

    public ModelInstance scaled(double uniform) {
        scale = new Vec3(uniform, uniform, uniform);
        return this;
    }

    public ModelInstance scaled(double x, double y, double z) {
        scale = new Vec3(x, y, z);
        return this;
    }

    public ModelInstance with(Material newMaterial) {
        material = newMaterial;
        return this;
    }

    // --- Getters / setters ---

    public String getDisplayName() { return displayName; }
    public String getModelPath() { return modelPath; }
    public Vec3 getPosition() { return position; }
    public Vec3 getRotation() { return rotation; }
    public Vec3 getScale() { return scale; }
    public Material getMaterial() { return material; }

    public void setDisplayName(String displayName) { this.displayName = displayName; }
    public void setPosition(Vec3 position) { this.position = position; }
    public void setRotation(Vec3 rotation) { this.rotation = rotation; }
    public void setScale(Vec3 scale) { this.scale = scale; }
    public void setMaterial(Material material) { this.material = material; }

    /**
     * Object-to-world transform as a row-major 3x4 matrix (the layout Vulkan's
     * VkTransformMatrixKHR expects): M = Translate * Rz * Ry * Rx * Scale.
     */
    public float[] toTransform3x4() {
        return Transforms.compose(position, rotation, scale);
    }

    /** Shown in the object list. */
    @Override
    public String toString() {
        return displayName;
    }
}
