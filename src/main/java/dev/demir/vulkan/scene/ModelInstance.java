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
    private Animation animation = Animation.NONE;

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
        copy.animation = animation;
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

    /** Orbits the named object (or {@link Animation#WORLD_ORIGIN}) once every 'periodSeconds'. */
    public ModelInstance orbiting(String center, double periodSeconds) {
        animation = new Animation(center, periodSeconds, animation.spinPeriod());
        return this;
    }

    /** Turns around its own vertical axis once every 'periodSeconds'. */
    public ModelInstance spinning(double periodSeconds) {
        animation = new Animation(animation.orbitCenter(), animation.orbitPeriod(), periodSeconds);
        return this;
    }

    // --- Getters / setters ---

    public String getDisplayName() { return displayName; }
    public String getModelPath() { return modelPath; }
    public Vec3 getPosition() { return position; }
    public Vec3 getRotation() { return rotation; }
    public Vec3 getScale() { return scale; }
    public Material getMaterial() { return material; }
    public Animation getAnimation() { return animation; }
    public void setAnimation(Animation animation) { this.animation = animation; }

    public void setDisplayName(String displayName) { this.displayName = displayName; }
    public void setPosition(Vec3 position) { this.position = position; }
    public void setRotation(Vec3 rotation) { this.rotation = rotation; }
    public void setScale(Vec3 scale) { this.scale = scale; }
    public void setMaterial(Material material) { this.material = material; }

    /** Shown in the object list. */
    @Override
    public String toString() {
        return displayName;
    }
}
