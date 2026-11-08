package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

import java.util.List;

/**
 * Immutable snapshot of the scene, handed from the UI / scene rebuild thread to the
 * Vulkan render thread. Holds no Vulkan objects. Object transforms are evaluated for a
 * point in time with {@link #transformsAt(double)}, which applies the animations.
 * compactObject (black hole or neutron star) is null for scenes without one.
 */
public record SceneDescription(List<Instance> instances, CompactObject compactObject) {

    /** Orbit parent values besides an instance index. */
    public static final int NO_ORBIT = -1;
    public static final int ORBIT_WORLD_ORIGIN = -2;

    public SceneDescription {
        instances = List.copyOf(instances);
    }

    /**
     * One placed copy of a geometry.
     *
     * @param geometry    shared shape data (one BLAS per distinct geometry on the GPU)
     * @param position    placement at time 0
     * @param rotation    Euler angles in degrees (X, then Y, then Z) at time 0
     * @param scale       per-axis scale
     * @param material    surface material
     * @param orbitParent index of the instance this one orbits, {@link #ORBIT_WORLD_ORIGIN} or {@link #NO_ORBIT}
     * @param orbitPeriod seconds per orbit (0 = none)
     * @param spinPeriod  seconds per turn around the own vertical axis (0 = none)
     */
    public record Instance(Geometry geometry, Vec3 position, Vec3 rotation, Vec3 scale, Material material,
                           int orbitParent, double orbitPeriod, double spinPeriod) {

        public boolean isAnimated() {
            return (orbitParent != NO_ORBIT && orbitPeriod != 0) || spinPeriod != 0;
        }
    }

    public boolean isAnimated() {
        for (Instance instance : instances) {
            if (instance.isAnimated()) return true;
        }
        return false;
    }

    public long triangleCount() {
        long total = 0;
        for (Instance instance : instances) {
            total += instance.geometry().triangleCount();
        }
        return total;
    }

    /** Object-to-world transforms (row-major 3x4) of all instances at time t (seconds). */
    public float[][] transformsAt(double t) {
        Vec3[] positions = new Vec3[instances.size()];
        float[][] transforms = new float[instances.size()][];
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            Vec3 rotation = instance.rotation();
            if (instance.spinPeriod() != 0) {
                rotation = new Vec3(rotation.x, rotation.y + 360.0 * t / instance.spinPeriod(), rotation.z);
            }
            transforms[i] = Transforms.compose(positionAt(i, t, positions, 0), rotation, instance.scale());
        }
        return transforms;
    }

    /**
     * Position of instance i at time t. An orbiting object keeps its time-0 offset from its
     * center, rotated around the vertical axis; the center may itself be orbiting.
     */
    private Vec3 positionAt(int i, double t, Vec3[] memo, int depth) {
        if (memo[i] != null) return memo[i];
        Instance instance = instances.get(i);
        Vec3 position = instance.position();
        boolean orbits = instance.orbitParent() != NO_ORBIT && instance.orbitPeriod() != 0;
        if (orbits && depth < instances.size()) { // depth guard against orbit cycles
            Vec3 center0 = Vec3.ZERO;
            Vec3 centerNow = Vec3.ZERO;
            if (instance.orbitParent() >= 0) {
                center0 = instances.get(instance.orbitParent()).position();
                centerNow = positionAt(instance.orbitParent(), t, memo, depth + 1);
            }
            Vec3 offset = position.sub(center0);
            double angle = 2 * Math.PI * t / instance.orbitPeriod();
            double cos = Math.cos(angle), sin = Math.sin(angle);
            // Counter-clockwise when seen from above (+Y).
            Vec3 rotated = new Vec3(offset.x * cos + offset.z * sin, offset.y, -offset.x * sin + offset.z * cos);
            position = centerNow.add(rotated);
        }
        memo[i] = position;
        return position;
    }
}
