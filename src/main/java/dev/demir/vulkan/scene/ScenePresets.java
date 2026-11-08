package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

import java.util.ArrayList;
import java.util.List;

/**
 * The built-in scenes. All models come from the repository's objects/ folder.
 */
public final class ScenePresets {

    public static final String HUMAN = "./objects/FinalBaseMesh.obj";   // ~20.7 tall, feet at y = 0
    public static final String QUAD = "./objects/ground_plane.obj";     // 2 x 2 in the XZ plane
    public static final String CUBE = "./objects/cube.obj";             // 1 x 1 x 1, centered
    public static final String AIRPLANE = "./objects/plane.obj";        // ~1580 wide, Z up
    public static final String MOON = "./objects/Moon.obj";             // radius ~1.75
    public static final String BLOB = "./objects/white_mesh.obj";       // ~2 across, centered
    public static final String SPHERE = SphereGeometry.KEY;

    private ScenePresets() {
    }

    public static List<ScenePreset> all() {
        return List.of(custom(), mirrorRoom(), eclipse(), showRoom());
    }

    /** The free editor: a ground plane and a figure under the sky. */
    public static ScenePreset custom() {
        return new ScenePreset("Custom", () -> List.of(
                new ModelInstance(QUAD, "Ground").at(0, -10, 0).scaled(150, 1, 150)
                        .with(Material.diffuse(0.5f, 0.5f, 0.5f)),
                new ModelInstance(HUMAN, "Human").at(0, -10, 0).scaled(1.5)
                        .with(Material.metal(0.6f, 0.7f, 0.1f, 0f))
        ), new Vec3(-25, 30, 140), new Vec3(0, 0, 0), 20, RenderSettings.DEFAULT);
    }

    /**
     * A closed room with mirror walls and a small light: reflections repeat "forever",
     * so it needs many bounces and no sky.
     */
    public static ScenePreset mirrorRoom() {
        return new ScenePreset("Mirror Room", () -> {
            List<ModelInstance> objects = new ArrayList<>();
            Material mirror = Material.mirror();
            objects.add(new ModelInstance(QUAD, "Floor").scaled(20, 1, 20)
                    .with(Material.diffuse(0.35f, 0.35f, 0.38f)));
            objects.add(new ModelInstance(QUAD, "Ceiling").at(0, 24, 0).scaled(20, 1, 20)
                    .with(Material.diffuse(0.8f, 0.8f, 0.8f)));
            objects.add(new ModelInstance(QUAD, "Mirror Back").at(0, 12, -20).rotated(90, 0, 0).scaled(20, 1, 12).with(mirror));
            objects.add(new ModelInstance(QUAD, "Mirror Front").at(0, 12, 20).rotated(90, 0, 0).scaled(20, 1, 12).with(mirror));
            objects.add(new ModelInstance(QUAD, "Mirror Left").at(-20, 12, 0).rotated(0, 0, 90).scaled(12, 1, 20).with(mirror));
            objects.add(new ModelInstance(QUAD, "Mirror Right").at(20, 12, 0).rotated(0, 0, 90).scaled(12, 1, 20).with(mirror));
            objects.add(new ModelInstance(SPHERE, "Lamp").at(0, 21, 0).scaled(1.2)
                    .with(Material.light(1f, 0.85f, 0.6f, 60f)));
            objects.add(new ModelInstance(HUMAN, "Figure").at(0, 0, -4).rotated(0, 20, 0).scaled(0.6)
                    .with(Material.diffuse(0.75f, 0.08f, 0.06f)));
            objects.add(new ModelInstance(SPHERE, "Glass Ball").at(7, 2.5, 3).scaled(2.5).with(Material.glass(1.5f)));
            objects.add(new ModelInstance(SPHERE, "Gold Ball").at(-7, 2, 4).scaled(2)
                    .with(Material.metal(1f, 0.78f, 0.34f, 0.05f)));
            return objects;
        }, new Vec3(-6, 13, 17), new Vec3(0, 7, -6), 65, new RenderSettings(false, 0.5f, 30, 8192));
    }

    /**
     * A sun and planets in space: the moon sits between the sun and the blue planet and casts
     * its shadow on it. Lit only by the sun, so shadow rays towards the sun do the work.
     */
    public static ScenePreset eclipse() {
        return new ScenePreset("Eclipse", () -> List.of(
                new ModelInstance(SPHERE, "Sun").scaled(10).with(Material.light(1f, 0.9f, 0.7f, 20f)),
                new ModelInstance(SPHERE, "Earth").at(48, 0, 0).scaled(4).with(Material.diffuse(0.15f, 0.35f, 0.85f)),
                // Umbra length = r_moon * d / (r_sun - r_moon) = 2 * 41 / 8 ~ 10 > 41 -> 44: it reaches Earth.
                new ModelInstance(SPHERE, "Moon").at(41, 0.5, 0).scaled(2).with(Material.diffuse(0.7f, 0.7f, 0.68f)),
                new ModelInstance(SPHERE, "Mars").at(-30, -2, 22).scaled(3).with(Material.diffuse(0.8f, 0.25f, 0.1f)),
                new ModelInstance(SPHERE, "Venus").at(-20, 3, -26).scaled(2.4).with(Material.diffuse(0.85f, 0.75f, 0.45f)),
                new ModelInstance(MOON, "Asteroid").at(22, 6, 26).rotated(30, 40, 0).scaled(0.8)
                        .with(Material.diffuse(0.5f, 0.45f, 0.4f))
        // Camera between the sun and Earth, looking at Earth's sunlit side where the moon's shadow falls.
        ), new Vec3(26, 8, 22), new Vec3(45, 0.5, 0), 30, new RenderSettings(false, 0f, 6, 8192));
    }

    /** A display floor with models, a mirror wall behind them and the sky as light. */
    public static ScenePreset showRoom() {
        return new ScenePreset("ShowRoom", () -> List.of(
                new ModelInstance(CUBE, "Platform").at(0, -0.5, 0).scaled(70, 1, 34)
                        .with(Material.diffuse(0.75f, 0.75f, 0.75f)),
                new ModelInstance(QUAD, "Mirror Wall").at(0, 10, -17).rotated(90, 0, 0).scaled(35, 1, 10)
                        .with(Material.mirror()),
                // Model is Z-up: rotate it upright; its lowest point (z = 29) then sits on the platform.
                new ModelInstance(AIRPLANE, "Airplane").at(0, -0.45, -3).rotated(-90, 0, 0).scaled(0.016)
                        .with(Material.metal(0.85f, 0.15f, 0.1f, 0.15f)),
                new ModelInstance(HUMAN, "Gold Figure").at(-22, 0, 4).rotated(0, 30, 0).scaled(0.5)
                        .with(Material.metal(1f, 0.78f, 0.34f, 0.02f)),
                new ModelInstance(HUMAN, "Glass Figure").at(22, 0, 4).rotated(0, -30, 0).scaled(0.5)
                        .with(Material.glass(1.5f)),
                new ModelInstance(BLOB, "Sculpture").at(-10, 2.5, 13).scaled(2.5)
                        .with(Material.diffuse(0.9f, 0.9f, 0.88f)),
                new ModelInstance(CUBE, "Cube").at(10, 2, 13).rotated(0, 35, 0).scaled(4)
                        .with(Material.metal(0.3f, 0.6f, 0.9f, 0.3f)),
                new ModelInstance(SPHERE, "Chrome Ball").at(0, 2.5, 14).scaled(2.5).with(Material.mirror())
        ), new Vec3(0, 14, 48), new Vec3(0, 5, 0), 45, new RenderSettings(true, 0f, 10, 8192));
    }
}
