package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * A non-rotating (Schwarzschild) black hole with an optional thin accretion disk in its
 * horizontal plane. Units: G = c = 1, so the horizon radius is 2 * mass (scene units).
 *
 * @param position       center of the black hole
 * @param mass           M; horizon at 2M, photon sphere at 3M, innermost stable orbit at 6M
 * @param diskInner      inner disk radius (6M = innermost stable circular orbit)
 * @param diskOuter      outer disk radius
 * @param diskBrightness brightness multiplier of the disk (0 = no disk)
 */
public record BlackHole(Vec3 position, double mass, double diskInner, double diskOuter, double diskBrightness) {

    public double horizonRadius() {
        return 2 * mass;
    }
}
