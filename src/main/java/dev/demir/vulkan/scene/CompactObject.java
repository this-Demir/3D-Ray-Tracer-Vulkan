package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * A compact object whose gravity bends light: a non-rotating (Schwarzschild) black hole, or a
 * neutron star with a glowing surface. Units: G = c = 1, so the horizon radius is 2 * mass.
 *
 * @param position       center
 * @param mass           M; horizon at 2M, photon sphere at 3M, innermost stable orbit at 6M
 * @param surfaceRadius  0 for a black hole; the star's radius (at least 3M) for a neutron star
 * @param diskInner      inner accretion disk radius (6M = innermost stable circular orbit)
 * @param diskOuter      outer disk radius
 * @param diskBrightness brightness multiplier of the disk (0 = no disk)
 * @param spinPeriod     neutron star: seconds per turn around the vertical axis (0 = no spin)
 * @param magneticTilt   neutron star: angle between the magnetic axis (hot spots) and the spin axis, degrees
 */
public record CompactObject(Vec3 position, double mass, double surfaceRadius,
                            double diskInner, double diskOuter, double diskBrightness,
                            double spinPeriod, double magneticTilt) {

    /** A black hole with an accretion disk (diskBrightness 0 = no disk). */
    public static CompactObject blackHole(Vec3 position, double mass, double diskInner, double diskOuter,
                                          double diskBrightness) {
        return new CompactObject(position, mass, 0, diskInner, diskOuter, diskBrightness, 0, 0);
    }

    /** A neutron star of the given radius, spinning, with hot spots on a tilted magnetic axis. */
    public static CompactObject neutronStar(Vec3 position, double mass, double radius, double spinPeriod,
                                            double magneticTilt) {
        return new CompactObject(position, mass, radius, 0, 0, 0, spinPeriod, magneticTilt);
    }

    public boolean isBlackHole() {
        return surfaceRadius == 0;
    }
}
