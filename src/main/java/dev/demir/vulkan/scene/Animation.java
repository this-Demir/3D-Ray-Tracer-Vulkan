package dev.demir.vulkan.scene;

/**
 * Simple motion for an object.
 *
 * Orbit: the object circles around a center (another object, which may itself be moving, or
 * the world origin) in the horizontal plane. The radius, height and starting angle are taken
 * from where the object is placed relative to that center, so a scene can be laid out at
 * time 0 (for example with a moon exactly between the sun and a planet) and then set in motion.
 *
 * @param orbitCenter name of the object to orbit, {@link #WORLD_ORIGIN}, or null for no orbit
 * @param orbitPeriod seconds per full orbit (0 = no orbit)
 * @param spinPeriod  seconds per full turn around the object's own vertical axis (0 = no spin)
 */
public record Animation(String orbitCenter, double orbitPeriod, double spinPeriod) {

    public static final String WORLD_ORIGIN = "(world origin)";
    public static final Animation NONE = new Animation(null, 0, 0);

    public boolean orbits() {
        return orbitCenter != null && orbitPeriod != 0;
    }

    public boolean spins() {
        return spinPeriod != 0;
    }

    public boolean isStatic() {
        return !orbits() && !spins();
    }
}
