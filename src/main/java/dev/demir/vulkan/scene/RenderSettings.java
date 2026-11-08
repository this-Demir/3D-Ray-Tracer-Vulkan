package dev.demir.vulkan.scene;

/**
 * Global render settings, sent from the UI to the render thread. Changing any of them restarts accumulation.
 *
 * @param skyEnabled  sky light on (true) or a black background (false)
 * @param exposure    exposure in stops (EV); the image is scaled by 2^exposure
 * @param maxBounces  maximum path length; mirror rooms need many bounces
 * @param maxSamples  stop rendering after this many accumulated samples per pixel (0 = never stop)
 * @param animate     play the scene's animations (orbits, spins)
 * @param gravity     black hole / neutron star scenes: bend light rays (true) or trace straight lines (false)
 * @param splitView   black hole / neutron star scenes: left half of the image with straight light, right half bent
 * @param pulsarBeams neutron star scenes: show the magnetosphere (radiation cones and magnetic field lines)
 * @param diagnostic  black hole / neutron star scenes: diagnostic surface (grid, regions only bent light shows) instead of the realistic look
 */
public record RenderSettings(boolean skyEnabled, float exposure, int maxBounces, int maxSamples,
                             boolean animate, boolean gravity, boolean splitView, boolean pulsarBeams,
                             boolean diagnostic) {

    /** Stops after 8192 samples: the image has converged and the GPU can rest. */
    public static final RenderSettings DEFAULT = new RenderSettings(true, 0f, 10, 8192, false, true, false, false, false);
}
