package dev.demir.vulkan.scene;

/**
 * Global render settings, sent from the UI to the render thread. Changing any of them restarts accumulation.
 *
 * @param skyEnabled  sky light on (true) or a black background (false)
 * @param exposure    exposure in stops (EV); the image is scaled by 2^exposure
 * @param maxBounces  maximum path length; mirror rooms need many bounces
 * @param maxSamples  stop rendering after this many accumulated samples per pixel (0 = never stop)
 * @param animate     play the scene's animations (orbits, spins)
 * @param gravity     black hole scenes: bend light rays (true) or trace straight lines (false)
 */
public record RenderSettings(boolean skyEnabled, float exposure, int maxBounces, int maxSamples,
                             boolean animate, boolean gravity) {

    /** Stops after 8192 samples: the image has converged and the GPU can rest. */
    public static final RenderSettings DEFAULT = new RenderSettings(true, 0f, 10, 8192, false, true);

    public RenderSettings withSky(boolean sky) {
        return new RenderSettings(sky, exposure, maxBounces, maxSamples, animate, gravity);
    }

    public RenderSettings withExposure(float ev) {
        return new RenderSettings(skyEnabled, ev, maxBounces, maxSamples, animate, gravity);
    }

    public RenderSettings withMaxBounces(int bounces) {
        return new RenderSettings(skyEnabled, exposure, bounces, maxSamples, animate, gravity);
    }

    public RenderSettings withMaxSamples(int samples) {
        return new RenderSettings(skyEnabled, exposure, maxBounces, samples, animate, gravity);
    }

    public RenderSettings withAnimate(boolean play) {
        return new RenderSettings(skyEnabled, exposure, maxBounces, maxSamples, play, gravity);
    }

    public RenderSettings withGravity(boolean bend) {
        return new RenderSettings(skyEnabled, exposure, maxBounces, maxSamples, animate, bend);
    }
}
