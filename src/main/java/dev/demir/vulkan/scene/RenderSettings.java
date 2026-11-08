package dev.demir.vulkan.scene;

/**
 * Global render settings, sent from the UI to the render thread. Changing any of them restarts accumulation.
 *
 * @param skyEnabled  sky light on (true) or a black background (false)
 * @param exposure    exposure in stops (EV); the image is scaled by 2^exposure
 * @param maxBounces  maximum path length; mirror rooms need many bounces
 * @param maxSamples  stop rendering after this many accumulated samples per pixel (0 = never stop)
 * @param animate     play the scene's animations (orbits, spins)
 */
public record RenderSettings(boolean skyEnabled, float exposure, int maxBounces, int maxSamples, boolean animate) {

    /** Stops after 8192 samples: the image has converged and the GPU can rest. */
    public static final RenderSettings DEFAULT = new RenderSettings(true, 0f, 10, 8192, false);

    public RenderSettings withSky(boolean sky) {
        return new RenderSettings(sky, exposure, maxBounces, maxSamples, animate);
    }

    public RenderSettings withExposure(float ev) {
        return new RenderSettings(skyEnabled, ev, maxBounces, maxSamples, animate);
    }

    public RenderSettings withMaxBounces(int bounces) {
        return new RenderSettings(skyEnabled, exposure, bounces, maxSamples, animate);
    }

    public RenderSettings withMaxSamples(int samples) {
        return new RenderSettings(skyEnabled, exposure, maxBounces, samples, animate);
    }

    public RenderSettings withAnimate(boolean play) {
        return new RenderSettings(skyEnabled, exposure, maxBounces, maxSamples, play);
    }
}
