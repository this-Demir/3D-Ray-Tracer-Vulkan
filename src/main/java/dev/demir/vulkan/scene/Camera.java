package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Pinhole camera, owned and mutated by the UI thread only.
 * The render thread receives immutable {@link CameraState} snapshots via {@link #state()}.
 */
public class Camera {

    private Vec3 origin;
    private Vec3 lookAt;
    private final Vec3 vUp;
    private double vfov; // vertical field-of-view in degrees
    private final double aspectRatio;

    public Camera(Vec3 origin, Vec3 lookAt, Vec3 vUp, double vfov, double aspectRatio) {
        this.origin = origin;
        this.lookAt = lookAt;
        this.vUp = vUp;
        this.vfov = vfov;
        this.aspectRatio = aspectRatio;
    }

    /** Computes the viewport vectors the shader needs for the current camera. */
    public CameraState state() {
        double theta = Math.toRadians(vfov);
        double viewportHeight = 2.0 * Math.tan(theta / 2.0);
        double viewportWidth = aspectRatio * viewportHeight;

        Vec3 w = Vec3.unitVector(origin.sub(lookAt));    // backwards
        Vec3 u = Vec3.unitVector(Vec3.cross(vUp, w));    // right
        Vec3 v = Vec3.cross(w, u);                        // up

        Vec3 horizontal = u.multiply(viewportWidth);
        Vec3 vertical = v.multiply(viewportHeight);
        Vec3 lowerLeft = origin.sub(horizontal.div(2.0)).sub(vertical.div(2.0)).sub(w);
        return new CameraState(origin, lowerLeft, horizontal, vertical);
    }

    public Vec3 getOrigin() { return origin; }
    public Vec3 getLookAt() { return lookAt; }
    public double getVfov() { return vfov; }

    public void setOrigin(Vec3 origin) { this.origin = origin; }
    public void setLookAt(Vec3 lookAt) { this.lookAt = lookAt; }
    public void setVfov(double vfov) { this.vfov = vfov; }
}
