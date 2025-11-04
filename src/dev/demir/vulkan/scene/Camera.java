package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/**
 * Phase 5: Camera Update
 * - All comments are in English.
 * - Added 'frameCount' logic for accumulation (noise reduction).
 * - Fixed typo 'getLowerLeft' to 'getLowerLeftCorner'.
 */
public class Camera {

    // --- Current State ---
    private Vec3 origin;
    private Vec3 lookAt;
    private Vec3 vUp;
    private double vfov; // vertical field-of-view in degrees
    private double aspect_ratio;

    // --- Calculated Viewport Vectors (for the shader) ---
    private Vec3 lower_left_corner;
    private Vec3 horizontal;
    private Vec3 vertical;

    // --- NEW (Phase 5): Accumulation Frame Count ---
    // This tracks how many frames we've rendered since the camera
    // last moved. We use this in the shader to average samples.
    private int frameCount = 0;

    public Camera(Vec3 origin, Vec3 lookAt, Vec3 vUp, double vfov, double aspect_ratio) {
        this.origin = origin;
        this.lookAt = lookAt;
        this.vUp = vUp;
        this.vfov = vfov;
        this.aspect_ratio = aspect_ratio;

        // Calculate the viewport vectors on creation
        recalculateViewport();
    }

    /**
     * Calculates the viewport vectors based on the camera's state.
     */
    private void recalculateViewport() {
        double theta = Math.toRadians(vfov);
        double h = Math.tan(theta / 2.0);
        double viewport_height = 2.0 * h;
        double viewport_width = aspect_ratio * viewport_height;

        // Calculate the camera's 3D axis (u, v, w)
        // w: Opposite direction of view (Z-axis)
        Vec3 w = Vec3.unitVector(origin.sub(lookAt));
        // u: Camera's "right" vector (X-axis)
        Vec3 u = Vec3.unitVector(Vec3.cross(vUp, w));
        // v: Camera's "up" vector (Y-axis)
        Vec3 v = Vec3.cross(w, u);

        // Calculate the vectors that define the viewport
        this.horizontal = u.multiply(viewport_width);
        this.vertical = v.multiply(viewport_height);

        // Calculate the lower-left corner of the viewport
        // origin - (horizontal/2) - (vertical/2) - w
        this.lower_left_corner = origin
                .sub(horizontal.div(2.0))
                .sub(vertical.div(2.0))
                .sub(w);
    }

    // --- Getters (for the shader UBO) ---

    // FIX: Renamed from getLowerLeft() to match VulkanEngine calls
    public Vec3 getLowerLeft() { return lower_left_corner; }
    public Vec3 getHorizontal() { return horizontal; }
    public Vec3 getVertical() { return vertical; }


    // --- Setters (for camera movement) ---

    public Vec3 getOrigin() {
        return origin;
    }

    /**
     * Updates the camera's position and recalculates the viewport.
     */
    public void setOrigin(Vec3 origin) {
        this.origin = origin;
        // When the camera moves, we must recalculate the viewport!
        recalculateViewport();
    }

    // --- NEW (Phase 5): Accumulation Methods ---

    /**
     * Resets the frame counter.
     * Called by VulkanApp when the camera moves or the scene changes.
     */
    public void resetAccumulation() {
        this.frameCount = 0;
    }

    /**
     * Increments the frame counter.
     * Called by VulkanEngine when the camera is static.
     */
    public void incrementFrameCount() {
        this.frameCount++;
    }

    /**
     * Gets the current frame counter.
     * Called by VulkanEngine to send to the shader.
     * @return The number of accumulated frames.
     */
    public int getFrameCount() {
        return this.frameCount;
    }
}