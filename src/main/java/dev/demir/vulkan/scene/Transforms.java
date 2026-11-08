package dev.demir.vulkan.scene;

import dev.demir.vulkan.util.Vec3;

/** Builds the row-major 3x4 object-to-world matrices used for TLAS instances. */
public final class Transforms {

    private Transforms() {
    }

    /**
     * M = Translate(position) * Rz * Ry * Rx * Scale, rotation in degrees.
     */
    public static float[] compose(Vec3 position, Vec3 rotationDegrees, Vec3 scale) {
        double ax = Math.toRadians(rotationDegrees.x), ay = Math.toRadians(rotationDegrees.y), az = Math.toRadians(rotationDegrees.z);
        double cx = Math.cos(ax), sx = Math.sin(ax);
        double cy = Math.cos(ay), sy = Math.sin(ay);
        double cz = Math.cos(az), sz = Math.sin(az);
        double[][] r = { // Rz * Ry * Rx
                {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx},
                {sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx},
                {-sy,     cy * sx,                cy * cx}
        };
        double[] s = {scale.x, scale.y, scale.z};
        double[] t = {position.x, position.y, position.z};
        float[] m = new float[12];
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                m[row * 4 + col] = (float) (r[row][col] * s[col]);
            }
            m[row * 4 + 3] = (float) t[row];
        }
        return m;
    }
}
