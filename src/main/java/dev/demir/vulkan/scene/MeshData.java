package dev.demir.vulkan.scene;

/**
 * CPU-side indexed triangle mesh, in object space.
 * Immutable once loaded; shared by every instance that uses the same model file.
 *
 * @param path      file the mesh was loaded from (also its cache key)
 * @param positions x, y, z per vertex
 * @param indices   three vertex indices per triangle
 */
public record MeshData(String path, float[] positions, int[] indices) implements Geometry {

    @Override
    public String key() {
        return path;
    }

    public int vertexCount() {
        return positions.length / 3;
    }

    @Override
    public int triangleCount() {
        return indices.length / 3;
    }
}
