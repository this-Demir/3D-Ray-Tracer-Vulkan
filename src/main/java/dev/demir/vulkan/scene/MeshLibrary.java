package dev.demir.vulkan.scene;

import org.lwjgl.assimp.AIFace;
import org.lwjgl.assimp.AIMesh;
import org.lwjgl.assimp.AIScene;
import org.lwjgl.assimp.AIVector3D;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.lwjgl.assimp.Assimp.*;

/**
 * Loads model files with Assimp into indexed {@link MeshData} and caches them by path,
 * so every model file is parsed once no matter how many instances use it or how often
 * the scene is edited. Safe to call from any thread (used by the scene rebuild thread).
 */
public final class MeshLibrary {

    private final Map<String, MeshData> cache = new ConcurrentHashMap<>();

    /**
     * Returns the mesh for the given file, loading it on first use.
     * @throws UncheckedIOException if the file cannot be loaded
     */
    public MeshData get(String path) {
        return cache.computeIfAbsent(path, p -> {
            try {
                return load(p);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    private static MeshData load(String path) throws IOException {
        long start = System.nanoTime();
        AIScene scene = aiImportFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
        if (scene == null || (scene.mFlags() & AI_SCENE_FLAGS_INCOMPLETE) != 0 || scene.mRootNode() == null) {
            if (scene != null) aiReleaseImport(scene);
            throw new IOException("Failed to load model " + path + ": " + aiGetErrorString());
        }
        try {
            // All meshes of the file are merged into one vertex/index list.
            int totalVertices = 0;
            int maxIndices = 0;
            for (int m = 0; m < scene.mNumMeshes(); m++) {
                AIMesh mesh = AIMesh.create(scene.mMeshes().get(m));
                totalVertices += mesh.mNumVertices();
                maxIndices += mesh.mNumFaces() * 3;
            }
            float[] positions = new float[totalVertices * 3];
            int[] indices = new int[maxIndices];
            int vertexBase = 0;
            int indexCount = 0;
            for (int m = 0; m < scene.mNumMeshes(); m++) {
                AIMesh mesh = AIMesh.create(scene.mMeshes().get(m));
                AIVector3D.Buffer vertices = mesh.mVertices();
                for (int v = 0; v < mesh.mNumVertices(); v++) {
                    AIVector3D vertex = vertices.get(v);
                    positions[(vertexBase + v) * 3] = vertex.x();
                    positions[(vertexBase + v) * 3 + 1] = vertex.y();
                    positions[(vertexBase + v) * 3 + 2] = vertex.z();
                }
                AIFace.Buffer faces = mesh.mFaces();
                for (int f = 0; f < mesh.mNumFaces(); f++) {
                    AIFace face = faces.get(f);
                    if (face.mNumIndices() != 3) continue; // points/lines left over after triangulation
                    indices[indexCount++] = vertexBase + face.mIndices().get(0);
                    indices[indexCount++] = vertexBase + face.mIndices().get(1);
                    indices[indexCount++] = vertexBase + face.mIndices().get(2);
                }
                vertexBase += mesh.mNumVertices();
            }
            MeshData data = new MeshData(path, positions, Arrays.copyOf(indices, indexCount));
            System.out.printf("LOG (SRT): Loaded %s: %,d vertices, %,d triangles in %.1f ms%n",
                    path, data.vertexCount(), data.triangleCount(), (System.nanoTime() - start) / 1e6);
            return data;
        } finally {
            aiReleaseImport(scene);
        }
    }
}
