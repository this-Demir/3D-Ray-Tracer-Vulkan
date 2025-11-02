package dev.demir.vulkan.renderer;

import dev.demir.vulkan.bvh.BVHBuilder;
import dev.demir.vulkan.bvh.BVHFlattener;
import dev.demir.vulkan.bvh.BVHNode;
import dev.demir.vulkan.scene.Hittable;
import dev.demir.vulkan.scene.ModelInstance; // NEW IMPORT
import dev.demir.vulkan.scene.Scene;         // NEW IMPORT
import dev.demir.vulkan.scene.Triangle;
import dev.demir.vulkan.util.Vec3;
import org.lwjgl.assimp.*;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.assimp.Assimp.*;
import static org.lwjgl.system.MemoryUtil.memAllocFloat;
import static org.lwjgl.system.MemoryStack.stackPush;

/**
 * Advanced Phase 1: Updated SceneBuilder.
 * This class now consumes a 'Scene' object containing 'ModelInstance's.
 * It applies transformations (position, scale) and materials
 * from each instance *before* building the unified BVH.
 */
public class SceneBuilder {

    /**
     * The main method that executes the scene build process.
     * This entire method should be run on a background thread (SRT).
     *
     * @param scene The 'Scene' object to build.
     * @return BuiltCpuData object containing data ready for GPU upload.
     */
    public BuiltCpuData buildScene(Scene scene) { // UPDATED: Signature now takes Scene
        System.out.println("LOG (SRT): Starting scene build...");

        // 1. Load ALL models from instances and create the CPU-side Triangle list
        List<Hittable> allHittableTriangles = new ArrayList<>();
        int totalInstances = scene.getInstances().size();
        System.out.println("LOG (SRT): Loading " + totalInstances + " model instance(s)...");

        int instanceIndex = 0;
        for (ModelInstance instance : scene.getInstances()) {
            instanceIndex++;
            System.out.println("LOG (SRT): Loading instance [" + instanceIndex + "/" + totalInstances + "]: " +
                    instance.getDisplayName() + " from " + instance.getModelPath());
            try {
                // Pass the whole instance to loadModel, where transformations
                // and materials will be applied.
                allHittableTriangles.addAll(loadModel(instance));
            } catch (Exception e) {
                System.err.println("WARN (SRT): Failed to load model " + instance.getModelPath() + ": " + e.getMessage());
                // Skip this model and continue with the rest of the scene
            }
        }

        if (allHittableTriangles.isEmpty()) {
            // This is not an error; a scene can be empty.
            System.out.println("LOG (SRT): Scene built, but 0 triangles were loaded.");
            // We must return empty (but valid) data
            return new BuiltCpuData(
                    memAllocFloat(1), // empty buffer
                    memAllocFloat(1), // empty buffer
                    ByteBuffer.allocateDirect(1), // empty buffer
                    0
            );
        }

        System.out.println("LOG (SRT): Total triangles from all instances: " + allHittableTriangles.size());

        // 2. Build the BVH Tree on the CPU (for the *combined* list)
        BVHNode bvhRootNode = BVHBuilder.build(allHittableTriangles);

        // 3. Flatten the BVH and prepare data for GPU buffers
        System.out.println("LOG (SRT): Flattening BVH tree for GPU...");
        BVHFlattener flattener = new BVHFlattener();

        // 3a. Run the flattening process
        ByteBuffer flatBvhData = flattener.flatten(bvhRootNode);

        // 3b. Get the *re-ordered* triangle list from the flattener
        List<Triangle> orderedTriangles = flattener.flattenedTriangles;
        int numFlattenedTriangles = orderedTriangles.size();

        System.out.println("LOG (SRT): BVH flattened. Triangle list re-ordered.");

        // 3c. Fill the GPU buffers according to this *newly ordered* list
        FloatBuffer modelVertexData = memAllocFloat(numFlattenedTriangles * 3 * 4);
        FloatBuffer modelMaterialData = memAllocFloat(numFlattenedTriangles * 4);

        for (Triangle tri : orderedTriangles) {
            // Add vertex data (v0, v1, v2)
            modelVertexData.put((float) tri.v0.x).put((float) tri.v0.y).put((float) tri.v0.z).put(0.0f); // pad
            modelVertexData.put((float) tri.v1.x).put((float) tri.v1.y).put((float) tri.v1.z).put(0.0f); // pad
            modelVertexData.put((float) tri.v2.x).put((float) tri.v2.y).put((float) tri.v2.z).put(0.0f); // pad

            // Add material data (Now comes from the Triangle,
            // which got it from the ModelInstance)
            modelMaterialData.put(tri.r).put(tri.g).put(tri.b).put(tri.materialType);
        }

        modelVertexData.flip();
        modelMaterialData.flip();

        System.out.println("LOG (SRT): Vertex, Material, and BVH buffers are ready for upload.");

        // 4. Return the prepared data in a single package
        return new BuiltCpuData(
                modelVertexData,
                modelMaterialData,
                flatBvhData,
                numFlattenedTriangles
        );
    }

    /**
     * Loads a model using Assimp and converts it into a list
     * of 'Hittable' (Triangle) objects.
     *
     * CRITICAL: This method now applies the instance's transformation
     * (position, scale) and material properties to each triangle.
     *
     * @param instance The ModelInstance to load.
     */
    private List<Hittable> loadModel(ModelInstance instance) {
        String modelPath = instance.getModelPath();
        // Get transformation and material data from the instance
        Vec3 position = instance.getPosition();
        Vec3 scale = instance.getScale();
        Vec3 color = instance.getColor();
        float matType = instance.getMaterialType();
        float r = (float) color.x;
        float g = (float) color.y;
        float b = (float) color.z;

        List<Hittable> hittableTriangles = new ArrayList<>();
        int triangleCount = 0;

        try (MemoryStack stack = stackPush()) {
            AIScene scene = aiImportFile(modelPath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
            if (scene == null || (scene.mFlags() & AI_SCENE_FLAGS_INCOMPLETE) != 0 || scene.mRootNode() == null) {
                throw new RuntimeException("Failed to load model: " + aiGetErrorString());
            }

            int meshCount = scene.mNumMeshes();
            for (int i = 0; i < meshCount; i++) {
                AIMesh mesh = AIMesh.create(scene.mMeshes().get(i));
                int faceCount = mesh.mNumFaces();
                AIVector3D.Buffer aiVertices = mesh.mVertices();

                for (int j = 0; j < faceCount; j++) {
                    AIFace face = mesh.mFaces().get(j);
                    if (face.mNumIndices() != 3) continue;

                    AIVector3D aiV0 = aiVertices.get(face.mIndices().get(0));
                    AIVector3D aiV1 = aiVertices.get(face.mIndices().get(1));
                    AIVector3D aiV2 = aiVertices.get(face.mIndices().get(2));

                    Vec3 v0 = new Vec3(aiV0.x(), aiV0.y(), aiV0.z());
                    Vec3 v1 = new Vec3(aiV1.x(), aiV1.y(), aiV1.z());
                    Vec3 v2 = new Vec3(aiV2.x(), aiV2.y(), aiV2.z());

                    // --- APPLY TRANSFORMATION ---
                    // 1. Apply Scale
                    // 2. Apply Rotation (TODO: Implement with matrices)
                    // 3. Apply Position (Translation)
                    // Note: Order matters! (Scale first, then translate)
                    v0 = v0.multiply(scale).add(position);
                    v1 = v1.multiply(scale).add(position);
                    v2 = v2.multiply(scale).add(position);
                    // -----------------------------

                    int materialIndex = triangleCount; // Still relative to this model
                    int vertexIndex = triangleCount * 3;

                    // Create triangle with TRANSFORMED vertices and INSTANCE materials
                    hittableTriangles.add(new Triangle(v0, v1, v2, materialIndex, vertexIndex, r, g, b, matType));
                    triangleCount++;
                }
            }
            aiReleaseImport(scene);
            // Log moved to buildScene()
            return hittableTriangles;
        } catch (Exception e) {
            throw new RuntimeException("Failed during model loading (" + modelPath + "): " + e.getMessage(), e);
        }
    }
}