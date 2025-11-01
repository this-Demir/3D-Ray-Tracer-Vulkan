package dev.demir.vulkan.renderer;

import dev.demir.vulkan.bvh.BVHBuilder;
import dev.demir.vulkan.bvh.BVHFlattener;
import dev.demir.vulkan.bvh.BVHNode;
import dev.demir.vulkan.scene.Hittable;
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
 * "Phase 1: Refactor Engine"
 * Bu sınıf, SRT (Scene Rebuild Thread) üzerinde çalışmak üzere tasarlanmıştır.
 * CPU-yoğun O(n log n) işlemler olan model yükleme, BVH inşası ve
 * BVH düzleştirme işlemlerini yapar.
 *
 * Sonuç olarak, VRT'nin GPU'ya yüklemesi için hazır
 * CPU-tarafı tamponlarını içeren bir 'BuiltCpuData' nesnesi döndürür.
 */
public class SceneBuilder {

    /**
     * Sahne inşa işlemini yürüten ana metod.
     * Bu metodun tamamı bir arka plan iş parçacığında (SRT) çalıştırılmalıdır.
     *
     * @param modelPath Yüklenecek .obj modelinin dosya yolu.
     * @return GPU'ya yüklenmeye hazır veriyi içeren BuiltCpuData nesnesi.
     */
    public BuiltCpuData buildScene(String modelPath) {
        System.out.println("LOG (SRT): Starting scene build for: " + modelPath);

        // 1. Modeli yükle ve CPU-tarafı Üçgen listesini oluştur
        List<Hittable> hittableTriangles = loadModel(modelPath);

        if (hittableTriangles.isEmpty()) {
            throw new RuntimeException("Model loaded, but 0 triangles found.");
        }

        // 2. BVH Ağacını CPU'da inşa et
        // (VulkanApp.java:237'den  [cite: 3, 237-238] alınan mantık)
        BVHNode bvhRootNode = BVHBuilder.build(hittableTriangles);

        // 3. BVH'yi düzleştir ve GPU tamponları için verileri hazırla
        // (VulkanApp.java:246'dan  alınan mantık)
        System.out.println("LOG (SRT): Flattening BVH tree for GPU...");
        BVHFlattener flattener = new BVHFlattener();

        // 3a. Düzleştirme işlemini çalıştır
        ByteBuffer flatBvhData = flattener.flatten(bvhRootNode);

        // 3b. Düzleştiriciden *yeniden sıralanmış* üçgen listesini al
        List<Triangle> orderedTriangles = flattener.flattenedTriangles;
        int numFlattenedTriangles = orderedTriangles.size();

        System.out.println("LOG (SRT): BVH flattened. Triangle list re-ordered.");

        // 3c. GPU tamponlarını bu *yeni sıralanmış* listeye göre doldur
        FloatBuffer modelVertexData = memAllocFloat(numFlattenedTriangles * 3 * 4);
        FloatBuffer modelMaterialData = memAllocFloat(numFlattenedTriangles * 4);

        for (Triangle tri : orderedTriangles) {
            // Vertex verisini ekle (v0, v1, v2)
            modelVertexData.put((float) tri.v0.x).put((float) tri.v0.y).put((float) tri.v0.z).put(0.0f); // pad
            modelVertexData.put((float) tri.v1.x).put((float) tri.v1.y).put((float) tri.v1.z).put(0.0f); // pad
            modelVertexData.put((float) tri.v2.x).put((float) tri.v2.y).put((float) tri.v2.z).put(0.0f); // pad

            // Materyal verisini ekle
            modelMaterialData.put(tri.r).put(tri.g).put(tri.b).put(tri.materialType);
        }

        modelVertexData.flip();
        modelMaterialData.flip();

        System.out.println("LOG (SRT): Vertex, Material, and BVH buffers are ready for upload.");

        // 4. Hazırlanan veriyi tek bir pakette döndür
        return new BuiltCpuData(
                modelVertexData,
                modelMaterialData,
                flatBvhData,
                numFlattenedTriangles
        );
    }

    /**
     * Assimp kullanarak bir modeli yükler ve onu 'Hittable' (Triangle)
     * nesnelerinden oluşan bir listeye dönüştürür.
     * (VulkanApp.java:169'dan  uyarlanmıştır)
     */
    private List<Hittable> loadModel(String modelPath) {
        System.out.println("LOG (SRT): Loading model: " + modelPath);
        List<Hittable> hittableTriangles = new ArrayList<>();
        int triangleCount = 0;

        // TODO: Materyal seçimi de dinamik olmalı
        float r = 0.8f, g = 0.8f, b = 0.8f; // Varsayılan renk
        float matType = 0.0f; // Varsayılan (Lambertian)

        try (MemoryStack stack = stackPush()) {
            AIScene scene = aiImportFile(modelPath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
            if (scene == null || (scene.mFlags() & AI_SCENE_FLAGS_INCOMPLETE) != 0 || scene.mRootNode() == null) {
                throw new RuntimeException("Failed to load model: " + aiGetErrorString());
            }

            int meshCount = scene.mNumMeshes();
            System.out.println("LOG (SRT): Model contains " + meshCount + " mesh(es).");

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

                    int materialIndex = triangleCount;
                    int vertexIndex = triangleCount * 3;

                    hittableTriangles.add(new Triangle(v0, v1, v2, materialIndex, vertexIndex, r, g, b, matType));
                    triangleCount++;
                }
            }
            aiReleaseImport(scene);
            System.out.println("LOG (SRT): Model loaded. Total triangles: " + triangleCount);
            return hittableTriangles;
        } catch (Exception e) {
            throw new RuntimeException("Failed during model loading: " + e.getMessage(), e);
        }
    }
}