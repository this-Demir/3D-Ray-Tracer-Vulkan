package dev.demir.vulkan.renderer;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * POJO: SRT (Scene Rebuild Thread) tarafından CPU'da hazırlanan ve
 * VRT'ye (Vulkan Render Thread) yüklenmek üzere gönderilen veriyi tutar.
 */
public class BuiltCpuData {

    public final FloatBuffer modelVertexData;
    public final FloatBuffer modelMaterialData;
    public final ByteBuffer flatBvhData;
    public final int triangleCount;

    public BuiltCpuData(FloatBuffer modelVertexData, FloatBuffer modelMaterialData,
                        ByteBuffer flatBvhData, int triangleCount) {
        this.modelVertexData = modelVertexData;
        this.modelMaterialData = modelMaterialData;
        this.flatBvhData = flatBvhData;
        this.triangleCount = triangleCount;
    }
}