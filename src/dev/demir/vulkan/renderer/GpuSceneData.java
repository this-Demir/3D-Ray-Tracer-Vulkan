package dev.demir.vulkan.renderer;

/**
 * POJO: Vulkan'a yüklenmiş ve render için hazır olan TÜM sahne verisi
 * tamponlarının (buffer) tanıtıcılarını (handle) tutar.
 * Bu nesne, VRT (Vulkan Render Thread) üzerinde yaşar.
 */
public class GpuSceneData {

    public final long triangleBuffer;
    public final long triangleBufferMemory;
    public final long materialBuffer;
    public final long materialBufferMemory;
    public final long bvhBuffer;
    public final long bvhBufferMemory;
    public final int triangleCount;

    public GpuSceneData(long triangleBuffer, long triangleBufferMemory,
                        long materialBuffer, long materialBufferMemory,
                        long bvhBuffer, long bvhBufferMemory, int triangleCount) {
        this.triangleBuffer = triangleBuffer;
        this.triangleBufferMemory = triangleBufferMemory;
        this.materialBuffer = materialBuffer;
        this.materialBufferMemory = materialBufferMemory;
        this.bvhBuffer = bvhBuffer;
        this.bvhBufferMemory = bvhBufferMemory;
        this.triangleCount = triangleCount;
    }
}