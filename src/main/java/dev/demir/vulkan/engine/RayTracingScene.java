package dev.demir.vulkan.engine;

import dev.demir.vulkan.scene.Geometry;
import dev.demir.vulkan.scene.Material;
import dev.demir.vulkan.scene.MeshData;
import dev.demir.vulkan.scene.SceneDescription;
import dev.demir.vulkan.scene.SphereGeometry;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static dev.demir.vulkan.engine.GpuMemory.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.VK10.*;

/**
 * GPU side of the scene for hardware ray tracing.
 *
 * - One bottom-level acceleration structure (BLAS) per distinct geometry, built once and
 *   reused by every instance and every scene edit: triangle meshes from VRAM vertex/index
 *   buffers, analytic spheres as a single AABB that the shader intersects exactly.
 * - One top-level acceleration structure (TLAS) over all instances. It is built when the
 *   scene changes, and for animated scenes rebuilt inside each frame's command buffer from
 *   that frame slot's instance buffer (only transforms change, so this is cheap).
 * - An instance-info buffer (mesh buffer addresses, material) and, per frame slot, a light
 *   buffer with the emissive spheres the shader samples directly.
 *
 * Not thread-safe; used only on the Vulkan render thread.
 */
public final class RayTracingScene {

    /** Bytes per entry in the instance-info buffer; must match InstanceInfo in pathtrace.comp. */
    public static final int INSTANCE_INFO_SIZE = 64;
    /** Bytes per entry in the light buffer; must match Light in pathtrace.comp. */
    public static final int LIGHT_SIZE = 48;

    /** Geometry types; must match GEOMETRY_* in pathtrace.comp. */
    public static final int GEOMETRY_TRIANGLES = 0;
    public static final int GEOMETRY_SPHERE = 1;

    private static final int GEOMETRY_INPUT_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    private static final int HOST_VISIBLE = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    private record AccelerationStructure(long handle, GpuBuffer storage, long deviceAddress) {
    }

    /** A geometry on the GPU. Triangle meshes have vertices/indices; spheres have aabbs. */
    private record GpuGeometry(int type, GpuBuffer vertices, GpuBuffer indices, GpuBuffer aabbs,
                               AccelerationStructure blas) {
    }

    private final VkDevice device;
    private final GpuMemory memory;
    private final long scratchAlignment;
    private final int frameSlots;

    private final Map<String, GpuGeometry> geometries = new HashMap<>();
    private SceneDescription scene;
    private AccelerationStructure tlas;
    private GpuBuffer tlasScratch;        // kept for in-frame TLAS rebuilds
    private long tlasScratchAddress;
    private final GpuBuffer[] tlasInstances; // per slot: VkAccelerationStructureInstanceKHR[]
    private final GpuBuffer[] lights;        // per slot: Light[]
    private GpuBuffer instanceInfo;          // InstanceInfo[] (does not change with time)
    private int[] lightInstances = new int[0]; // instance index of each light

    public RayTracingScene(VkDevice device, GpuMemory memory, int frameSlots) {
        this.device = device;
        this.memory = memory;
        this.frameSlots = frameSlots;
        this.tlasInstances = new GpuBuffer[frameSlots];
        this.lights = new GpuBuffer[frameSlots];
        try (MemoryStack stack = stackPush()) {
            VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties =
                    VkPhysicalDeviceAccelerationStructurePropertiesKHR.calloc(stack).sType$Default();
            VkPhysicalDeviceProperties2 properties2 = VkPhysicalDeviceProperties2.calloc(stack)
                    .sType$Default()
                    .pNext(asProperties.address());
            VK11.vkGetPhysicalDeviceProperties2(device.getPhysicalDevice(), properties2);
            scratchAlignment = Math.max(1, asProperties.minAccelerationStructureScratchOffsetAlignment());
        }
    }

    /**
     * Makes the GPU scene match the description (GPU must be idle): builds BLASes for new
     * geometries, frees unused ones, and builds the TLAS for the given time.
     */
    public void update(SceneDescription newScene, double time) {
        long start = System.nanoTime();
        int newGeometries = 0;
        Set<String> used = new HashSet<>();
        for (SceneDescription.Instance instance : newScene.instances()) {
            Geometry geometry = instance.geometry();
            used.add(geometry.key());
            if (!geometries.containsKey(geometry.key())) {
                geometries.put(geometry.key(), upload(geometry));
                newGeometries++;
            }
        }
        long blasDone = System.nanoTime();

        for (Iterator<Map.Entry<String, GpuGeometry>> it = geometries.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, GpuGeometry> entry = it.next();
            if (!used.contains(entry.getKey())) {
                destroyGeometry(entry.getValue());
                it.remove();
            }
        }

        destroyTopLevel();
        scene = newScene;
        buildTopLevel(newScene.transformsAt(time));
        System.out.printf("LOG (VRT-RT): Scene updated: %d instance(s), %,d triangles, %d geometr%s on GPU "
                        + "(%d new BLAS in %.1f ms, TLAS in %.1f ms)%s%n",
                newScene.instances().size(), newScene.triangleCount(), geometries.size(),
                geometries.size() == 1 ? "y" : "ies",
                newGeometries, (blasDone - start) / 1e6, (System.nanoTime() - blasDone) / 1e6,
                newScene.isAnimated() ? ", animated" : "");
    }

    public boolean isAnimated() {
        return scene != null && scene.isAnimated();
    }

    /** The scene's black hole or neutron star, or null. */
    public dev.demir.vulkan.scene.CompactObject compactObject() {
        return scene == null ? null : scene.compactObject();
    }

    /**
     * Animated scenes: writes the transforms for 'time' into this slot's instance and light
     * buffers (the slot's previous frame has finished) and records a TLAS rebuild into cmd,
     * with barriers so the previous frame's rays finish first and this frame's rays see it.
     */
    public void recordAnimationUpdate(VkCommandBuffer cmd, int slot, double time) {
        float[][] transforms = scene.transformsAt(time);
        writeSlot(slot, transforms);
        try (MemoryStack stack = stackPush()) {
            memoryBarrier(stack, cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

            VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo = topLevelBuildInfo(stack, slot);
            buildInfo.get(0).dstAccelerationStructure(tlas.handle());
            buildInfo.get(0).scratchData().deviceAddress(tlasScratchAddress);
            VkAccelerationStructureBuildRangeInfoKHR.Buffer range = VkAccelerationStructureBuildRangeInfoKHR
                    .calloc(1, stack).primitiveCount(scene.instances().size());
            vkCmdBuildAccelerationStructuresKHR(cmd, buildInfo, stack.pointers(range.address()));

            memoryBarrier(stack, cmd,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        }
    }

    /** Writes only this slot's light buffer for 'time' (when the TLAS was already rebuilt for it). */
    public void updateSlotLights(int slot, double time) {
        writeLights(slot, scene.transformsAt(time));
    }

    public long topLevelAccelerationStructure() {
        return tlas.handle();
    }

    public GpuBuffer instanceInfoBuffer() {
        return instanceInfo;
    }

    public GpuBuffer lightBuffer(int slot) {
        return lights[slot];
    }

    /** Number of emissive spheres the shader samples directly. */
    public int lightCount() {
        return lightInstances.length;
    }

    public void destroy() {
        destroyTopLevel();
        geometries.values().forEach(this::destroyGeometry);
        geometries.clear();
    }

    // --- Geometry / BLAS ---

    private GpuGeometry upload(Geometry geometry) {
        return switch (geometry) {
            case MeshData mesh -> uploadMesh(mesh);
            case SphereGeometry sphere -> uploadSphere();
        };
    }

    private GpuGeometry uploadMesh(MeshData mesh) {
        ByteBuffer vertexBytes = memAlloc(mesh.positions().length * Float.BYTES);
        ByteBuffer indexBytes = memAlloc(mesh.indices().length * Integer.BYTES);
        GpuBuffer vertices;
        GpuBuffer indices;
        try {
            vertexBytes.asFloatBuffer().put(mesh.positions());
            indexBytes.asIntBuffer().put(mesh.indices());
            vertices = memory.createDeviceLocalBuffer(vertexBytes, GEOMETRY_INPUT_USAGE);
            indices = memory.createDeviceLocalBuffer(indexBytes, GEOMETRY_INPUT_USAGE);
        } finally {
            memFree(vertexBytes);
            memFree(indexBytes);
        }

        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geometry = VkAccelerationStructureGeometryKHR.calloc(1, stack);
            geometry.get(0)
                    .sType$Default()
                    .geometryType(VK_GEOMETRY_TYPE_TRIANGLES_KHR)
                    .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
            VkAccelerationStructureGeometryTrianglesDataKHR triangles = geometry.get(0).geometry().triangles();
            triangles.sType$Default()
                    .vertexFormat(VK_FORMAT_R32G32B32_SFLOAT)
                    .vertexStride(3 * Float.BYTES)
                    .maxVertex(Math.max(0, mesh.vertexCount() - 1))
                    .indexType(VK_INDEX_TYPE_UINT32);
            triangles.vertexData().deviceAddress(vertices.deviceAddress());
            triangles.indexData().deviceAddress(indices.deviceAddress());

            AccelerationStructure blas = buildBlocking(stack, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    geometry, mesh.triangleCount(), false);
            return new GpuGeometry(GEOMETRY_TRIANGLES, vertices, indices, null, blas);
        }
    }

    /** Unit sphere: one AABB [-1, 1]^3; the shader computes the exact hit inside it. */
    private GpuGeometry uploadSphere() {
        ByteBuffer box = memAlloc(VkAabbPositionsKHR.SIZEOF);
        GpuBuffer aabbs;
        try {
            box.putFloat(0, -1f).putFloat(4, -1f).putFloat(8, -1f)
                    .putFloat(12, 1f).putFloat(16, 1f).putFloat(20, 1f);
            aabbs = memory.createDeviceLocalBuffer(box, GEOMETRY_INPUT_USAGE);
        } finally {
            memFree(box);
        }

        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geometry = VkAccelerationStructureGeometryKHR.calloc(1, stack);
            geometry.get(0)
                    .sType$Default()
                    .geometryType(VK_GEOMETRY_TYPE_AABBS_KHR)
                    .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
            VkAccelerationStructureGeometryAabbsDataKHR aabbData = geometry.get(0).geometry().aabbs();
            aabbData.sType$Default().stride(VkAabbPositionsKHR.SIZEOF);
            aabbData.data().deviceAddress(aabbs.deviceAddress());

            AccelerationStructure blas = buildBlocking(stack, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    geometry, 1, false);
            return new GpuGeometry(GEOMETRY_SPHERE, null, null, aabbs, blas);
        }
    }

    private void destroyGeometry(GpuGeometry geometry) {
        destroy(geometry.blas());
        memory.destroy(geometry.vertices());
        memory.destroy(geometry.indices());
        memory.destroy(geometry.aabbs());
    }

    // --- Instances / TLAS ---

    private void buildTopLevel(float[][] transforms) {
        int count = scene.instances().size();
        int capacity = Math.max(1, count); // Vulkan buffers cannot be empty

        List<Integer> lightList = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            if (isSphereLight(scene.instances().get(i))) lightList.add(i);
        }
        lightInstances = lightList.stream().mapToInt(Integer::intValue).toArray();

        instanceInfo = memory.createBuffer((long) capacity * INSTANCE_INFO_SIZE,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        ByteBuffer info = instanceInfo.mapped();
        for (int i = 0; i < count; i++) {
            SceneDescription.Instance instance = scene.instances().get(i);
            writeInstanceInfo(info, i * INSTANCE_INFO_SIZE, geometries.get(instance.geometry().key()), instance.material());
        }

        for (int slot = 0; slot < frameSlots; slot++) {
            tlasInstances[slot] = memory.createBuffer((long) capacity * VkAccelerationStructureInstanceKHR.SIZEOF,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                            | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    HOST_VISIBLE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            lights[slot] = memory.createBuffer((long) Math.max(1, lightInstances.length) * LIGHT_SIZE,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            writeSlot(slot, transforms);
        }

        try (MemoryStack stack = stackPush()) {
            tlas = buildBlocking(stack, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                    topLevelGeometry(stack, 0), count, true);
        }
    }

    /** Fills a slot's TLAS instance records and light buffer for the given transforms. */
    private void writeSlot(int slot, float[][] transforms) {
        int count = scene.instances().size();
        VkAccelerationStructureInstanceKHR.Buffer vkInstances =
                VkAccelerationStructureInstanceKHR.create(tlasInstances[slot].mappedAddress(), Math.max(1, count));
        for (int i = 0; i < count; i++) {
            SceneDescription.Instance instance = scene.instances().get(i);
            VkAccelerationStructureInstanceKHR vkInstance = vkInstances.get(i);
            for (int k = 0; k < 12; k++) {
                vkInstance.transform().matrix(k, transforms[i][k]);
            }
            vkInstance.instanceCustomIndex(i) // index into instanceInfo in the shader
                    .mask(0xFF)
                    .instanceShaderBindingTableRecordOffset(0)
                    .flags(VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)
                    .accelerationStructureReference(geometries.get(instance.geometry().key()).blas().deviceAddress());
        }
        writeLights(slot, transforms);
    }

    /**
     * Light layout (std430, 48 bytes):
     *   0  vec4 center.xyz, radius      16 vec4 emission.rgb, unused
     *  32  uint instance (to recognize the light in a shadow ray)   36..47 padding
     * Only spheres become lights; emissive meshes still glow when a ray hits them.
     */
    private void writeLights(int slot, float[][] transforms) {
        ByteBuffer data = lights[slot].mapped();
        for (int light = 0; light < lightInstances.length; light++) {
            int i = lightInstances[light];
            float[] m = transforms[i];
            float radius = 0f; // largest axis scale (spheres are expected to be scaled uniformly)
            for (int col = 0; col < 3; col++) {
                radius = Math.max(radius, (float) Math.sqrt(m[col] * m[col] + m[4 + col] * m[4 + col] + m[8 + col] * m[8 + col]));
            }
            Material material = scene.instances().get(i).material();
            int base = light * LIGHT_SIZE;
            data.putFloat(base, m[3]).putFloat(base + 4, m[7]).putFloat(base + 8, m[11]).putFloat(base + 12, radius);
            data.putFloat(base + 16, material.albedo()[0] * material.emission());
            data.putFloat(base + 20, material.albedo()[1] * material.emission());
            data.putFloat(base + 24, material.albedo()[2] * material.emission());
            data.putFloat(base + 28, 0f);
            data.putInt(base + 32, i);
        }
    }

    private static boolean isSphereLight(SceneDescription.Instance instance) {
        return instance.geometry() instanceof SphereGeometry
                && instance.material().type() == Material.Type.EMISSIVE
                && instance.material().emission() > 0f;
    }

    /**
     * InstanceInfo layout (std430, 64 bytes):
     *   0  uint64 vertexAddress     8 uint64 indexAddress
     *  16  vec4   albedo.rgb, roughness
     *  32  vec4   emission.rgb (albedo * strength), ior
     *  48  uint   materialType     52 uint geometryType    56..63 padding
     */
    private static void writeInstanceInfo(ByteBuffer info, int base, GpuGeometry geometry, Material material) {
        float[] albedo = material.albedo();
        float emission = material.type() == Material.Type.EMISSIVE ? material.emission() : 0f;
        info.putLong(base, geometry.vertices() != null ? geometry.vertices().deviceAddress() : 0L);
        info.putLong(base + 8, geometry.indices() != null ? geometry.indices().deviceAddress() : 0L);
        info.putFloat(base + 16, albedo[0]);
        info.putFloat(base + 20, albedo[1]);
        info.putFloat(base + 24, albedo[2]);
        info.putFloat(base + 28, material.roughness());
        info.putFloat(base + 32, albedo[0] * emission);
        info.putFloat(base + 36, albedo[1] * emission);
        info.putFloat(base + 40, albedo[2] * emission);
        info.putFloat(base + 44, material.ior());
        info.putInt(base + 48, material.type().ordinal());
        info.putInt(base + 52, geometry.type());
        info.putInt(base + 56, 0);
        info.putInt(base + 60, 0);
    }

    private VkAccelerationStructureGeometryKHR.Buffer topLevelGeometry(MemoryStack stack, int slot) {
        VkAccelerationStructureGeometryKHR.Buffer geometry = VkAccelerationStructureGeometryKHR.calloc(1, stack);
        geometry.get(0)
                .sType$Default()
                .geometryType(VK_GEOMETRY_TYPE_INSTANCES_KHR)
                .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
        VkAccelerationStructureGeometryInstancesDataKHR instancesData = geometry.get(0).geometry().instances();
        instancesData.sType$Default().arrayOfPointers(false);
        instancesData.data().deviceAddress(tlasInstances[slot].deviceAddress());
        return geometry;
    }

    private VkAccelerationStructureBuildGeometryInfoKHR.Buffer topLevelBuildInfo(MemoryStack stack, int slot) {
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo =
                VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
        buildInfo.get(0)
                .sType$Default()
                .type(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
                .flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
                .geometryCount(1)
                .pGeometries(topLevelGeometry(stack, slot));
        return buildInfo;
    }

    private void destroyTopLevel() {
        if (tlas != null) {
            destroy(tlas);
            tlas = null;
        }
        memory.destroy(tlasScratch);
        tlasScratch = null;
        memory.destroy(instanceInfo);
        instanceInfo = null;
        for (int slot = 0; slot < frameSlots; slot++) {
            memory.destroy(tlasInstances[slot]);
            memory.destroy(lights[slot]);
            tlasInstances[slot] = null;
            lights[slot] = null;
        }
        lightInstances = new int[0];
    }

    // --- Shared build code ---

    /**
     * Creates an acceleration structure of the given type and builds it on the GPU (blocking).
     * With keepScratch, the scratch buffer is kept for later in-frame rebuilds of the TLAS.
     */
    private AccelerationStructure buildBlocking(MemoryStack stack, int type,
                                                VkAccelerationStructureGeometryKHR.Buffer geometry,
                                                int primitiveCount, boolean keepScratch) {
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo =
                VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
        buildInfo.get(0)
                .sType$Default()
                .type(type)
                .flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
                .geometryCount(1)
                .pGeometries(geometry);

        VkAccelerationStructureBuildSizesInfoKHR sizes = VkAccelerationStructureBuildSizesInfoKHR.calloc(stack).sType$Default();
        vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                buildInfo.get(0), stack.ints(primitiveCount), sizes);

        GpuBuffer storage = memory.createBuffer(sizes.accelerationStructureSize(),
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        VkAccelerationStructureCreateInfoKHR createInfo = VkAccelerationStructureCreateInfoKHR.calloc(stack)
                .sType$Default()
                .buffer(storage.buffer())
                .size(sizes.accelerationStructureSize())
                .type(type);
        LongBuffer pHandle = stack.mallocLong(1);
        check(vkCreateAccelerationStructureKHR(device, createInfo, null, pHandle), "vkCreateAccelerationStructureKHR");
        long handle = pHandle.get(0);

        GpuBuffer scratch = memory.createBuffer(sizes.buildScratchSize() + scratchAlignment,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        long scratchAddress = (scratch.deviceAddress() + scratchAlignment - 1) / scratchAlignment * scratchAlignment;
        try {
            buildInfo.get(0).dstAccelerationStructure(handle);
            buildInfo.get(0).scratchData().deviceAddress(scratchAddress);

            VkAccelerationStructureBuildRangeInfoKHR.Buffer range =
                    VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack).primitiveCount(primitiveCount);
            PointerBuffer ranges = stack.pointers(range.address());
            memory.submitAndWait(cmd -> vkCmdBuildAccelerationStructuresKHR(cmd, buildInfo, ranges));
        } finally {
            if (keepScratch) {
                tlasScratch = scratch;
                tlasScratchAddress = scratchAddress;
            } else {
                memory.destroy(scratch);
            }
        }

        long deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device,
                VkAccelerationStructureDeviceAddressInfoKHR.calloc(stack).sType$Default().accelerationStructure(handle));
        return new AccelerationStructure(handle, storage, deviceAddress);
    }

    private static void memoryBarrier(MemoryStack stack, VkCommandBuffer cmd, int srcStage, int srcAccess,
                                      int dstStage, int dstAccess) {
        VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack)
                .sType$Default()
                .srcAccessMask(srcAccess)
                .dstAccessMask(dstAccess);
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, barrier, null, null);
    }

    private void destroy(AccelerationStructure as) {
        vkDestroyAccelerationStructureKHR(device, as.handle(), null);
        memory.destroy(as.storage());
    }
}
