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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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
 * - One top-level acceleration structure (TLAS) over all instances, rebuilt on each
 *   scene update (cheap: it only holds transforms and BLAS references).
 * - An instance-info buffer the shader indexes with the instance's custom index, holding
 *   the mesh buffer addresses and the instance's material.
 *
 * Not thread-safe; used only on the Vulkan render thread, while the GPU is idle.
 */
public final class RayTracingScene {

    /** Bytes per entry in the instance-info buffer; must match InstanceInfo in pathtrace.comp. */
    public static final int INSTANCE_INFO_SIZE = 64;

    /** Geometry types; must match GEOMETRY_* in pathtrace.comp. */
    public static final int GEOMETRY_TRIANGLES = 0;
    public static final int GEOMETRY_SPHERE = 1;

    private static final int GEOMETRY_INPUT_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    private record AccelerationStructure(long handle, GpuBuffer storage, long deviceAddress) {
    }

    /** A geometry on the GPU. Triangle meshes have vertices/indices; spheres have aabbs. */
    private record GpuGeometry(int type, GpuBuffer vertices, GpuBuffer indices, GpuBuffer aabbs,
                               AccelerationStructure blas) {
    }

    private final VkDevice device;
    private final GpuMemory memory;
    private final long scratchAlignment;

    private final Map<String, GpuGeometry> geometries = new HashMap<>();
    private AccelerationStructure tlas;
    private GpuBuffer tlasInstances; // VkAccelerationStructureInstanceKHR[] read by the TLAS build
    private GpuBuffer instanceInfo;  // InstanceInfo[] read by the shader

    public RayTracingScene(VkDevice device, GpuMemory memory) {
        this.device = device;
        this.memory = memory;
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
     * Makes the GPU scene match the description: builds BLASes for new geometries, frees
     * geometries no instance uses any more, and rebuilds the TLAS and instance-info buffer.
     */
    public void update(SceneDescription scene) {
        long start = System.nanoTime();
        int newGeometries = 0;
        Set<String> used = new HashSet<>();
        for (SceneDescription.Instance instance : scene.instances()) {
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
        buildTopLevel(scene);
        System.out.printf("LOG (VRT-RT): Scene updated: %d instance(s), %,d triangles, %d geometr%s on GPU "
                        + "(%d new BLAS in %.1f ms, TLAS in %.1f ms)%n",
                scene.instances().size(), scene.triangleCount(), geometries.size(),
                geometries.size() == 1 ? "y" : "ies",
                newGeometries, (blasDone - start) / 1e6, (System.nanoTime() - blasDone) / 1e6);
    }

    public long topLevelAccelerationStructure() {
        return tlas.handle();
    }

    public GpuBuffer instanceInfoBuffer() {
        return instanceInfo;
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

            AccelerationStructure blas = build(stack, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    geometry, mesh.triangleCount());
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

            AccelerationStructure blas = build(stack, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry, 1);
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

    private void buildTopLevel(SceneDescription scene) {
        int count = scene.instances().size();
        int capacity = Math.max(1, count); // Vulkan buffers cannot be empty
        int hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        tlasInstances = memory.createBuffer((long) capacity * VkAccelerationStructureInstanceKHR.SIZEOF,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                        | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                hostVisible, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        instanceInfo = memory.createBuffer((long) capacity * INSTANCE_INFO_SIZE,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkAccelerationStructureInstanceKHR.Buffer vkInstances =
                VkAccelerationStructureInstanceKHR.create(tlasInstances.mappedAddress(), capacity);
        ByteBuffer info = instanceInfo.mapped();
        for (int i = 0; i < count; i++) {
            SceneDescription.Instance instance = scene.instances().get(i);
            GpuGeometry geometry = geometries.get(instance.geometry().key());

            VkAccelerationStructureInstanceKHR vkInstance = vkInstances.get(i);
            float[] transform = instance.transform();
            for (int k = 0; k < 12; k++) {
                vkInstance.transform().matrix(k, transform[k]);
            }
            vkInstance.instanceCustomIndex(i) // index into instanceInfo in the shader
                    .mask(0xFF)
                    .instanceShaderBindingTableRecordOffset(0)
                    .flags(VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)
                    .accelerationStructureReference(geometry.blas().deviceAddress());

            writeInstanceInfo(info, i * INSTANCE_INFO_SIZE, geometry, instance.material());
        }

        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geometry = VkAccelerationStructureGeometryKHR.calloc(1, stack);
            geometry.get(0)
                    .sType$Default()
                    .geometryType(VK_GEOMETRY_TYPE_INSTANCES_KHR)
                    .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
            VkAccelerationStructureGeometryInstancesDataKHR instancesData = geometry.get(0).geometry().instances();
            instancesData.sType$Default().arrayOfPointers(false);
            instancesData.data().deviceAddress(tlasInstances.deviceAddress());

            tlas = build(stack, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, geometry, count);
        }
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

    private void destroyTopLevel() {
        if (tlas != null) {
            destroy(tlas);
            tlas = null;
        }
        memory.destroy(tlasInstances);
        memory.destroy(instanceInfo);
        tlasInstances = null;
        instanceInfo = null;
    }

    // --- Shared build code ---

    /**
     * Creates an acceleration structure of the given type and builds it on the GPU (blocking).
     */
    private AccelerationStructure build(MemoryStack stack, int type,
                                        VkAccelerationStructureGeometryKHR.Buffer geometry, int primitiveCount) {
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
        try {
            long scratchAddress = (scratch.deviceAddress() + scratchAlignment - 1) / scratchAlignment * scratchAlignment;
            buildInfo.get(0).dstAccelerationStructure(handle);
            buildInfo.get(0).scratchData().deviceAddress(scratchAddress);

            VkAccelerationStructureBuildRangeInfoKHR.Buffer range =
                    VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack).primitiveCount(primitiveCount);
            PointerBuffer ranges = stack.pointers(range.address());
            memory.submitAndWait(cmd -> vkCmdBuildAccelerationStructuresKHR(cmd, buildInfo, ranges));
        } finally {
            memory.destroy(scratch);
        }

        long deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device,
                VkAccelerationStructureDeviceAddressInfoKHR.calloc(stack).sType$Default().accelerationStructure(handle));
        return new AccelerationStructure(handle, storage, deviceAddress);
    }

    private void destroy(AccelerationStructure as) {
        vkDestroyAccelerationStructureKHR(device, as.handle(), null);
        memory.destroy(as.storage());
    }
}
