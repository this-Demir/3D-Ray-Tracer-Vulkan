package dev.demir.vulkan.engine;

import dev.demir.vulkan.scene.MeshData;
import dev.demir.vulkan.scene.SceneDescription;
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
 * - One bottom-level acceleration structure (BLAS) per distinct mesh, built once from
 *   VRAM vertex/index buffers and reused by every instance and every scene edit.
 * - One top-level acceleration structure (TLAS) over all instances, rebuilt on each
 *   scene update (cheap: it only holds transforms and BLAS references).
 * - An instance-info buffer the shader indexes with the instance's custom index, holding
 *   the mesh's vertex/index buffer addresses and the instance's material.
 *
 * Not thread-safe; used only on the Vulkan render thread, while the GPU is idle.
 */
public final class RayTracingScene {

    /** Bytes per entry in the instance-info buffer; must match InstanceInfo in pathtrace.comp. */
    public static final int INSTANCE_INFO_SIZE = 32;

    private static final int GEOMETRY_INPUT_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    private record AccelerationStructure(long handle, GpuBuffer storage, long deviceAddress) {
    }

    private record GpuMesh(GpuBuffer vertices, GpuBuffer indices, AccelerationStructure blas, int triangleCount) {
    }

    private final VkDevice device;
    private final GpuMemory memory;
    private final long scratchAlignment;

    private final Map<String, GpuMesh> meshes = new HashMap<>();
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
     * Makes the GPU scene match the description: builds BLASes for new meshes, frees
     * meshes no instance uses any more, and rebuilds the TLAS and instance-info buffer.
     */
    public void update(SceneDescription scene) {
        long start = System.nanoTime();
        int newMeshes = 0;
        Set<String> used = new HashSet<>();
        for (SceneDescription.Instance instance : scene.instances()) {
            MeshData mesh = instance.mesh();
            used.add(mesh.path());
            if (!meshes.containsKey(mesh.path())) {
                meshes.put(mesh.path(), uploadMesh(mesh));
                newMeshes++;
            }
        }
        long blasDone = System.nanoTime();

        for (Iterator<Map.Entry<String, GpuMesh>> it = meshes.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, GpuMesh> entry = it.next();
            if (!used.contains(entry.getKey())) {
                destroyMesh(entry.getValue());
                it.remove();
            }
        }

        destroyTopLevel();
        buildTopLevel(scene);
        System.out.printf("LOG (VRT-RT): Scene updated: %d instance(s), %,d triangles, %d mesh(es) on GPU "
                        + "(%d new BLAS in %.1f ms, TLAS in %.1f ms)%n",
                scene.instances().size(), scene.triangleCount(), meshes.size(),
                newMeshes, (blasDone - start) / 1e6, (System.nanoTime() - blasDone) / 1e6);
    }

    public long topLevelAccelerationStructure() {
        return tlas.handle();
    }

    public GpuBuffer instanceInfoBuffer() {
        return instanceInfo;
    }

    public void destroy() {
        destroyTopLevel();
        meshes.values().forEach(this::destroyMesh);
        meshes.clear();
    }

    // --- Meshes / BLAS ---

    private GpuMesh uploadMesh(MeshData mesh) {
        ByteBuffer vertexBytes = memAlloc(mesh.positions().length * Float.BYTES);
        ByteBuffer indexBytes = memAlloc(mesh.indices().length * Integer.BYTES);
        GpuBuffer vertices = null;
        GpuBuffer indices = null;
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
            return new GpuMesh(vertices, indices, blas, mesh.triangleCount());
        }
    }

    private void destroyMesh(GpuMesh mesh) {
        destroy(mesh.blas());
        memory.destroy(mesh.vertices());
        memory.destroy(mesh.indices());
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
            GpuMesh mesh = meshes.get(instance.mesh().path());

            VkAccelerationStructureInstanceKHR vkInstance = vkInstances.get(i);
            float[] transform = instance.transform();
            for (int k = 0; k < 12; k++) {
                vkInstance.transform().matrix(k, transform[k]);
            }
            vkInstance.instanceCustomIndex(i) // index into instanceInfo in the shader
                    .mask(0xFF)
                    .instanceShaderBindingTableRecordOffset(0)
                    .flags(VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)
                    .accelerationStructureReference(mesh.blas().deviceAddress());

            int base = i * INSTANCE_INFO_SIZE;
            info.putLong(base, mesh.vertices().deviceAddress());
            info.putLong(base + 8, mesh.indices().deviceAddress());
            info.putFloat(base + 16, instance.albedo()[0]);
            info.putFloat(base + 20, instance.albedo()[1]);
            info.putFloat(base + 24, instance.albedo()[2]);
            info.putFloat(base + 28, instance.materialType());
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
