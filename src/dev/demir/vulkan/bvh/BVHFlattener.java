package dev.demir.vulkan.bvh;

import dev.demir.vulkan.scene.Hittable;
import dev.demir.vulkan.scene.Triangle;
import dev.demir.vulkan.util.Vec3;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Implements the "Flattener" logic from the project report.
 *
 */
public class BVHFlattener {


    public static final int GPU_NODE_SIZE = 48; // bytes

    // This list will hold the triangles *in the order* the GPU needs to read them.
    public final List<Triangle> flattenedTriangles;
    private int currentNodeIndex;

    public BVHFlattener() {
        this.flattenedTriangles = new ArrayList<>();
        this.currentNodeIndex = 0;
    }

    public ByteBuffer flatten(BVHNode root) {
        if (root == null) {
            throw new RuntimeException("BVH Root node is null before flattening!");
        }
        int nodeCount = countNodes(root);
        int bufferSize = nodeCount * GPU_NODE_SIZE;

        System.out.println("LOG: Flattening BVH tree. Total nodes: " + nodeCount);
        System.out.println("LOG: Allocating BVH Buffer (bytes): " + bufferSize);

        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN);

        flattenRecursive(root, buffer);

        buffer.position(bufferSize);
        buffer.flip();

        return buffer;
    }


    private int flattenRecursive(Hittable node, ByteBuffer buffer) {
        int myIndex = currentNodeIndex++;
        int bufferPos = myIndex * GPU_NODE_SIZE;

        AABB bbox = node.boundingBox();
        if (bbox == null) {
            throw new RuntimeException("Null bounding box encountered during flattening.");
        }

        // 1. Write Bounding Box (min/max)
        writeVec3(buffer, bufferPos + 0,  bbox.min);
        writeVec3(buffer, bufferPos + 16, bbox.max);

        if (node instanceof BVHNode) {
            // ---  (Internal Node) ---
            BVHNode bvh = (BVHNode) node;

            int leftChildIndex = flattenRecursive(bvh.left, buffer);

            int rightChildIndex = flattenRecursive(bvh.right, buffer);

            // Write data for an INTERNAL node:
            buffer.putInt(bufferPos + 32, leftChildIndex);   // data = leftChildIndex
            buffer.putInt(bufferPos + 36, rightChildIndex);  // count = rightChildIndex

        } else {
            // --- Leaf Node ---
            Triangle tri = (Triangle) node;

            int triangleDataIndex = flattenedTriangles.size();
            flattenedTriangles.add(tri);


            // Write data for a LEAF node:
            buffer.putInt(bufferPos + 32, -(triangleDataIndex + 1)); // data = -(index + 1)
            buffer.putInt(bufferPos + 36, -1); // count = -1 (Yaprak olduğunu belirtir)
        }

        return myIndex;
    }

    private void writeVec3(ByteBuffer buffer, int offset, Vec3 v) {
        buffer.putFloat(offset + 0, (float) v.x);
        buffer.putFloat(offset + 4, (float) v.y);
        buffer.putFloat(offset + 8, (float) v.z);
        buffer.putFloat(offset + 12, 0.0f); // std430 padding
    }

    private int countNodes(Hittable node) {
        if (node instanceof BVHNode) {
            BVHNode bvh = (BVHNode) node;
            return 1 + countNodes(bvh.left) + countNodes(bvh.right);
        } else {
            return 1;
        }
    }
}
