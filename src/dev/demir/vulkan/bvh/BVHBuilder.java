package dev.demir.vulkan.bvh;

import dev.demir.vulkan.scene.Hittable;
import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This class contains the logic for building the BVH acceleration structure.
 * It implements the recursive partitioning algorithm described in
 * 'raytracer-java/core/BVHNode.java'.
 *
 * It takes a list of 'Hittable' objects (Triangles) and
 * returns a single 'BVHNode' which is the root of the tree.
 */
public class BVHBuilder {

    /**
     * Public entry point for building the BVH tree.
     * @param sceneObjects The complete list of objects (Triangles) in the scene.
     * @return The root node of the constructed BVH tree.
     */
    public static BVHNode build(List<Hittable> sceneObjects) {
        if (sceneObjects == null || sceneObjects.isEmpty()) {
            throw new IllegalArgumentException("Cannot build BVH from empty object list.");
        }

        System.out.println("LOG: Starting BVH build for " + sceneObjects.size() + " objects...");

        // We must work on a mutable copy of the list to sort it
        List<Hittable> mutableObjects = new ArrayList<>(sceneObjects);

        long startTime = System.nanoTime();
        BVHNode root = buildRecursive(mutableObjects, 0, mutableObjects.size());
        long endTime = System.nanoTime();

        double durationMs = (endTime - startTime) / 1_000_000.0;
        System.out.println(String.format("LOG: BVH build finished in %.2f ms.", durationMs));

        return root;
    }

    /**
     * The core recursive build algorithm.
     * Ported from the constructor logic of 'raytracer-java/core/BVHNode.java'.
     */
    private static BVHNode buildRecursive(List<Hittable> objects, int start, int end) {

        int n = end - start;

        // 1. Choose a random axis to sort on (0=x, 1=y, 2=z)
        int axis = ThreadLocalRandom.current().nextInt(0, 3);
        Comparator<Hittable> comparator = getComparator(axis);

        Hittable left;
        Hittable right;

        // 2. Handle base cases (leaf nodes)
        if (n == 1) {
            // If only one object, it's a leaf node
            left = right = objects.get(start);
        } else if (n == 2) {
            // If two objects, compare and split them
            if (comparator.compare(objects.get(start), objects.get(start + 1)) < 0) {
                left = objects.get(start);
                right = objects.get(start + 1);
            } else {
                left = objects.get(start + 1);
                right = objects.get(start);
            }
        } else {
            // 3. Recursive case: Sort the sub-list and split in the middle
            objects.subList(start, end).sort(comparator);

            int mid = start + n / 2;
            left = buildRecursive(objects, start, mid);
            right = buildRecursive(objects, mid, end);
        }

        // 4. Calculate the bounding box for THIS new node
        AABB boxLeft = left.boundingBox();
        AABB boxRight = right.boundingBox();

        if (boxLeft == null || boxRight == null) {
            throw new RuntimeException("Child node has null bounding box during BVH construction.");
        }

        AABB bbox = AABB.surroundingBox(boxLeft, boxRight);

        // 5. Create the new node
        return new BVHNode(left, right, bbox);
    }

    /**
     * Returns a comparator for sorting Hittables along a specific axis.
     */
    private static Comparator<Hittable> getComparator(int axis) {
        return (a, b) -> {
            AABB boxA = a.boundingBox();
            AABB boxB = b.boundingBox();

            // Calculate center of each bounding box
            double centerA = (boxA.min.get(axis) + boxA.max.get(axis)) / 2.0;
            double centerB = (boxB.min.get(axis) + boxB.max.get(axis)) / 2.0;

            return Double.compare(centerA, centerB);
        };
    }
}