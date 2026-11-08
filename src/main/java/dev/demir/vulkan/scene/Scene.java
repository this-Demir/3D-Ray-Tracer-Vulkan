package dev.demir.vulkan.scene;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Advanced Phase 1: Scene Graph System
 * This class holds the description of the scene on the CPU side.
 * It now manages a list of 'ModelInstance' objects, each with its
 * own transformation (position, scale) and material properties.
 *
 * This class is designed to be thread-safe for UI modifications
 * by using CopyOnWriteArrayList.
 */
public class Scene {

    // Use CopyOnWriteArrayList to allow the UI thread to add/remove
    // items while the SceneBuilder thread (SRT) is reading from it.
    private final List<ModelInstance> instances = new CopyOnWriteArrayList<>();

    /**
     * Adds a new ModelInstance to the scene.
     * @param instance The model instance to add.
     */
    public void addInstance(ModelInstance instance) {
        instances.add(instance);
    }

    /**
     * Removes a ModelInstance from the scene.
     * @param instance The model instance to remove.
     */
    public void removeInstance(ModelInstance instance) {
        instances.remove(instance);
    }

    /**
     * Clears all model instances from the scene.
     */
    public void clear() {
        instances.clear();
    }

    /**
     * Returns an unmodifiable view of the model instances.
     * The UI thread turns these into an immutable SceneDescription for rendering.
     * @return A list of ModelInstances.
     */
    public List<ModelInstance> getInstances() {
        return Collections.unmodifiableList(instances);
    }
}