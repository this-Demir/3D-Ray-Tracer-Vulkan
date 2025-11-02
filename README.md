# Vulkan BVH Ray Tracer

This project is a high-performance, real-time ray tracer written in Java using Vulkan (via LWJGL) and a Java Swing UI. It uses a Bounding Volume Hierarchy (BVH) built on the CPU to achieve dynamic `O(log n)` rendering performance on the GPU.

## Current Status (Live Viewer)

The project has evolved from a "headless" renderer into a "smart," multi-threaded, real-time viewer with a stable Swing (`JFrame`) UI.

* **High Performance:** Renders complex models (e.g., 49,000+ triangles) at 60+ FPS using an `O(log n)` BVH acceleration structure.
* **Dynamic Camera:** Features a fully dynamic camera controlled by the keyboard (WASD), sending updates to the shader via a Uniform Buffer Object (UBO).
* **Stable Multi-Threaded Architecture:** The application is split into three decoupled threads for maximum performance and UI responsiveness:
    1.  **UI Thread (Swing EDT):** Manages the `JFrame` and user input. It never freezes.
    2.  **VRT (Vulkan Engine Thread):** A dedicated thread running the `VulkanEngine` class, which manages all Vulkan calls and the render loop.
    3.  **SRT (Scene Rebuild Thread):** A background thread (`CompletableFuture`) that runs the `SceneBuilder` to perform CPU-heavy tasks (like BVH construction) without blocking the UI.

## Future Plan: Interactive Scene Editor

The next goal is to evolve this viewer into a fully interactive scene editor, similar to the `raytracer-java` project, by implementing a path tracing shader and a scene management system.

### Current Development Roadmap

1.  **Phase 1: Scene Graph System (CPU)**
    * Create a `Scene.java` class to manage a dynamic list of objects (`Meshes`, `Spheres`, etc.) instead of a single static model.
    * Update `SceneBuilder.java` to consume this `Scene` object and build a single, unified BVH for all objects.
    * Implement Swing UI controls to add/remove objects and trigger a scene rebuild (`rebuildSceneAsync`).

2.  **Phase 2: Path Tracing Shader (GPU)**
    * Rewrite the `compute_dynamic.comp` shader to be a full, multi-bounce path tracer (porting logic from `raytracer-java`'s `PathTracer.java`).
    * Implement a Random Number Generator (RNG) inside the shader.
    * Implement a sky gradient background when a ray hits nothing.

3.  **Phase 3: Material System (CPU + GPU)**
    * Expand the `Material` struct on the GPU and CPU to support different types (Lambertian, Metal, Dielectric).
    * Implement `scatter` logic in the shader to handle realistic reflections and refractions (mirrors, glass).

4.  **Phase 4: Emissive Lighting (CPU + GPU)**
    * Add an `Emissive` material type.
    * Update the shader so that rays hitting an emissive object stop bouncing and return light, turning those objects into light sources.
    * Add UI controls to add/move light sources (which are just objects with an emissive material) and rebuild the scene. 



## Old Status (Headless Engine)

The project is currently a "headless" single-shot renderer (`VulkanApp.java`). In its present state, it successfully:

1.  Loads a 3D model (`.obj`) from disk.
2.  Builds a Bounding Volume Hierarchy (BVH) on the CPU, a one-time O(n log n) task.
3.  "Flattens" the BVH tree and uploads all scene geometry, materials, and the BVH structure to GPU-exclusive memory.
4.  Executes a Vulkan compute shader (`compute.comp`) that traces rays against the BVH, achieving O(log n) average-case complexity per ray.
5.  Waits for the GPU to finish, copies the resulting image back from VRAM, and saves it to a PNG file before exiting.

##  OLD Future Plan: Dynamic JavaFX Editor : DONE

The next goal is to transform this powerful headless engine into a fully dynamic, real-time scene editor with a JavaFX UI.

To achieve a smooth, 60 FPS user experience, the application **must never block the JavaFX Application Thread (FXAT)**. This requires a multi-threaded architecture that separates UI logic, rendering, and scene construction.

This architecture will be built on three distinct thread types:

### 1. The FXAT (JavaFX Application Thread)

* **Role:** Manages the JavaFX UI, including the `AnimationTimer`, all buttons, sliders, and keyboard/mouse input.
* **Policy:** This thread **never** performs heavy work. In its 60 FPS `AnimationTimer` loop, it only does two things:
    1.  Grabs the *latest completed frame* from a thread-safe queue.
    2.  Displays that frame in a JavaFX `ImageView`.
* When a user moves the camera or adds an object, the FXAT simply sends a non-blocking message to the `RenderService` (our "CPU Manager") and continues its loop.

### 2. The VRT (Vulkan Render Thread)

* **Role:** This is a dedicated, long-running `Thread` that runs the `RenderLoop`.
* **Policy:** This is the **only thread in the entire application allowed to make Vulkan API calls** (`vkQueueSubmit`, `vkWaitForFences`, `vkCreateBuffer`, etc.).
    * It runs in a continuous loop, producing frames as fast as the GPU will allow.
    * After rendering a frame, it copies the image from VRAM to a `ByteBuffer` in RAM.
    * It places this `ByteBuffer` into a thread-safe `AtomicReference` for the FXAT to pick up.
    * It also polls its own thread-safe queues for new camera data or new scene data to upload.

### 3. The SRT (Scene Rebuild Thread)

* **Role:** A *temporary* background thread (e.g., a `CompletableFuture`) that is launched only when the scene is modified (e.g., "Add Sphere" is clicked).
* **Policy:** This thread performs the slow, **CPU-only** O(n log n) tasks:
    1.  Runs the `BVHBuilder.build()` algorithm.
    2.  Runs the `BVHFlattener.flatten()` algorithm.
    3.  Prepares all scene data (vertices, materials, BVH) in CPU-side `ByteBuffer`s and `FloatBuffer`s.
* It **does not** make any Vulkan calls. When finished, it hands its prepared CPU buffers to the `RenderService`, which safely queues them for the **VRT** to upload.

### Thread-Safe Data Flow (The "CPU Manager" Model)

This entire process is managed by a central `RenderService` class to ensure thread safety.

1.  **Camera Move (Fast Path):**
    * `FXAT` (Key press) -> `RenderService.updateCamera()`
    * `RenderService` -> `VRT.cameraQueue.add(newCam)`
    * `VRT` (next loop) -> `renderer.updateCameraUBO(newCam)`

2.  **Scene Edit (Slow Path):**
    * `FXAT` (Button click) -> `RenderService.rebuildScene()`
    * `RenderService` -> Launches `SRT` with a snapshot of the scene.
    * *(...FXAT and VRT continue running, rendering the OLD scene...)*
    * `SRT` (finishes) -> `RenderService.onSceneBuilt(cpuData)`
    * `RenderService` -> `VRT.sceneSwapQueue.add(cpuData)`
    * `VRT` (next loop) -> `vkDeviceWaitIdle()`, destroys old GPU buffers, uploads new `cpuData`, and proceeds.
    * *(The VRT now renders the NEW scene. The UI never froze.)*

## Development Roadmap

1.  **Phase 1: Refactor Engine**
    * Convert `VulkanApp.java` into a reusable `VulkanRenderer.java` class with `init()`, `renderFrame(): ByteBuffer`, and `destroy()` methods.
2.  **Phase 2: JavaFX Wrapper**
    * Create the `MainApp` (JavaFX) and `RenderLoop` (VRT).
    * Get the `ByteBuffer` from the `VRT` and display it in a JavaFX `ImageView` using an `AnimationTimer` on the `FXAT`.
3.  **Phase 3: The "CPU Manager"**
    * Create the `RenderService` to manage the threads.
    * Create `SceneBuilder.java` to house the `BVHBuilder` and `BVHFlattener` logic.
    * Implement the full, thread-safe "Scene Edit" data flow.
4.  **Phase 4: Features**
    * Implement a Vulkan Uniform Buffer Object (UBO) for a dynamic camera.
    * Build the UI controls (sliders, buttons) to add objects and modify materials.
    * Add debug features (e.g., "show ray" on mouse click).


