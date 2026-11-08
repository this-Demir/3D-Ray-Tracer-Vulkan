# Vulkan Ray Tracer: Development Log

This project is a real-time ray tracer written in Java using Vulkan (via LWJGL) and a Java Swing UI. The sections below go from newest to oldest; the older ones are kept as the history of the project.

## Current Status: Hardware Ray Tracing (November 2026)

The renderer was rebuilt around the GPU's ray tracing hardware (RT cores). In order:

1. **Runs out of the box:** validation layers are optional (`-Dvulkan.validation=true`), GLFW was removed (not needed for a compute-only renderer), and the LWJGL stack size was raised so instance creation no longer overflows on drivers with many extensions.
2. **Cleanup:** removed an unused older renderer copy, unused classes, shaders and libraries (about 1800 lines).
3. **Gradle build:** LWJGL comes from Maven Central; shaders are compiled from `src/main/shaders` with shaderc during the build (`./gradlew run`).
4. **Hardware ray tracing:** the CPU-built BVH was replaced by Vulkan acceleration structures (one BLAS per mesh, one TLAS over the instances) traced with `rayQueryEXT` from the compute shader. GPU time per frame dropped from 29 ms to 0.6 ms (~49k triangles) and from 1129 ms to 0.7 ms (~1M triangles).
5. **Frame pipeline:** two frames in flight, cached readback, BGRA output copied straight into the Swing image, GPU timestamp queries: about 2000 rendered frames per second on the default scene.
6. **Analytic spheres** (single-AABB BLAS, exact intersection in the shader) and **path tracing**: progressive accumulation, matte / metal / glass / emissive materials, direct light sampling with shadow rays towards spherical lights, Russian roulette, ACES tone mapping.
7. **Editor and scenes:** new side panel (scene settings, object pool, object list, object settings), mouse orbit / zoom camera, and the presets Custom, Mirror Room, Eclipse and ShowRoom.
8. **Animation:** objects can orbit other objects and spin; the TLAS is rebuilt on the GPU inside every frame.
9. **Black hole:** camera rays follow exact Schwarzschild light paths (RK4 on dx/dλ = v, dv/dλ = -3Mh²x/r⁵), each step traced as a ray query; thin accretion disk from 6M with gravitational and Doppler shift. A CPU twin of the integrator is unit-tested against exact results, and the rendered shadow measures 171.0 px for a predicted 171.4 px. Details: [docs/black-hole.md](docs/black-hole.md).

### Next

A denoiser so that moving cameras and animations look clean sooner, presenting frames through a Vulkan swapchain instead of the Swing image, and a rotating (Kerr) black hole.

## Earlier Status: Live Viewer with a CPU-built BVH

The project has evolved from a "headless" renderer into a "smart," multi-threaded, real-time viewer with a stable Swing (`JFrame`) UI.

* **High Performance:** Renders complex models (e.g., 49,000+ triangles) at 60+ FPS using an `O(log n)` BVH acceleration structure.
* **Dynamic Camera:** Features a fully dynamic camera controlled by the keyboard (WASD), sending updates to the shader via a Uniform Buffer Object (UBO).
* **Stable Multi-Threaded Architecture:** The application is split into three decoupled threads for maximum performance and UI responsiveness:
    1.  **UI Thread (Swing EDT):** Manages the `JFrame` and user input. It never freezes.
    2.  **VRT (Vulkan Engine Thread):** A dedicated thread running the `VulkanEngine` class, which manages all Vulkan calls and the render loop.
    3.  **SRT (Scene Rebuild Thread):** A background thread (`CompletableFuture`) that runs the `SceneBuilder` to perform CPU-heavy tasks (like BVH construction) without blocking the UI.

## Earlier Plan: Interactive Scene Editor (done, see Current Status)

The next goal is to evolve this viewer into a fully interactive scene editor, similar to the `raytracer-java` project, by implementing a path tracing shader and a scene management system.

### Roadmap at the time (done)

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

## Roadmap of the JavaFX plan (done)

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


