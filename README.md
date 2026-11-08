# Gpu Based 3D Ray Tracer 

![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)
![Vulkan](https://img.shields.io/badge/Vulkan-%23AC162C.svg?style=for-the-badge&logo=vulkan&logoColor=white)

**Author:** Demir Demirdöğen

> **Work in Progress:** This project is actively under development. Features, render techniques, and UI elements are subject to change as new capabilities (like Path Tracing) are implemented.

A high-performance, real-time **GPU Ray Tracer** written in Java using **LWJGL** and **Vulkan**.

This project, developed as a **Software Engineering Student Project**, demonstrates a robust, multi-threaded architecture capable of rendering complex 3D scenes. It utilizes a custom CPU-built **Bounding Volume Hierarchy (BVH)** to achieve O(log n) ray intersection on the GPU, allowing for real-time performance even on integrated graphics.

### [➔Click here to visit the Render Gallery ](GALLERY.md)


![Ray Tracer Demo](render_images/car-render-ss.png)
##  Key Features

* **Real-Time GPU Rendering:** Uses Vulkan Compute Shaders (`.comp`) for massive parallel ray tracing.
* **BVH Acceleration:** Implements a Linear BVH structure built on the CPU and flattened for GPU traversal, allowing rendering of meshes with 50k+ triangles efficiently.
* **Multi-Threaded Architecture:**
    * **UI Thread:** Responsive Swing Interface (never freezes).
    * **Vulkan Render Thread (VRT):** Dedicated rendering loop managing command queues.
    * **Scene Rebuild Thread (SRT):** Asynchronous BVH construction for non-blocking scene editing.
* **Interactive Editor:**
    * Add/Remove `.obj` models dynamically.
    * Modify object **Position**, **Scale**, and **Color** in real-time.
    * Change Material properties (**Matte**, **Metal**, **Emissive**).
* **Dynamic Camera:** Smooth WASD controls with Uniform Buffer Object (UBO) updates.

##  System & Performance

This engine is highly optimized to run efficiently without requiring high-end dedicated hardware.

**Tested System Specs:**
* **OS:** Windows 11 Home Single Language 64-bit
* **CPU:** 13th Gen Intel(R) Core(TM) i9-13900H
* **GPU:** Intel(R) Iris(R) Xe Graphics (Integrated)
* **RAM:** 16GB

> **Performance Note:** Despite running on an integrated GPU (no external graphics card), the application maintains a stable **~40 FPS** in complex scenes. This demonstrates the efficiency of the BVH acceleration structure and memory management.

##  Architecture Overview

The application is built on a "Decoupled Rendering" model to ensure the UI remains responsive even during heavy computation like BVH construction.

```mermaid
graph TD
    User[User Input] --> UI[Swing UI Thread]
    UI -- "Camera Updates" --> Q1[Camera Queue]
    UI -- "Scene Changes" --> SRT[Scene Rebuild Thread]
    SRT -- "Built BVH & Buffers" --> Q2[Scene Queue]
    
    subgraph VRT [Vulkan Render Thread]
        Loop[Render Loop]
        Q1 --> Loop
        Q2 --> Loop
        Loop --> GPU[Vulkan Compute Shader]
    end
    
    GPU -- "Rendered Frame" --> UI
```

##  Controls

### Camera Movement
* **W / S:** Move Forward / Backward
* **A / D:** Move Left / Right
* **Q / E:** Move Up / Down

### UI Panel
* **Object List:** Select objects to modify properties.
* **Add Model:** Import standard `.obj` files from disk.
* **Global Settings:** Toggle Sky Light, Adjust Exposure.
* **Material Editor:** Switch between Lambertian, Metallic, or Emissive materials.

##  Technical Details

* **Language:** Java 21
* **Graphics API:** Vulkan 1.0+ compute (via LWJGL 3.3.6)
* **Shading Language:** GLSL 4.50
* **Acceleration Structure:** Linear Bounding Volume Hierarchy (LBVH) built on CPU.
* **Synchronization:** Explicit memory barriers and layout transitions (General <-> Transfer) to manage resource synchronization between Compute and Transfer queues.

##  Build & Run

### Prerequisites
* **Java:** JDK 21.
* **GPU:** Hardware ray tracing through Vulkan 1.2 (`VK_KHR_acceleration_structure` + `VK_KHR_ray_query`), e.g. NVIDIA RTX 20 series or newer, AMD RX 6000 / RDNA2 or newer, Intel Arc. The Vulkan runtime comes with the GPU driver; the Vulkan SDK is only needed for validation layers.
* **Build Tool:** None to install. The project uses the Gradle wrapper, which downloads Gradle and LWJGL on the first build.

### Running
```bash
./gradlew run          # Windows (cmd/PowerShell): gradlew run
```

Shaders live in `src/main/shaders/` and are compiled to SPIR-V during the build (Gradle task `compileShaders`, using shaderc through LWJGL), so no Vulkan SDK or `glslc` is needed. The default scene uses models from `objects/`.
To turn on Vulkan validation (requires the Vulkan SDK), run `./gradlew run -Dvulkan.validation=true`.

To get a standalone build: `./gradlew installDist`, then start `build/install/ray-tracer-vulkan/bin/ray-tracer-vulkan` from the repository root (shaders and models are loaded relative to the working directory).

In IntelliJ IDEA, open the folder and let it import the Gradle project.

##  Development Log

For a detailed history of the project's evolution, technical hurdles, and future roadmap (Path Tracing implementation), please see the [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).

*Powered by LWJGL and Vulkan.*

