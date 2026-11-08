plugins {
    application
}

group = "dev.demir"
version = "0.1.0"

val lwjglVersion = "3.3.6"

// LWJGL ships native libraries per platform; pick the ones for the OS we build on.
val lwjglNatives = run {
    val os = System.getProperty("os.name").lowercase()
    val arch = System.getProperty("os.arch")
    val arm = arch.startsWith("aarch64") || arch.startsWith("arm")
    when {
        os.contains("win") -> if (arm) "natives-windows-arm64" else "natives-windows"
        os.contains("mac") -> if (arm) "natives-macos-arm64" else "natives-macos"
        else -> if (arm) "natives-linux-arm64" else "natives-linux"
    }
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(platform("org.lwjgl:lwjgl-bom:$lwjglVersion"))

    implementation("org.lwjgl:lwjgl")
    implementation("org.lwjgl:lwjgl-vulkan")
    implementation("org.lwjgl:lwjgl-assimp")

    runtimeOnly("org.lwjgl:lwjgl::$lwjglNatives")
    runtimeOnly("org.lwjgl:lwjgl-assimp::$lwjglNatives")
    // On Windows/Linux Vulkan comes from the GPU driver; macOS needs the bundled MoltenVK.
    if (lwjglNatives.startsWith("natives-macos")) {
        runtimeOnly("org.lwjgl:lwjgl-vulkan::$lwjglNatives")
    }
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
}

application {
    mainClass = "dev.demir.vulkan.VulkanApp"
}

tasks.named<JavaExec>("run") {
    // Shaders and models are loaded relative to the working directory (shaders_spv/, objects/).
    workingDir = rootDir
    // Forward -Dvulkan.validation=true from the Gradle command line to the app.
    System.getProperty("vulkan.validation")?.let { systemProperty("vulkan.validation", it) }
}
