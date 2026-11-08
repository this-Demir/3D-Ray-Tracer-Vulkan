package dev.demir.vulkan.tools;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.lwjgl.util.shaderc.Shaderc.*;

/**
 * Build-time tool (run by the Gradle task {@code compileShaders}).
 * Compiles every GLSL shader in the source directory to SPIR-V for Vulkan 1.2,
 * writing {@code <name>.<stage>.spv} into the output directory. Fails the build on any error.
 *
 * Usage: ShaderCompiler <sourceDir> <outputDir>
 */
public final class ShaderCompiler {

    private static final Map<String, Integer> STAGES = Map.of(
            "comp", shaderc_glsl_compute_shader
    );

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: ShaderCompiler <sourceDir> <outputDir>");
        }
        Path sourceDir = Path.of(args[0]);
        Path outputDir = Path.of(args[1]);
        Files.createDirectories(outputDir);

        List<Path> shaders;
        try (Stream<Path> files = Files.list(sourceDir)) {
            shaders = files.filter(p -> STAGES.containsKey(extension(p))).sorted().toList();
        }

        long compiler = shaderc_compiler_initialize();
        long options = shaderc_compile_options_initialize();
        if (compiler == 0 || options == 0) {
            throw new IllegalStateException("Failed to initialize shaderc.");
        }
        try {
            shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
            shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_5);
            shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);

            int failures = 0;
            for (Path shader : shaders) {
                failures += compile(compiler, options, shader, outputDir) ? 0 : 1;
            }
            if (failures > 0) {
                throw new IllegalStateException(failures + " shader(s) failed to compile.");
            }
        } finally {
            shaderc_compile_options_release(options);
            shaderc_compiler_release(compiler);
        }
    }

    private static boolean compile(long compiler, long options, Path shader, Path outputDir) throws IOException {
        String source = Files.readString(shader);
        String fileName = shader.getFileName().toString();
        long result = shaderc_compile_into_spv(compiler, source, STAGES.get(extension(shader)), fileName, "main", options);
        if (result == 0) {
            throw new IllegalStateException("shaderc returned no result for " + fileName);
        }
        try {
            if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
                System.err.println("ERROR compiling " + fileName + ":\n" + shaderc_result_get_error_message(result));
                return false;
            }
            long warnings = shaderc_result_get_num_warnings(result);
            if (warnings > 0) {
                System.err.println("WARN " + fileName + ":\n" + shaderc_result_get_error_message(result));
            }
            ByteBuffer spirv = shaderc_result_get_bytes(result);
            Path out = outputDir.resolve(fileName + ".spv");
            try (FileChannel channel = FileChannel.open(out, StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                while (spirv.hasRemaining()) {
                    channel.write(spirv);
                }
            }
            System.out.println("Compiled " + fileName + " -> " + out.getFileName() + " (" + Files.size(out) + " bytes)");
            return true;
        } finally {
            shaderc_result_release(result);
        }
    }

    private static String extension(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot + 1);
    }
}
