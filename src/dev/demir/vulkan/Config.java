package dev.demir.vulkan;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Properties;

/**
 * A simple utility class to read configuration from the .env file.
 */
public class Config {
    private static final Properties props = new Properties();

    static {
        try (BufferedReader reader = new BufferedReader(new FileReader(".env"))) {
            props.load(reader);
            System.out.println("LOG: .env file loaded successfully.");
        } catch (Exception e) {
            System.err.println("WARN: Could not read .env file. Using default settings.");
            // Set defaults if .env is missing
            props.setProperty("ENABLE_VALIDATION_LAYERS", "1");
            props.setProperty("SHADER_PATH", "shaders_spv/");
        }
    }

    public static String getString(String key, String defaultValue) {
        return props.getProperty(key, defaultValue);
    }

    public static boolean getBoolean(String key, boolean defaultValue) {
        String val = props.getProperty(key, String.valueOf(defaultValue));
        return val.equals("1") || val.equalsIgnoreCase("true");
    }
}